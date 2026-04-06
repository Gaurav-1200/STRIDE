from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM

from distributed_inference.model_registry import get_adapter_for_model


@dataclass
class PartitionMetadata:
    model_id: str
    dtype: str
    num_layers : int
    num_workers : int
    worker_idx: int
    layer_indices: List[int]
    has_embeddings: bool
    has_final_norm: bool
    has_lm_head: bool


class RuntimePartition(nn.Module):
    def __init__(self, partition_dir: str, device: str = "cuda"):
        super().__init__()
        self.partition_dir = Path(partition_dir)
        self.device = torch.device(device)

        with open(self.partition_dir / "meta.json", "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.meta = PartitionMetadata(**raw)

        self.dtype = getattr(torch, self.meta.dtype)
        self.config = AutoConfig.from_pretrained(self.meta.model_id, trust_remote_code=False)
        self.adapter = get_adapter_for_model(self.meta.model_id)

        base_model = AutoModelForCausalLM.from_config(self.config, trust_remote_code=False)
        self.rotary_emb = base_model.model.rotary_emb.to(self.device)
        base_model.eval()

        self.embed_tokens = None
        if self.meta.has_embeddings:
            self.embed_tokens = self.adapter.embed_tokens(base_model).to(self.device, dtype=self.dtype)

        full_layers = self.adapter.layers(base_model)
        self.layers = nn.ModuleList()
        self.layer_lookup: Dict[int, nn.Module] = {}
        for idx in self.meta.layer_indices:
            layer = full_layers[idx].to(self.device, dtype=self.dtype)
            self.layers.append(layer)
            self.layer_lookup[idx] = layer

        self.final_norm = None
        if self.meta.has_final_norm:
            self.final_norm = self.adapter.final_norm(base_model).to(self.device, dtype=self.dtype)

        self.lm_head = None
        if self.meta.has_lm_head:
            self.lm_head = self.adapter.lm_head(base_model).to(self.device, dtype=self.dtype)

        self._load_weights()

    def _load_weights(self) -> None:
        weights = load_file(str(self.partition_dir / "weights.safetensors"))

        if self.embed_tokens is not None:
            embed_sd = {k.split("embed_tokens.", 1)[1]: v for k, v in weights.items() if k.startswith("embed_tokens.")}
            self.embed_tokens.load_state_dict(embed_sd, strict=True)

        for idx in self.meta.layer_indices:
            prefix = f"layers.{idx}."
            layer_sd = {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}
            self.layer_lookup[idx].load_state_dict(layer_sd, strict=True)

        if self.final_norm is not None:
            norm_sd = {k.split("final_norm.", 1)[1]: v for k, v in weights.items() if k.startswith("final_norm.")}
            self.final_norm.load_state_dict(norm_sd, strict=True)

        if self.lm_head is not None:
            head_sd = {k.split("lm_head.", 1)[1]: v for k, v in weights.items() if k.startswith("lm_head.")}
            self.lm_head.load_state_dict(head_sd, strict=True)

    @torch.inference_mode()
    def forward_partition(
        self,
        *,
        input_ids: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_logits: bool = False,
    ):
        if hidden_states is None:
            if self.embed_tokens is None:
                raise ValueError("This partition cannot accept raw input_ids because it has no embeddings.")
            if input_ids is None:
                raise ValueError("input_ids must be provided on the first partition.")
            hidden_states = self.embed_tokens(input_ids.to(self.device))
        else:
            hidden_states = hidden_states.to(self.device, dtype=self.dtype)

        
        seq_len = hidden_states.shape[1]
        if position_ids is None:
            position_ids = torch.arange(seq_len,device=self.device).unsqueeze(0)
        else:
            position_ids = position_ids.to(self.device)
        
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        cos, sin = self.rotary_emb(hidden_states, position_ids)

        for idx in self.meta.layer_indices:
            layer = self.layer_lookup[idx]
            # 3. Pass position_embeddings to the layer
            out = layer(
                hidden_states, 
                attention_mask=attention_mask, 
                position_ids=position_ids,
                position_embeddings=(cos, sin), 
                use_cache=False
            )
            hidden_states = out[0] if isinstance(out, tuple) else out

        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        for idx in self.meta.layer_indices:
            layer = self.layer_lookup[idx]
            out = layer(hidden_states, attention_mask=attention_mask, use_cache=False)
            hidden_states = out[0] if isinstance(out, tuple) else out

        logits = None
        if return_logits:
            if self.final_norm is not None:
                hidden_states = self.final_norm(hidden_states)
            if self.lm_head is None:
                raise ValueError("return_logits=True requires the final partition with lm_head.")
            logits = self.lm_head(hidden_states)

        return hidden_states, logits
