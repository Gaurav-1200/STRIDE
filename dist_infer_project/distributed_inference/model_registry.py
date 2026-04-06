from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from transformers import AutoConfig, AutoModelForCausalLM


@dataclass
class CausalLMAdapter:
    model_type: str

    def _backbone(self, model: Any) -> Any:
        if hasattr(model, "model"):
            return model.model
        raise ValueError(f"Unsupported model wrapper for type={type(model)}")

    def embed_tokens(self, model: Any) -> Any:
        return self._backbone(model).embed_tokens

    def layers(self, model: Any) -> List[Any]:
        return list(self._backbone(model).layers)

    def final_norm(self, model: Any) -> Any:
        backbone = self._backbone(model)
        if hasattr(backbone, "norm"):
            return backbone.norm
        raise ValueError("Backbone has no final norm module")

    def lm_head(self, model: Any) -> Any:
        return model.lm_head

    def supported(self, config: Any) -> bool:
        return config.model_type in {
            "llama",
            "mistral",
            "qwen2",
            "qwen3",
            "gemma",
            "gemma2",
        }


def get_adapter_for_model(model_id: str) -> CausalLMAdapter:
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
    adapter = CausalLMAdapter(model_type=config.model_type)
    if not adapter.supported(config):
        raise ValueError(
            f"Unsupported model_type={config.model_type}. "
            "This scaffold currently targets decoder-only Hugging Face CausalLM models "
            "whose structure exposes model.embed_tokens, model.layers, model.norm and lm_head."
        )
    return adapter


def load_full_model(model_id: str, dtype: str):
    import torch

    torch_dtype = getattr(torch, dtype)
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
