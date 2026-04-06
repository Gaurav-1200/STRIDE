from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer

from distributed_inference.model_registry import get_adapter_for_model, load_full_model


@dataclass
class PartitionPlan:
    model_id: str
    num_layers: int
    partitions: List[List[int]]


def contiguous_partition(num_layers: int, num_workers: int) -> PartitionPlan:
    base = num_layers // num_workers
    rem = num_layers % num_workers
    partitions: List[List[int]] = []
    start = 0
    for i in range(num_workers):
        size = base + (1 if i < rem else 0)
        end = start + size
        partitions.append(list(range(start, end)))
        start = end
    return PartitionPlan(model_id="", num_layers=num_layers, partitions=partitions)

def dynamic_partition(num_layers:int, layer_counts:List[int]) ->PartitionPlan:
    if sum(layer_counts)!=num_layers:
        raise ValueError(f"Total layers are mismatching{sum(layer_counts)}!={num_layers}")
    
    partitions : List[List[int]] =[]
    current_layer = 0
    for count in layer_counts:
        end_layer = current_layer + count
        partitions.append(list(range(current_layer,end_layer)))
        current_layer = end_layer
    
    return PartitionPlan(model_id="",num_layers=num_layers,partitions=partitions)


def export_partitions(model_id: str, output_dir: str, num_workers: int, layer_counts:List[int], dtype: str = "float16") -> None:

    if len(layer_counts)!=num_workers:
        raise ValueError("All the workers are not getting layers to execute")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = load_full_model(model_id, dtype=dtype)
    model.eval()
    adapter = get_adapter_for_model(model_id)
    layers = adapter.layers(model)

    # plan = contiguous_partition(len(layers), num_workers)
    plan = dynamic_partition(len(layers),layer_counts)
    plan.model_id = model_id

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    tokenizer.save_pretrained(out / "tokenizer")

    common_meta = {
        "model_id": model_id,
        "dtype": dtype,
        "num_layers": len(layers),
        "num_workers": num_workers,
    }

    for worker_idx, layer_ids in enumerate(plan.partitions):
        worker_dir = out / f"partition_{worker_idx}"
        worker_dir.mkdir(parents=True, exist_ok=True)

        print(f"Worker {worker_idx} | Layers: {layer_ids} | Saving to: {worker_dir.absolute()}")


        state: Dict[str, torch.Tensor] = {}
        if worker_idx == 0:
            for k, v in adapter.embed_tokens(model).state_dict().items():
                state[f"embed_tokens.{k}"] = v.detach().cpu().contiguous()

        for layer_id in layer_ids:
            layer = layers[layer_id]
            for k, v in layer.state_dict().items():
                state[f"layers.{layer_id}.{k}"] = v.detach().cpu().contiguous()

        if worker_idx == num_workers - 1:
            for k, v in adapter.final_norm(model).state_dict().items():
                state[f"final_norm.{k}"] = v.detach().cpu().contiguous()
            for k, v in adapter.lm_head(model).state_dict().items():
                state[f"lm_head.{k}"] = v.detach().cpu().contiguous()

        save_file(state, str(worker_dir / "weights.safetensors"))

        meta = {
            **common_meta,
            "worker_idx": worker_idx,
            "layer_indices": layer_ids,
            "has_embeddings": worker_idx == 0,
            "has_final_norm": worker_idx == num_workers - 1,
            "has_lm_head": worker_idx == num_workers - 1,
        }
        with open(worker_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    with open(out / "plan.json", "w", encoding="utf-8") as f:
        json.dump({"model_id": model_id, "partitions": plan.partitions}, f, indent=2)
