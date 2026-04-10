from __future__ import annotations

"""@@@CHANGE@@@ Offline per-layer model profiler for planner input generation."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoTokenizer

from distributed_inference.model_registry import get_adapter_for_model, load_full_model
from distributed_inference.profiling.layer_profiler import LayerProfiler


def _aggregate_block_metrics(metrics, block_prefixes: List[str]) -> List[Dict]:
    aggregated: List[Dict] = []
    for block_id, prefix in enumerate(block_prefixes):
        matched = [m for m in metrics if m.layer_name == prefix or m.layer_name.startswith(prefix + ".")]
        aggregated.append(
            {
                "layer_index": block_id,
                "layer_name": prefix,
                "latency_ms": float(sum(m.latency_ms for m in matched)),
                "flops": float(sum(m.flops for m in matched)),
                "mem_delta_mb": float(max((m.mem_delta_mb for m in matched), default=0.0)),
                "num_profiled_modules": len(matched),
            }
        )
    return aggregated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="Explain attention in one sentence.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    args = parser.parse_args()

    model = load_full_model(args.model_id, dtype=args.dtype)
    model = model.to(args.device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=False)
    encoded = tokenizer(args.prompt, return_tensors="pt")
    encoded = {k: v.to(args.device) for k, v in encoded.items()}

    profiler = LayerProfiler(model, device=args.device)
    profiler.attach()
    with profiler.record():
        with torch.inference_mode():
            _ = model(**encoded, use_cache=False)
    profiler.detach()

    adapter = get_adapter_for_model(args.model_id)
    # @@@CHANGE@@@ Prefer stable block-level module names over exhaustive submodule names.
    model_layers = adapter.layers(model)
    layer_id_map = {id(layer): idx for idx, layer in enumerate(model_layers)}
    block_prefixes = [name for name, module in model.named_modules() if id(module) in layer_id_map]

    raw_metrics = [m.to_dict() for m in profiler.get_metrics()]
    block_metrics = _aggregate_block_metrics(profiler.get_metrics(), block_prefixes)

    output = {
        "model_id": args.model_id,
        "dtype": args.dtype,
        "device": args.device,
        "prompt": args.prompt,
        "num_layers": len(block_metrics),
        "raw_metrics": raw_metrics,
        "block_metrics": block_metrics,
        "total_flops_fvcore": profiler.get_total_flops_fvcore(encoded),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote layer profile to {out_path}")


if __name__ == "__main__":
    main()
