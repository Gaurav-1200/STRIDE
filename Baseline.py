"""
run_baseline.py — Run full model on a single machine and collect metrics.

This is your Phase 1a / 2a: establish the ground truth that split inference
will be compared against.

Usage:
    python experiments/run_baseline.py --model gpt2
    python experiments/run_baseline.py --model bert
    python experiments/run_baseline.py --model gpt2 --device cuda --runs 10
"""

import argparse
import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profiler.LayerProfiler import LayerProfiler
from profiler.DeviceProfiler import DeviceProfiler
from utils.Metrics import RunMetrics, get_peak_memory_mb, reset_peak_memory, timer
from utils.Config import InferenceConfig


# ── Helpers ───────────────────────────────────────────────────────────────────

SAMPLE_PROMPTS = {
    "gpt2": "The future of artificial intelligence is",
    "bert": "The capital of [MASK] is Paris.",
}


def load_model_and_tokenizer(model_name: str, device: str):
    if model_name == "bert":
        from Models import BERTFull, load_bert_tokenizer,load_bert
        base = load_bert(device)
        model = BERTFull(base)
        tokenizer = load_bert_tokenizer()
        return model, tokenizer
    else:
        raise ValueError(f"Unknown model: {model_name}")


def encode(tokenizer, text: str, device: str, model_name: str):
    enc = tokenizer(text, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


# ── Core Experiment ───────────────────────────────────────────────────────────

def run_baseline(cfg: InferenceConfig) -> RunMetrics:
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  BASELINE RUN | model={cfg.model_name} | device={device}")
    print(f"{'='*60}")

    model, tokenizer = load_model_and_tokenizer(cfg.model_name, device)
    model.eval()

    prompt = SAMPLE_PROMPTS.get(cfg.model_name, "Hello world")
    inputs = encode(tokenizer, prompt, device, cfg.model_name)
    input_ids = inputs["input_ids"]

    dev_profiler = DeviceProfiler(device)
    dev_profiler.print_snapshot()

    layer_profiler = LayerProfiler(model.model if hasattr(model, "model") else model, device)

    metrics = RunMetrics(
        run_type="baseline",
        model_name=cfg.model_name,
        split_layer=0,
        device=device,
    )

    # ── Warmup ────────────────────────────────────────────────────────────────
    print(f"\n[Baseline] Warming up ({cfg.warmup_runs} runs)...")
    for _ in range(cfg.warmup_runs):
        with torch.no_grad():
            if cfg.model_name == "gpt2":
                model(input_ids)
            else:
                model(input_ids, inputs.get("attention_mask"))

    # ── Timed Runs ────────────────────────────────────────────────────────────
    print(f"[Baseline] Measuring ({cfg.timed_runs} runs)...")
    latencies = []
    reset_peak_memory(device)

    if cfg.profile_layers:
        layer_profiler.attach()

    for i in range(cfg.timed_runs):
        with timer() as t:
            with torch.no_grad():
                if cfg.profile_layers and i == 0:
                    with layer_profiler.record():
                        if cfg.model_name == "gpt2":
                            logits = model(input_ids)
                        else:
                            logits = model(input_ids, inputs.get("attention_mask"))
                else:
                    if cfg.model_name == "gpt2":
                        logits = model(input_ids)
                    else:
                        logits = model(input_ids, inputs.get("attention_mask"))
        latencies.append(t["elapsed_ms"])

    if cfg.profile_layers:
        layer_profiler.detach()

    # ── GPT-2: also measure TTFT (generation) ─────────────────────────────────
    ttft_ms = 0.0
    if cfg.model_name == "gpt2":
        print("[Baseline] Measuring TTFT (autoregressive generation)...")
        with timer() as t_ttft:
            with torch.no_grad():
                gen_ids = model.generate(input_ids, max_new_tokens=1)
        ttft_ms = t_ttft["elapsed_ms"]
        # Decode and print what the model generated
        generated = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        print(f"[Baseline] Generated: '{generated}'")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    layer_records = layer_profiler.get_metrics()
    total_flops = sum(lm.flops for lm in layer_records)
    peak_mem = get_peak_memory_mb(device)
    avg_latency = sum(latencies) / len(latencies)

    metrics.e2e_latency_ms = avg_latency
    metrics.ttft_ms = ttft_ms
    metrics.total_flops = total_flops
    metrics.peak_mem_mb = peak_mem
    metrics.total_mem_mb = sum(lm.mem_delta_mb for lm in layer_records if lm.mem_delta_mb > 0)
    metrics.layer_metrics = layer_records

    print(metrics.summary())
    if cfg.profile_layers:
        layer_profiler.print_layer_table(top_n=15)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(cfg.output_dir, exist_ok=True)
    save_path = os.path.join(cfg.output_dir, f"baseline_{cfg.model_name}.json")
    metrics.save(save_path)

    return metrics


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Baseline inference (single machine)")
    parser.add_argument("--model", default="bert", choices=["bert"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output-dir", default="./results")
    args = parser.parse_args()

    cfg = InferenceConfig(
        model_name=args.model,
        device=args.device,
        timed_runs=args.runs,
        warmup_runs=args.warmup,
        output_dir=args.output_dir,
    )
    run_baseline(cfg)


if __name__ == "__main__":
    main()