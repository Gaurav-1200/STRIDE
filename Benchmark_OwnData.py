"""
Benchmark.py — MLM accuracy benchmark for BERTFull vs BERTHead+BERTTail split.

Uses the GLUE SST-2 validation set. For each sentence we:
  1. Randomly mask one word
  2. Run inference (full or split)
  3. Check if the top-1 predicted token matches the original

This measures whether split inference preserves model quality, and also
records latency per sample so you can compare throughput.

Usage:
    # Full model baseline
    python Benchmark.py --mode full --device cuda

    # Split inference (requires server running)
    python Benchmark.py --mode split --split-layer 6 \
        --server-host 192.168.1.50 --server-port 50051

    # Local split (no network, both head and tail on same machine)
    python Benchmark.py --mode local_split --split-layer 6 --device cuda

    # Compare all modes and print summary table
    python Benchmark.py --mode compare --device cuda
"""

import argparse
import json
import os
import sys
import time
import random
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── Result Container ──────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    mode: str
    split_layer: int
    device: str
    num_samples: int
    correct: int = 0
    latencies_ms: List[float] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.num_samples if self.num_samples > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return float(np.mean(self.latencies_ms)) if self.latencies_ms else 0.0

    @property
    def p50_latency_ms(self) -> float:
        return float(np.percentile(self.latencies_ms, 50)) if self.latencies_ms else 0.0

    @property
    def p95_latency_ms(self) -> float:
        return float(np.percentile(self.latencies_ms, 95)) if self.latencies_ms else 0.0

    @property
    def throughput_qps(self) -> float:
        return 1000.0 / self.avg_latency_ms if self.avg_latency_ms > 0 else 0.0

    def summary(self) -> str:
        lines = [
            f"\n{'='*56}",
            f"  Mode        : {self.mode}",
            f"  Split Layer : {self.split_layer}",
            f"  Device      : {self.device}",
            f"  Samples     : {self.num_samples}",
            f"  Accuracy    : {self.accuracy*100:.2f}%",
            f"  Avg Latency : {self.avg_latency_ms:.2f} ms",
            f"  P50 Latency : {self.p50_latency_ms:.2f} ms",
            f"  P95 Latency : {self.p95_latency_ms:.2f} ms",
            f"  Throughput  : {self.throughput_qps:.1f} QPS",
            f"{'='*56}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "split_layer": self.split_layer,
            "device": self.device,
            "num_samples": self.num_samples,
            "accuracy": self.accuracy,
            "avg_latency_ms": self.avg_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "throughput_qps": self.throughput_qps,
        }


# ── Dataset Loading ───────────────────────────────────────────────────────────

def load_sst2_sentences(max_samples: int = 500) -> List[str]:
    """
    Load SST-2 validation sentences from HuggingFace datasets.
    Falls back to a small built-in set if datasets is not installed.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("glue", "sst2", split="validation")
        sentences = [row["sentence"] for row in ds]
        random.shuffle(sentences)
        print(f"[Benchmark] Loaded {len(sentences)} SST-2 sentences from HuggingFace.")
        return sentences[:max_samples]
    except Exception as e:
        print(f"[Benchmark] datasets not available ({e}), using built-in sentences.")
        return _builtin_sentences()[:max_samples]


def _builtin_sentences() -> List[str]:
    """Small fallback corpus — no external dependencies needed."""
    return [
        "The film is a great piece of entertainment.",
        "This movie was absolutely terrible and boring.",
        "The acting in this production was outstanding.",
        "I found the story to be quite predictable.",
        "A wonderful journey through breathtaking landscapes.",
        "The director failed to capture the essence of the book.",
        "Every scene was beautifully crafted and memorable.",
        "The plot had too many holes to be enjoyable.",
        "An inspiring tale of courage and determination.",
        "The dialogue was stilted and unconvincing throughout.",
        "A masterpiece of modern cinema that everyone should see.",
        "The special effects could not save this weak script.",
        "The performances were genuinely moving and authentic.",
        "I walked out halfway through it was so dull.",
        "A clever and witty comedy that kept me laughing.",
        "The music was the only redeeming feature of the film.",
        "Superb direction and a compelling narrative arc.",
        "The characters were flat and completely unrelatable.",
        "One of the best films I have seen this year.",
        "A disappointing sequel that tarnishes the original.",
    ] * 25   # repeat to get ~500 samples


# ── Masking Helper ────────────────────────────────────────────────────────────

def mask_random_word(tokenizer, sentence: str, device: str):
    """
    Tokenize sentence, mask one non-special token at random.
    Returns (input_ids, attention_mask, masked_position, original_token_id).
    """
    enc = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
    input_ids    = enc["input_ids"].to(device)
    attn_mask    = enc["attention_mask"].to(device)

    # Positions of non-special tokens (exclude [CLS]=101, [SEP]=102, [PAD]=0)
    special_ids = {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}
    # In mask_random_word(), replace the candidates line with:
    common_ids = set(tokenizer.convert_tokens_to_ids(
        ["the", "a", "an", "is", "was", "are", "of", "in", "to", "and", "it", "that"]
    ))
    candidates = [
        i for i in range(input_ids.shape[1])
        if input_ids[0, i].item() in common_ids
    ]
    # fall back to any non-special token if none found
    if not candidates:
        candidates = [i for i in range(input_ids.shape[1])
                    if input_ids[0, i].item() not in special_ids]

    pos = random.choice(candidates)
    original_id = input_ids[0, pos].item()
    input_ids[0, pos] = tokenizer.mask_token_id

    return input_ids, attn_mask, pos, original_id


# ── Inference Wrappers ────────────────────────────────────────────────────────

def predict_masked_token(logits: torch.Tensor, pos: int) -> int:
    """Return the token id with highest logit at the masked position."""
    return int(logits[0, pos].argmax().item())


# ── Benchmark Runners ─────────────────────────────────────────────────────────

def run_full_benchmark(
    sentences: List[str],
    tokenizer,
    model,
    device: str,
    warmup: int = 5,
) -> BenchmarkResult:
    """Benchmark BERTFull — standard single-machine inference."""
    result = BenchmarkResult(
        mode="full", split_layer=0, device=device, num_samples=len(sentences)
    )
    model.eval()

    # Warmup
    for s in sentences[:warmup]:
        masked = mask_random_word(tokenizer, s, device)
        if masked is None:
            continue
        input_ids, attn_mask, pos, _ = masked
        with torch.no_grad():
            model(input_ids, attn_mask)

    # Timed runs
    for sentence in sentences:
        masked = mask_random_word(tokenizer, sentence, device)
        if masked is None:
            result.num_samples -= 1
            continue
        input_ids, attn_mask, pos, original_id = masked

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(input_ids, attn_mask)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        pred_id = predict_masked_token(logits, pos)
        if pred_id == original_id:
            result.correct += 1
        result.latencies_ms.append(elapsed_ms)

    return result


def run_local_split_benchmark(
    sentences: List[str],
    tokenizer,
    head,
    tail,
    device: str,
    warmup: int = 5,
) -> BenchmarkResult:
    """
    Benchmark BERTHead + BERTTail on the same machine (no network).
    Isolates compute cost of split from communication cost.
    """
    result = BenchmarkResult(
        mode="local_split",
        split_layer=head.split_layer,
        device=device,
        num_samples=len(sentences),
    )
    head.eval()
    tail.eval()

    for s in sentences[:warmup]:
        masked = mask_random_word(tokenizer, s, device)
        if masked is None:
            continue
        input_ids, attn_mask, _, _ = masked
        with torch.no_grad():
            hidden = head(input_ids, attn_mask)
            tail(hidden, attn_mask)

    for sentence in sentences:
        masked = mask_random_word(tokenizer, sentence, device)
        if masked is None:
            result.num_samples -= 1
            continue
        input_ids, attn_mask, pos, original_id = masked

        t0 = time.perf_counter()
        with torch.no_grad():
            hidden = head(input_ids, attn_mask)
            logits = tail(hidden, attn_mask)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        pred_id = predict_masked_token(logits, pos)
        if pred_id == original_id:
            result.correct += 1
        result.latencies_ms.append(elapsed_ms)

    return result


def run_grpc_split_benchmark(
    sentences: List[str],
    tokenizer,
    head,
    grpc_client,
    model_name: str,
    split_layer: int,
    device: str,
    warmup: int = 5,
) -> BenchmarkResult:
    """
    Benchmark BERTHead (local) + BERTTail (remote via gRPC).
    Measures real end-to-end latency including network.
    """
    result = BenchmarkResult(
        mode="grpc_split",
        split_layer=split_layer,
        device=device,
        num_samples=len(sentences),
    )
    head.eval()

    for s in sentences[:warmup]:
        masked = mask_random_word(tokenizer, s, device)
        if masked is None:
            continue
        input_ids, attn_mask, _, _ = masked
        with torch.no_grad():
            hidden = head(input_ids, attn_mask)
        grpc_client.run_tail(hidden, model_name, split_layer, attn_mask)

    for sentence in sentences:
        masked = mask_random_word(tokenizer, sentence, device)
        if masked is None:
            result.num_samples -= 1
            continue
        input_ids, attn_mask, pos, original_id = masked

        t0 = time.perf_counter()
        with torch.no_grad():
            hidden = head(input_ids, attn_mask)
        tail_result = grpc_client.run_tail(hidden, model_name, split_layer, attn_mask)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        logits = tail_result.logits.to(device)
        pred_id = predict_masked_token(logits, pos)
        if pred_id == original_id:
            result.correct += 1
        result.latencies_ms.append(elapsed_ms)

    return result


# ── Comparison Printer ────────────────────────────────────────────────────────

def print_comparison(results: List[BenchmarkResult]):
    """Print a side-by-side comparison table of benchmark results."""
    print(f"\n{'='*80}")
    print(f"  BENCHMARK COMPARISON")
    print(f"{'='*80}")
    header = f"  {'Mode':<18} {'SplitLayer':>10} {'Accuracy':>10} {'AvgLat(ms)':>12} {'P95Lat(ms)':>12} {'QPS':>8}"
    print(header)
    print(f"  {'-'*76}")

    baseline = next((r for r in results if r.mode == "full"), None)
    for r in results:
        acc_str = f"{r.accuracy*100:.2f}%"
        # Mark accuracy drop vs full
        if baseline and r.mode != "full":
            drop = (baseline.accuracy - r.accuracy) * 100
            acc_str += f" ({drop:+.2f}%)"
        print(f"  {r.mode:<18} {r.split_layer:>10} {acc_str:>16} "
              f"{r.avg_latency_ms:>12.2f} {r.p95_latency_ms:>12.2f} {r.throughput_qps:>8.1f}")

    if baseline:
        print(f"\n  Accuracy delta shows quality preserved by split (should be ~0%).")
        print(f"  QPS shows server throughput — split offloads compute to edge.")
    print(f"{'='*80}")


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MLM accuracy + latency benchmark")
    parser.add_argument("--mode",        default="compare",
                        choices=["full", "local_split", "grpc_split", "compare"])
    parser.add_argument("--split-layer", type=int, default=6)
    parser.add_argument("--device",      default=None)
    parser.add_argument("--samples",     type=int, default=200)
    parser.add_argument("--warmup",      type=int, default=5)
    parser.add_argument("--save-dir",    default="assets/BERT")
    parser.add_argument("--output-dir",  default="./results")
    parser.add_argument("--server-host", default="192.168.31.150")
    parser.add_argument("--server-port", type=int, default=50051)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Benchmark] device={device} samples={args.samples} split_layer={args.split_layer}")

    # ── Load data ──────────────────────────────────────────────────────────────
    from Models import load_bert_tokenizer
    tokenizer = load_bert_tokenizer()
    sentences = load_sst2_sentences(args.samples)
    print(f"[Benchmark] {len(sentences)} sentences loaded.")

    results = []

    # ── Full model ─────────────────────────────────────────────────────────────
    if args.mode in ("full", "compare"):
        print("\n[Benchmark] Running FULL model...")
        from Models import BERTFull, load_bert
        base = load_bert(device)
        full_model = BERTFull(base)
        r = run_full_benchmark(sentences, tokenizer, full_model, device, args.warmup)
        print(r.summary())
        results.append(r)

    # ── Local split ────────────────────────────────────────────────────────────
    if args.mode in ("local_split", "compare"):
        print(f"\n[Benchmark] Running LOCAL SPLIT (layer {args.split_layer})...")
        from Models import BERTHead, BERTTail, load_bert_head, load_bert_tail
        head = load_bert_head(args.save_dir, args.split_layer, device)
        tail = load_bert_tail(args.save_dir, args.split_layer, device)
        r = run_local_split_benchmark(
            sentences, tokenizer, head, tail, device, args.warmup
        )
        print(r.summary())
        results.append(r)

    # ── gRPC split ─────────────────────────────────────────────────────────────
    if args.mode in ("grpc_split",):
        print(f"\n[Benchmark] Running GRPC SPLIT (layer {args.split_layer})...")
        from Models import load_bert_head
        from communication.client import SplitInferenceClient
        head = load_bert_head(args.save_dir, args.split_layer, device)
        client = SplitInferenceClient(args.server_host, args.server_port)
        if not client.health_check():
            print("[Benchmark] Server unreachable — skipping gRPC split.")
        else:
            r = run_grpc_split_benchmark(
                sentences, tokenizer, head, client,
                "bert", args.split_layer, device, args.warmup
            )
            print(r.summary())
            results.append(r)
            client.close()

    # ── Comparison ─────────────────────────────────────────────────────────────
    if len(results) > 1:
        print_comparison(results)

    # ── Save ───────────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    out = {
        "config": vars(args),
        "results": [r.to_dict() for r in results],
    }
    save_path = os.path.join(args.output_dir, f"benchmark_split{args.split_layer}.json")
    with open(save_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[Benchmark] Saved → {save_path}")


if __name__ == "__main__":
    main()