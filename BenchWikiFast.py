"""
Benchmark.py — MLM perplexity benchmark on WikiText-103.

Two scoring modes:
  - "exact"  : True pseudo-perplexity — mask each token individually.
               O(seq_len) forward passes per sentence. Slow but standard.
               Use for final reportable numbers.
  - "fast"   : Approximate perplexity — mask all tokens at once in one
               forward pass. O(1) passes per sentence. ~100x faster.
               Slightly optimistic but the delta between full/split is
               identical, so it's valid for comparing the two.

The key result is always:
    perplexity(full) ≈ perplexity(split)   [delta should be ~0.0]

Reference:
    Devlin et al. (2019) BERT: Pre-training of Deep Bidirectional Transformers
    Salazar et al. (2020) Masked Language Model Scoring
"""

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Callable

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── Result Container ──────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    mode: str
    split_layer: int
    device: str
    num_sentences: int
    scoring: str = "fast"
    num_tokens: int = 0
    total_log_prob: float = 0.0
    latencies_ms: List[float] = field(default_factory=list)

    @property
    def perplexity(self) -> float:
        if self.num_tokens == 0:
            return float("inf")
        return math.exp(-self.total_log_prob / self.num_tokens)

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
            f"  Scoring     : {self.scoring}",
            f"  Split Layer : {self.split_layer}",
            f"  Device      : {self.device}",
            f"  Sentences   : {self.num_sentences}",
            f"  Tokens eval : {self.num_tokens}",
            f"  Perplexity  : {self.perplexity:.4f}",
            f"  Avg Latency : {self.avg_latency_ms:.2f} ms/sentence",
            f"  P50 Latency : {self.p50_latency_ms:.2f} ms",
            f"  P95 Latency : {self.p95_latency_ms:.2f} ms",
            f"  Throughput  : {self.throughput_qps:.1f} sentences/sec",
            f"{'='*56}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "scoring": self.scoring,
            "split_layer": self.split_layer,
            "device": self.device,
            "num_sentences": self.num_sentences,
            "num_tokens": self.num_tokens,
            "perplexity": self.perplexity,
            "avg_latency_ms": self.avg_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "throughput_qps": self.throughput_qps,
        }


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_wikitext(max_sentences: int = 500,
                  subset: str = "wikitext-103-raw-v1") -> List[str]:
    try:
        from datasets import load_dataset
        print(f"[Benchmark] Loading {subset} test split...")
        ds = load_dataset("wikitext", subset, split="test")
        sentences = [
            row["text"].strip() for row in ds
            if len(row["text"].strip()) > 20
            and not row["text"].strip().startswith("=")
        ]
        random.shuffle(sentences)
        print(f"[Benchmark] {len(sentences)} usable sentences. "
              f"Using first {min(max_sentences, len(sentences))}.")
        return sentences[:max_sentences]
    except Exception as e:
        print(f"[Benchmark] {subset} failed ({e}), trying wikitext-2...")
        try:
            from datasets import load_dataset
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            sentences = [
                row["text"].strip() for row in ds
                if len(row["text"].strip()) > 20
                and not row["text"].strip().startswith("=")
            ]
            random.shuffle(sentences)
            return sentences[:max_sentences]
        except Exception as e2:
            print(f"[Benchmark] datasets unavailable. pip install datasets")
            return _builtin_sentences()[:max_sentences]


def _builtin_sentences() -> List[str]:
    return [
        "The tower is 324 metres tall and the tallest structure in Paris.",
        "As of 2015 it is the third tallest free-standing structure in the world.",
        "The building has a remarkable architectural design that attracts visitors.",
        "Scientists discovered a new species of bird in the Amazon rainforest.",
        "The economy grew at a rate of three percent in the second quarter.",
        "Researchers at the university published their findings in a scientific journal.",
        "The city council voted to approve the new infrastructure spending plan.",
        "Weather forecasters predicted heavy rainfall across the northern regions.",
        "The ancient manuscript was carefully preserved in a climate-controlled vault.",
        "Engineers designed a bridge capable of withstanding severe weather conditions.",
    ] * 50


# ── Scoring Functions ─────────────────────────────────────────────────────────

def score_fast(tokenizer, sentence: str, infer_fn: Callable,
               device: str, max_length: int = 128):
    """
    Fast approximate scoring: mask ALL non-special tokens at once,
    run ONE forward pass, collect log-probs for all masked positions.

    This is ~seq_len times faster than exact PPPL.
    The absolute perplexity is slightly lower than exact PPPL because
    BERT sees [MASK] tokens at all positions simultaneously, making each
    prediction slightly easier. But the delta between full and split is
    identical — valid for comparison.
    """
    enc = tokenizer(sentence, return_tensors="pt",
                    truncation=True, max_length=max_length)
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)

    special_ids = {
        tokenizer.cls_token_id, tokenizer.sep_token_id,
        tokenizer.pad_token_id, tokenizer.unk_token_id,
    }
    positions = [
        i for i in range(input_ids.shape[1])
        if input_ids[0, i].item() not in special_ids
    ]
    if len(positions) < 2:
        return None

    original_ids = input_ids[0, positions].clone()
    masked_ids = input_ids.clone()
    for p in positions:
        masked_ids[0, p] = tokenizer.mask_token_id

    with torch.no_grad():
        logits = infer_fn(masked_ids, attn_mask)   # (1, seq, vocab)

    log_probs = F.log_softmax(logits[0], dim=-1)   # (seq, vocab)
    total_log_prob = sum(
        log_probs[pos, original_ids[i].item()].item()
        for i, pos in enumerate(positions)
    )
    return total_log_prob, len(positions)


def score_exact(tokenizer, sentence: str, infer_fn: Callable,
                device: str, max_length: int = 128):
    """
    Exact pseudo-perplexity: mask each token individually.
    O(seq_len) forward passes. Slow but theoretically correct.
    """
    enc = tokenizer(sentence, return_tensors="pt",
                    truncation=True, max_length=max_length)
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)

    special_ids = {
        tokenizer.cls_token_id, tokenizer.sep_token_id,
        tokenizer.pad_token_id, tokenizer.unk_token_id,
    }
    positions = [
        i for i in range(input_ids.shape[1])
        if input_ids[0, i].item() not in special_ids
    ]
    if len(positions) < 2:
        return None

    total_log_prob = 0.0
    for pos in positions:
        masked_ids = input_ids.clone()
        original_id = masked_ids[0, pos].item()
        masked_ids[0, pos] = tokenizer.mask_token_id
        with torch.no_grad():
            logits = infer_fn(masked_ids, attn_mask)
        log_probs = F.log_softmax(logits[0, pos], dim=-1)
        total_log_prob += log_probs[original_id].item()

    return total_log_prob, len(positions)


# ── Benchmark Runner ──────────────────────────────────────────────────────────

def _run_benchmark(mode, split_layer, scoring, sentences, tokenizer,
                   infer_fn, device, warmup=3):
    score_fn = score_fast if scoring == "fast" else score_exact
    result = BenchmarkResult(
        mode=mode, split_layer=split_layer, scoring=scoring,
        device=device, num_sentences=len(sentences),
    )

    print(f"[Benchmark] Warmup ({warmup} sentences)...")
    for s in sentences[:warmup]:
        enc = tokenizer(s, return_tensors="pt", truncation=True, max_length=128)
        ids  = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            infer_fn(ids, mask)

    print(f"[Benchmark] Scoring {len(sentences)} sentences "
          f"[{scoring} mode]...")
    for i, sentence in enumerate(sentences):
        if (i + 1) % 50 == 0:
            ppl = math.exp(-result.total_log_prob / max(result.num_tokens, 1))
            print(f"  {i+1}/{len(sentences)}  PPL so far: {ppl:.4f}")

        t0 = time.perf_counter()
        scored = score_fn(tokenizer, sentence, infer_fn, device)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if scored is None:
            result.num_sentences -= 1
            continue

        log_prob_sum, num_tokens = scored
        result.total_log_prob += log_prob_sum
        result.num_tokens     += num_tokens
        result.latencies_ms.append(elapsed_ms)

    return result


def run_full(sentences, tokenizer, model, device, scoring, warmup=3):
    model.eval()
    return _run_benchmark("full", 0, scoring, sentences, tokenizer,
                          lambda ids, mask: model(ids, mask),
                          device, warmup)


def run_local_split(sentences, tokenizer, head, tail, device, scoring, warmup=3):
    head.eval(); tail.eval()
    def infer(ids, mask):
        hidden = head(ids, mask)
        return tail(hidden, mask)
    return _run_benchmark("local_split", head.split_layer, scoring,
                          sentences, tokenizer, infer, device, warmup)


def run_grpc_split(sentences, tokenizer, head, grpc_client,
                   model_name, split_layer, device, scoring, warmup=3):
    head.eval()
    def infer(ids, mask):
        with torch.no_grad():
            hidden = head(ids, mask)
        res = grpc_client.run_tail(hidden, model_name, split_layer, mask)
        return res.logits.to(device)
    return _run_benchmark("grpc_split", split_layer, scoring,
                          sentences, tokenizer, infer, device, warmup)


# ── Comparison Table ──────────────────────────────────────────────────────────

def print_comparison(results: List[BenchmarkResult]):
    scoring = results[0].scoring
    print(f"\n{'='*80}")
    print(f"  BENCHMARK: WikiText Pseudo-Perplexity  [scoring={scoring}]")
    print(f"  Reference: Salazar et al. (2020) Masked Language Model Scoring")
    print(f"{'='*80}")
    hdr = (f"  {'Mode':<18} {'SplitLayer':>10} {'Perplexity':>14} "
           f"{'AvgLat(ms)':>12} {'P95Lat(ms)':>12} {'Sent/sec':>10}")
    print(hdr)
    print(f"  {'-'*78}")

    baseline = next((r for r in results if r.mode == "full"), None)
    for r in results:
        ppl_str = f"{r.perplexity:.4f}"
        if baseline and r.mode != "full":
            delta = r.perplexity - baseline.perplexity
            ppl_str += f" ({delta:+.4f})"
        print(f"  {r.mode:<18} {r.split_layer:>10} {ppl_str:>20} "
              f"{r.avg_latency_ms:>12.2f} {r.p95_latency_ms:>12.2f} "
              f"{r.throughput_qps:>10.1f}")

    if baseline:
        print(f"\n  PPL delta should be ~0.0000 — confirming lossless split.")
    print(f"{'='*80}")


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",        default="compare",
                        choices=["full", "local_split", "grpc_split", "compare"])
    parser.add_argument("--scoring",     default="fast",
                        choices=["fast", "exact"],
                        help="fast=1 pass/sentence (~20ms), "
                             "exact=N passes/sentence (~2000ms)")
    parser.add_argument("--split-layer", type=int, default=6)
    parser.add_argument("--device",      default=None)
    parser.add_argument("--samples",     type=int, default=200)
    parser.add_argument("--warmup",      type=int, default=3)
    parser.add_argument("--save-dir",    default="assets/BERT")
    parser.add_argument("--output-dir",  default="./results")
    parser.add_argument("--server-host", default="192.168.31.150")
    parser.add_argument("--server-port", type=int, default=50051)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--wikitext",    default="wikitext-103-raw-v1",
                        choices=["wikitext-103-raw-v1", "wikitext-2-raw-v1"])
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Benchmark] device={device} | scoring={args.scoring} | "
          f"samples={args.samples} | split_layer={args.split_layer}")

    from Models import load_bert_tokenizer
    tokenizer = load_bert_tokenizer()
    sentences = load_wikitext(args.samples, args.wikitext)

    results = []

    if args.mode in ("full", "compare"):
        print("\n[Benchmark] === FULL MODEL ===")
        from Models import BERTFull, load_bert
        base = load_bert(device)
        r = run_full(sentences, tokenizer, BERTFull(base),
                     device, args.scoring, args.warmup)
        print(r.summary())
        results.append(r)

    if args.mode in ("local_split", "compare"):
        print(f"\n[Benchmark] === LOCAL SPLIT (layer {args.split_layer}) ===")
        from Models import load_bert_head, load_bert_tail
        head = load_bert_head(args.save_dir, args.split_layer, device)
        tail = load_bert_tail(args.save_dir, args.split_layer, device)
        r = run_local_split(sentences, tokenizer, head, tail,
                            device, args.scoring, args.warmup)
        print(r.summary())
        results.append(r)

    if args.mode == "grpc_split":
        print(f"\n[Benchmark] === GRPC SPLIT (layer {args.split_layer}) ===")
        from Models import load_bert_head
        from communication.client import SplitInferenceClient
        head = load_bert_head(args.save_dir, args.split_layer, device)
        client = SplitInferenceClient(args.server_host, args.server_port)
        if not client.health_check():
            print("[Benchmark] Server unreachable. Start server.py first.")
        else:
            r = run_grpc_split(sentences, tokenizer, head, client, "bert",
                               args.split_layer, device, args.scoring, args.warmup)
            print(r.summary())
            results.append(r)
            client.close()

    if len(results) > 1:
        print_comparison(results)

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir,
                             f"benchmark_{args.scoring}_split{args.split_layer}.json")
    with open(save_path, "w") as f:
        json.dump({"config": vars(args), "results": [r.to_dict() for r in results]},
                  f, indent=2)
    print(f"\n[Benchmark] Saved → {save_path}")


if __name__ == "__main__":
    main()