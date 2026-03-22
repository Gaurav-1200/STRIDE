"""
Benchmark.py — Cloze evaluation on WikiText-103.

Protocol:
  - For each sentence, mask exactly K tokens chosen by a seeded RNG.
  - The masked positions are saved to a JSON file so every run uses
    identical masks — results are reproducible and comparable.
  - Metric: Top-1 accuracy (did the model predict the exact original token?)
    and pseudo-perplexity on the masked positions only.

This covers both goals:
  1. Meaningful absolute numbers (not just deltas) — accuracy is interpretable.
  2. Public benchmark corpus (WikiText-103) — citable in a report.
  3. Consistency — same masks for full and split, so comparison is fair.
  4. Speed — O(1) forward passes per sentence (all K masks in one pass).

Why this is better than masking all tokens:
  - Model still has rich context (only 1-2 positions masked).
  - Accuracy numbers are meaningful (published BERT gets ~70-80% top-1
    on random single-token cloze on WikiText).
  - Fast — one forward pass per sentence.

Usage:
    # Generate masks once, then run full and split against same masks
    python Benchmark.py --mode compare --split-layer 18 --device cuda

    # Use existing mask file (reproducible across machines)
    python Benchmark.py --mode compare --mask-file results/masks.json

    # Batched throughput sweep
    python Benchmark.py --mode compare --batch --split-layer 18 --device cuda

Reference:
    Taylor (1953) "Cloze procedure" — original cloze test protocol
    Devlin et al. (2019) BERT paper — MLM evaluation
"""

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── Mask Record ───────────────────────────────────────────────────────────────

@dataclass
class MaskedSample:
    """A sentence with pre-determined mask positions for reproducibility."""
    sentence: str
    token_ids: List[int]          # full tokenized sequence (including CLS/SEP)
    masked_positions: List[int]   # which positions to mask (non-special only)
    original_token_ids: List[int] # the original token ids at masked positions


def build_mask_file(
    tokenizer,
    sentences: List[str],
    masks_per_sentence: int = 1,
    seed: int = 42,
    max_length: int = 128,
    save_path: Optional[str] = None,
) -> List[MaskedSample]:
    """
    Pre-compute mask positions for all sentences using a fixed seed.
    Saves to JSON so the same masks are reused across all runs.
    """
    rng = random.Random(seed)
    special_ids = {
        tokenizer.cls_token_id, tokenizer.sep_token_id,
        tokenizer.pad_token_id, tokenizer.unk_token_id,
    }
    samples = []
    skipped = 0

    for sentence in sentences:
        enc = tokenizer(sentence, return_tensors="pt",
                        truncation=True, max_length=max_length)
        token_ids = enc["input_ids"][0].tolist()

        candidates = [
            i for i, t in enumerate(token_ids)
            if t not in special_ids
        ]
        if len(candidates) < masks_per_sentence:
            skipped += 1
            continue

        masked_positions = sorted(rng.sample(candidates, masks_per_sentence))
        original_token_ids = [token_ids[p] for p in masked_positions]

        samples.append(MaskedSample(
            sentence=sentence,
            token_ids=token_ids,
            masked_positions=masked_positions,
            original_token_ids=original_token_ids,
        ))

    if skipped:
        print(f"[Masks] Skipped {skipped} sentences (too short).")
    print(f"[Masks] Created {len(samples)} masked samples "
          f"({masks_per_sentence} mask(s) each, seed={seed}).")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(
                [{"sentence": s.sentence,
                  "token_ids": s.token_ids,
                  "masked_positions": s.masked_positions,
                  "original_token_ids": s.original_token_ids}
                 for s in samples],
                f, indent=2
            )
        print(f"[Masks] Saved → {save_path}")

    return samples


def load_mask_file(path: str) -> List[MaskedSample]:
    with open(path) as f:
        records = json.load(f)
    samples = [
        MaskedSample(
            sentence=r["sentence"],
            token_ids=r["token_ids"],
            masked_positions=r["masked_positions"],
            original_token_ids=r["original_token_ids"],
        )
        for r in records
    ]
    print(f"[Masks] Loaded {len(samples)} pre-masked samples from {path}.")
    return samples


# ── Result Container ──────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    mode: str
    split_layer: int
    device: str
    masks_per_sentence: int
    num_sentences: int = 0
    correct_top1: int = 0
    correct_top5: int = 0
    total_masked: int = 0
    total_log_prob: float = 0.0
    latencies_ms: List[float] = field(default_factory=list)

    @property
    def accuracy_top1(self) -> float:
        return self.correct_top1 / self.total_masked if self.total_masked > 0 else 0.0

    @property
    def accuracy_top5(self) -> float:
        return self.correct_top5 / self.total_masked if self.total_masked > 0 else 0.0

    @property
    def perplexity(self) -> float:
        if self.total_masked == 0:
            return float("inf")
        return math.exp(-self.total_log_prob / self.total_masked)

    @property
    def avg_latency_ms(self) -> float:
        return float(np.mean(self.latencies_ms)) if self.latencies_ms else 0.0

    @property
    def p95_latency_ms(self) -> float:
        return float(np.percentile(self.latencies_ms, 95)) if self.latencies_ms else 0.0

    @property
    def throughput_qps(self) -> float:
        return 1000.0 / self.avg_latency_ms if self.avg_latency_ms > 0 else 0.0

    def summary(self) -> str:
        return "\n".join([
            f"\n{'='*56}",
            f"  Mode          : {self.mode}",
            f"  Split Layer   : {self.split_layer}",
            f"  Device        : {self.device}",
            f"  Sentences     : {self.num_sentences}",
            f"  Masked tokens : {self.total_masked} "
            f"({self.masks_per_sentence} per sentence)",
            f"  Top-1 Accuracy: {self.accuracy_top1*100:.2f}%",
            f"  Top-5 Accuracy: {self.accuracy_top5*100:.2f}%",
            f"  Perplexity    : {self.perplexity:.4f}",
            f"  Avg Latency   : {self.avg_latency_ms:.2f} ms/sentence",
            f"  P95 Latency   : {self.p95_latency_ms:.2f} ms",
            f"  Throughput    : {self.throughput_qps:.1f} sentences/sec",
            f"{'='*56}",
        ])

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "split_layer": self.split_layer,
            "device": self.device,
            "masks_per_sentence": self.masks_per_sentence,
            "num_sentences": self.num_sentences,
            "total_masked": self.total_masked,
            "accuracy_top1": self.accuracy_top1,
            "accuracy_top5": self.accuracy_top5,
            "perplexity": self.perplexity,
            "avg_latency_ms": self.avg_latency_ms,
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
        print(f"[Benchmark] WikiText unavailable ({e}). "
              f"Install with: pip install datasets")
        return _builtin_sentences()[:max_sentences]


def _builtin_sentences() -> List[str]:
    return [
        "The tower is 324 metres tall and the tallest structure in Paris.",
        "As of 2015 it is the third tallest free-standing structure in the world.",
        "Scientists discovered a new species of bird in the Amazon rainforest.",
        "The economy grew at a rate of three percent in the second quarter.",
        "Researchers published their findings in a peer-reviewed scientific journal.",
        "The city council voted to approve the new infrastructure spending plan.",
        "Weather forecasters predicted heavy rainfall across the northern regions.",
        "The ancient manuscript was carefully preserved in a climate-controlled vault.",
        "Engineers designed a bridge capable of withstanding severe weather conditions.",
        "The new policy was announced by the government during a press conference.",
    ] * 50


# ── Core Evaluation ───────────────────────────────────────────────────────────

def evaluate_sample(
    sample: MaskedSample,
    infer_fn: Callable,
    device: str,
) -> Tuple[int, int, float]:
    """
    Run one masked sample through infer_fn.
    Returns (correct_top1, correct_top5, sum_log_prob) for the masked tokens.
    One forward pass — all masked positions evaluated simultaneously.
    """
    token_ids = torch.tensor(sample.token_ids, dtype=torch.long).unsqueeze(0).to(device)
    attn_mask = torch.ones_like(token_ids)

    # Apply masks
    masked_ids = token_ids.clone()
    for pos in sample.masked_positions:
        masked_ids[0, pos] = 50264  # will be set properly below

    # Get mask token id from the tokenizer indirectly via the sample
    # (we stored original token ids, so we just need [MASK])
    # We'll import tokenizer mask_token_id inline
    from Models import load_bert_tokenizer
    _tok = load_bert_tokenizer  # just for mask_token_id — cached below

    with torch.no_grad():
        logits = infer_fn(masked_ids, attn_mask)   # (1, seq, vocab)

    correct_top1 = 0
    correct_top5 = 0
    total_log_prob = 0.0

    log_probs = F.log_softmax(logits[0], dim=-1)  # (seq, vocab)

    for pos, orig_id in zip(sample.masked_positions, sample.original_token_ids):
        top5 = logits[0, pos].topk(5).indices.tolist()
        if top5[0] == orig_id:
            correct_top1 += 1
        if orig_id in top5:
            correct_top5 += 1
        total_log_prob += log_probs[pos, orig_id].item()

    return correct_top1, correct_top5, total_log_prob


# ── Benchmark Runner ──────────────────────────────────────────────────────────

def _run_benchmark(
    mode: str,
    split_layer: int,
    samples: List[MaskedSample],
    infer_fn: Callable,
    device: str,
    mask_token_id: int,
    warmup: int = 3,
) -> BenchmarkResult:

    result = BenchmarkResult(
        mode=mode,
        split_layer=split_layer,
        device=device,
        masks_per_sentence=len(samples[0].masked_positions) if samples else 1,
        num_sentences=len(samples),
    )

    # Warmup
    print(f"[Benchmark] Warmup ({warmup} samples)...")
    for s in samples[:warmup]:
        token_ids = torch.tensor(s.token_ids).unsqueeze(0).to(device)
        masked_ids = token_ids.clone()
        for pos in s.masked_positions:
            masked_ids[0, pos] = mask_token_id
        attn_mask = torch.ones_like(masked_ids)
        with torch.no_grad():
            infer_fn(masked_ids, attn_mask)

    print(f"[Benchmark] Evaluating {len(samples)} samples [{mode}]...")
    for i, sample in enumerate(samples):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(samples)}  "
                  f"Top-1: {result.accuracy_top1*100:.1f}%  "
                  f"PPL: {result.perplexity:.4f}")

        token_ids = torch.tensor(sample.token_ids).unsqueeze(0).to(device)
        masked_ids = token_ids.clone()
        for pos in sample.masked_positions:
            masked_ids[0, pos] = mask_token_id
        attn_mask = torch.ones_like(masked_ids)

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = infer_fn(masked_ids, attn_mask)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        log_probs = F.log_softmax(logits[0], dim=-1)
        for pos, orig_id in zip(sample.masked_positions, sample.original_token_ids):
            top5 = logits[0, pos].topk(5).indices.tolist()
            if top5[0] == orig_id:
                result.correct_top1 += 1
            if orig_id in top5:
                result.correct_top5 += 1
            result.total_log_prob += log_probs[pos, orig_id].item()
            result.total_masked   += 1

        result.latencies_ms.append(elapsed_ms)

    return result


# ── Public API ────────────────────────────────────────────────────────────────

def run_full(samples, tokenizer, model, device, warmup=3):
    model.eval()
    return _run_benchmark("full", 0, samples,
                          lambda ids, mask: model(ids, mask),
                          device, tokenizer.mask_token_id, warmup)


def run_local_split(samples, tokenizer, head, tail, device, warmup=3):
    head.eval(); tail.eval()
    def infer(ids, mask):
        hidden = head(ids, mask)
        return tail(hidden, mask)
    return _run_benchmark("local_split", head.split_layer, samples,
                          infer, device, tokenizer.mask_token_id, warmup)


def run_grpc_split(samples, tokenizer, head, grpc_client,
                   model_name, split_layer, device, warmup=3):
    head.eval()
    def infer(ids, mask):
        with torch.no_grad():
            hidden = head(ids, mask)
        res = grpc_client.run_tail(hidden, model_name, split_layer, mask)
        return res.logits.to(device)
    return _run_benchmark("grpc_split", split_layer, samples,
                          infer, device, tokenizer.mask_token_id, warmup)


# ── Comparison Table ──────────────────────────────────────────────────────────

def print_comparison(results: List[BenchmarkResult]):
    print(f"\n{'='*90}")
    print(f"  CLOZE EVALUATION — WikiText-103  "
          f"(masks={results[0].masks_per_sentence} per sentence)")
    print(f"  Metric: Top-1/5 accuracy + perplexity on masked tokens only")
    print(f"  Same mask positions used for all modes — results are directly comparable")
    print(f"{'='*90}")
    hdr = (f"  {'Mode':<18} {'Split':>6} {'Top-1 Acc':>10} {'Top-5 Acc':>10} "
           f"{'PPL':>10} {'ΔTop-1':>8} {'ΔPPL':>10} {'ms/sent':>9} {'QPS':>8}")
    print(hdr)
    print(f"  {'-'*88}")

    baseline = next((r for r in results if r.mode == "full"), None)
    for r in results:
        delta_acc = ""
        delta_ppl = ""
        if baseline and r.mode != "full":
            da = (r.accuracy_top1 - baseline.accuracy_top1) * 100
            dp = r.perplexity - baseline.perplexity
            delta_acc = f"{da:+.2f}%"
            delta_ppl = f"{dp:+.4f}"
        print(f"  {r.mode:<18} {r.split_layer:>6} "
              f"{r.accuracy_top1*100:>9.2f}% {r.accuracy_top5*100:>9.2f}% "
              f"{r.perplexity:>10.4f} {delta_acc:>8} {delta_ppl:>10} "
              f"{r.avg_latency_ms:>9.2f} {r.throughput_qps:>8.1f}")

    if baseline:
        print(f"\n  ΔTop-1 and ΔPPL should be ~0 — confirming lossless split.")
        print(f"  Top-1 accuracy is interpretable: published BERT-base gets ~65-75%")
        print(f"  on single random-token cloze on WikiText.")
    print(f"{'='*90}")


# ── Batched Sweep ─────────────────────────────────────────────────────────────

from torch.nn.utils.rnn import pad_sequence

def run_batched_sweep(
    mode: str,
    split_layer: int,
    samples: List[MaskedSample],
    tokenizer,
    infer_fn: Callable,
    device: str,
    batch_sizes: List[int] = (1, 4, 8, 16, 32),
    warmup_batches: int = 2,
) -> list:
    results = []
    for bs in batch_sizes:
        batches = [samples[i:i+bs] for i in range(0, len(samples), bs)
                   if len(samples[i:i+bs]) == bs]
        if not batches:
            continue

        correct1 = correct5 = total = 0
        total_log_prob = 0.0
        latencies = []

        # Warmup
        for batch in batches[:warmup_batches]:
            ids_list  = [torch.tensor(s.token_ids) for s in batch]
            input_ids = pad_sequence(ids_list, batch_first=True,
                                     padding_value=tokenizer.pad_token_id).to(device)
            masked_ids = input_ids.clone()
            for bi, s in enumerate(batch):
                for pos in s.masked_positions:
                    masked_ids[bi, pos] = tokenizer.mask_token_id
            attn_mask = (masked_ids != tokenizer.pad_token_id).long()
            with torch.no_grad():
                infer_fn(masked_ids, attn_mask)

        for batch in batches:
            ids_list   = [torch.tensor(s.token_ids) for s in batch]
            input_ids  = pad_sequence(ids_list, batch_first=True,
                                      padding_value=tokenizer.pad_token_id).to(device)
            masked_ids = input_ids.clone()
            for bi, s in enumerate(batch):
                for pos in s.masked_positions:
                    masked_ids[bi, pos] = tokenizer.mask_token_id
            attn_mask = (masked_ids != tokenizer.pad_token_id).long()

            t0 = time.perf_counter()
            with torch.no_grad():
                logits = infer_fn(masked_ids, attn_mask)
            latencies.append((time.perf_counter() - t0) * 1000)

            for bi, s in enumerate(batch):
                lp = F.log_softmax(logits[bi], dim=-1)
                for pos, orig in zip(s.masked_positions, s.original_token_ids):
                    top5 = logits[bi, pos].topk(5).indices.tolist()
                    if top5[0] == orig: correct1 += 1
                    if orig in top5:    correct5 += 1
                    total_log_prob += lp[pos, orig].item()
                    total += 1

        avg_batch_ms  = float(np.mean(latencies))
        avg_sample_ms = avg_batch_ms / bs
        ppl = math.exp(-total_log_prob / total) if total > 0 else float("inf")
        qps = 1000.0 / avg_sample_ms if avg_sample_ms > 0 else 0.0
        acc = correct1 / total if total > 0 else 0.0

        results.append({
            "mode": mode, "split_layer": split_layer, "batch_size": bs,
            "accuracy_top1": acc, "perplexity": ppl,
            "avg_batch_latency_ms": avg_batch_ms,
            "avg_sample_latency_ms": avg_sample_ms,
            "throughput_qps": qps,
        })
        print(f"  bs={bs:>3}  Top-1={acc*100:.2f}%  PPL={ppl:.4f}  "
              f"lat/sample={avg_sample_ms:.1f}ms  QPS={qps:.1f}")
    return results


def print_batched_comparison(full_results, split_results):
    print(f"\n{'='*88}")
    print(f"  BATCHED THROUGHPUT: Full vs Split  (same masks, same corpus)")
    print(f"{'='*88}")
    print(f"  {'BS':>4} {'Full Acc':>10} {'Split Acc':>10} {'ΔAcc':>8} "
          f"{'Full QPS':>10} {'Split QPS':>10} {'Speedup':>9}")
    print(f"  {'-'*72}")
    full_by_bs  = {r["batch_size"]: r for r in full_results}
    split_by_bs = {r["batch_size"]: r for r in split_results}
    for bs in sorted(full_by_bs):
        if bs not in split_by_bs:
            continue
        f = full_by_bs[bs];  s = split_by_bs[bs]
        da = (s["accuracy_top1"] - f["accuracy_top1"]) * 100
        sp = s["throughput_qps"] / f["throughput_qps"] if f["throughput_qps"] else 0
        print(f"  {bs:>4} {f['accuracy_top1']*100:>9.2f}% "
              f"{s['accuracy_top1']*100:>9.2f}% {da:>+7.2f}% "
              f"{f['throughput_qps']:>10.1f} {s['throughput_qps']:>10.1f} {sp:>8.2f}x")
    print(f"{'='*88}")


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cloze evaluation on WikiText-103 for split BERT"
    )
    parser.add_argument("--mode",         default="compare",
                        choices=["full", "local_split", "grpc_split", "compare"])
    parser.add_argument("--split-layer",  type=int, default=6)
    parser.add_argument("--device",       default=None)
    parser.add_argument("--samples",      type=int, default=500)
    parser.add_argument("--masks",        type=int, default=1,
                        help="Tokens to mask per sentence (1 or 2 recommended)")
    parser.add_argument("--mask-file",    default=None,
                        help="Path to pre-saved mask JSON. If absent, masks are "
                             "generated and saved here.")
    parser.add_argument("--warmup",       type=int, default=3)
    parser.add_argument("--save-dir",     default="assets/BERT")
    parser.add_argument("--output-dir",   default="./results")
    parser.add_argument("--server-host",  default="192.168.31.150")
    parser.add_argument("--server-port",  type=int, default=50051)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--wikitext",     default="wikitext-103-raw-v1",
                        choices=["wikitext-103-raw-v1", "wikitext-2-raw-v1"])
    parser.add_argument("--batch",        action="store_true")
    parser.add_argument("--batch-sizes",  default="1,4,8,16,32")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Benchmark] device={device} | masks={args.masks}/sentence | "
          f"samples={args.samples} | split_layer={args.split_layer}")

    from Models import load_bert_tokenizer
    tokenizer = load_bert_tokenizer()

    # ── Masks: load existing or generate + save ────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    mask_file = args.mask_file or os.path.join(
        args.output_dir, f"masks_n{args.samples}_k{args.masks}_seed{args.seed}.json"
    )
    if os.path.exists(mask_file):
        samples = load_mask_file(mask_file)
        samples = samples[:args.samples]
    else:
        sentences = load_wikitext(args.samples, args.wikitext)
        samples = build_mask_file(
            tokenizer, sentences,
            masks_per_sentence=args.masks,
            seed=args.seed,
            save_path=mask_file,
        )

    print(f"[Benchmark] Using mask file: {mask_file}")

    results = []

    if args.mode in ("full", "compare"):
        print("\n[Benchmark] === FULL MODEL ===")
        from Models import BERTFull, load_bert
        full_model = BERTFull(load_bert(device))
        r = run_full(samples, tokenizer, full_model, device, args.warmup)
        print(r.summary())
        results.append(r)

    if args.mode in ("local_split", "compare"):
        print(f"\n[Benchmark] === LOCAL SPLIT (layer {args.split_layer}) ===")
        from Models import load_bert_head, load_bert_tail
        head = load_bert_head(args.save_dir, args.split_layer, device)
        tail = load_bert_tail(args.save_dir, args.split_layer, device)
        r = run_local_split(samples, tokenizer, head, tail, device, args.warmup)
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
            r = run_grpc_split(samples, tokenizer, head, client,
                               "bert", args.split_layer, device, args.warmup)
            print(r.summary())
            results.append(r)
            client.close()

    if len(results) > 1:
        print_comparison(results)

    # Save per-sentence results
    save_path = os.path.join(
        args.output_dir,
        f"benchmark_cloze_split{args.split_layer}_k{args.masks}.json"
    )
    with open(save_path, "w") as f:
        json.dump({"config": vars(args), "results": [r.to_dict() for r in results]},
                  f, indent=2)
    print(f"\n[Benchmark] Saved → {save_path}")

    # ── Batched sweep ──────────────────────────────────────────────────────────
    if args.batch:
        batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
        print(f"\n[Benchmark] Batched sweep: {batch_sizes}")
        from Models import BERTFull, load_bert, load_bert_head, load_bert_tail

        full_model = BERTFull(load_bert(device)); full_model.eval()
        head = load_bert_head(args.save_dir, args.split_layer, device)
        tail = load_bert_tail(args.save_dir, args.split_layer, device)
        head.eval(); tail.eval()

        print("\n[Batched] === FULL ===")
        full_b = run_batched_sweep("full", 0, samples, tokenizer,
                                   lambda i, m: full_model(i, m),
                                   device, batch_sizes)
        print(f"\n[Batched] === SPLIT (layer {args.split_layer}) ===")
        split_b = run_batched_sweep(
            "local_split", args.split_layer, samples, tokenizer,
            lambda i, m: tail(head(i, m), m),
            device, batch_sizes,
        )
        print_batched_comparison(full_b, split_b)

        batch_save = os.path.join(
            args.output_dir,
            f"benchmark_batched_split{args.split_layer}_k{args.masks}.json"
        )
        with open(batch_save, "w") as f:
            json.dump({"config": vars(args), "full": full_b, "split": split_b},
                      f, indent=2)
        print(f"[Benchmark] Batched results → {batch_save}")


if __name__ == "__main__":
    main()