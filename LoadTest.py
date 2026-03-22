"""
LoadTest.py — Server-side throughput load test for split inference.

Goal: demonstrate that a split server (tail only) handles more concurrent
requests per second than a full server, because it does less compute per request.

Two modes:
  - "local"  : simulates the server by running full / tail locally in parallel
               threads. No network involved. Shows pure compute throughput.
  - "grpc"   : sends real concurrent gRPC requests to a running server.
               Shows real-world server throughput including network.

Experiment:
  Fix a number of concurrent "users" (threads). Each thread continuously
  sends requests. Measure:
    - Requests completed per second (server throughput)
    - Average latency per request
    - At what concurrency level does the server saturate

Usage:
    # Local simulation (no server needed)
    python LoadTest.py --mode local --split-layer 18 --device cuda

    # Real gRPC load test (server must be running)
    python LoadTest.py --mode grpc --split-layer 18 \
        --server-host 192.168.1.50 --server-port 50051

    # Sweep concurrency levels 1..16
    python LoadTest.py --mode local --split-layer 18 --concurrency 1,2,4,8,16
"""

import argparse
import json
import os
import queue
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import List, Callable, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── Result Container ──────────────────────────────────────────────────────────

@dataclass
class LoadResult:
    mode: str           # "full_server" or "split_server"
    split_layer: int
    concurrency: int    # number of parallel worker threads
    duration_sec: float
    completed: int = 0
    latencies_ms: List[float] = field(default_factory=list)
    errors: int = 0

    @property
    def throughput_qps(self) -> float:
        return self.completed / self.duration_sec if self.duration_sec > 0 else 0.0

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
    def p99_latency_ms(self) -> float:
        return float(np.percentile(self.latencies_ms, 99)) if self.latencies_ms else 0.0

    def summary(self) -> str:
        return "\n".join([
            f"\n{'='*58}",
            f"  Mode        : {self.mode}",
            f"  Split Layer : {self.split_layer}",
            f"  Concurrency : {self.concurrency} workers",
            f"  Duration    : {self.duration_sec:.1f}s",
            f"  Completed   : {self.completed} requests",
            f"  Errors      : {self.errors}",
            f"  Throughput  : {self.throughput_qps:.2f} req/sec",
            f"  Avg Latency : {self.avg_latency_ms:.2f} ms",
            f"  P50 Latency : {self.p50_latency_ms:.2f} ms",
            f"  P95 Latency : {self.p95_latency_ms:.2f} ms",
            f"  P99 Latency : {self.p99_latency_ms:.2f} ms",
            f"{'='*58}",
        ])

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "split_layer": self.split_layer,
            "concurrency": self.concurrency,
            "duration_sec": self.duration_sec,
            "completed": self.completed,
            "errors": self.errors,
            "throughput_qps": self.throughput_qps,
            "avg_latency_ms": self.avg_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
        }


# ── Request Preparation ───────────────────────────────────────────────────────

def prepare_requests(tokenizer, sentences: List[str], device: str,
                     max_length: int = 128):
    """
    Pre-tokenize all sentences so tokenization doesn't count toward
    server inference time. Returns list of (input_ids, attn_mask) tensors.
    """
    requests = []
    for s in sentences:
        enc = tokenizer(s, return_tensors="pt", truncation=True,
                        max_length=max_length, padding=False)
        requests.append((
            enc["input_ids"].to(device),
            enc["attention_mask"].to(device),
        ))
    return requests


# ── Load Test Core ────────────────────────────────────────────────────────────

def _worker(
    worker_id: int,
    requests: list,
    server_fn: Callable,        # server_fn(input_ids, attn_mask) — the server's work
    duration_sec: float,
    results_q: queue.Queue,
    lock: threading.Lock,
    stop_event: threading.Event,
):
    """
    One worker thread. Continuously sends requests to server_fn until
    stop_event is set. Records completed count and latencies.
    """
    completed = 0
    latencies = []
    errors = 0
    n = len(requests)
    idx = worker_id  # stagger starting positions so workers don't all hit same input

    while not stop_event.is_set():
        input_ids, attn_mask = requests[idx % n]
        idx += 1
        try:
            t0 = time.perf_counter()
            with torch.no_grad():
                server_fn(input_ids, attn_mask)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed_ms)
            completed += 1
        except Exception as e:
            errors += 1

    results_q.put((completed, latencies, errors))


def run_load_test(
    mode: str,
    split_layer: int,
    server_fn: Callable,
    requests: list,
    concurrency: int,
    duration_sec: float = 10.0,
    warmup_sec: float = 2.0,
) -> LoadResult:
    """
    Run concurrent load test against server_fn for duration_sec seconds.
    server_fn(input_ids, attn_mask) represents what the server does per request.
    """
    print(f"[LoadTest] mode={mode} concurrency={concurrency} "
          f"warmup={warmup_sec}s duration={duration_sec}s ...")

    # Warmup — let the GPU/model warm up before measuring
    for input_ids, attn_mask in requests[:5]:
        with torch.no_grad():
            server_fn(input_ids, attn_mask)

    stop_event = threading.Event()
    results_q  = queue.Queue()
    lock       = threading.Lock()

    threads = [
        threading.Thread(
            target=_worker,
            args=(i, requests, server_fn, duration_sec,
                  results_q, lock, stop_event),
            daemon=True,
        )
        for i in range(concurrency)
    ]

    t_start = time.perf_counter()
    for t in threads:
        t.start()

    time.sleep(duration_sec)
    stop_event.set()

    for t in threads:
        t.join(timeout=5.0)

    actual_duration = time.perf_counter() - t_start

    result = LoadResult(
        mode=mode,
        split_layer=split_layer,
        concurrency=concurrency,
        duration_sec=actual_duration,
    )
    while not results_q.empty():
        completed, latencies, errors = results_q.get()
        result.completed += completed
        result.latencies_ms.extend(latencies)
        result.errors    += errors

    return result


# ── Concurrency Sweep ─────────────────────────────────────────────────────────

def run_sweep(
    mode: str,
    split_layer: int,
    server_fn: Callable,
    requests: list,
    concurrency_levels: List[int],
    duration_sec: float = 10.0,
) -> List[LoadResult]:
    results = []
    for c in concurrency_levels:
        r = run_load_test(mode, split_layer, server_fn,
                          requests, c, duration_sec)
        print(f"  concurrency={c:>3}  QPS={r.throughput_qps:>8.2f}  "
              f"p50={r.p50_latency_ms:>7.1f}ms  "
              f"p95={r.p95_latency_ms:>7.1f}ms")
        results.append(r)
    return results


# ── Comparison Printer ────────────────────────────────────────────────────────

def print_comparison(full_results: List[LoadResult],
                     split_results: List[LoadResult]):
    print(f"\n{'='*92}")
    print(f"  SERVER THROUGHPUT: Full Model vs Split Tail")
    print(f"  server_fn = full model forward  vs  tail-only forward")
    print(f"  Higher QPS = server handles more concurrent users")
    print(f"{'='*92}")
    print(f"  {'Concurrency':>12} {'Full QPS':>10} {'Split QPS':>11} "
          f"{'Gain':>8} {'Full P95':>10} {'Split P95':>11} {'Lat Saved':>10}")
    print(f"  {'-'*88}")

    full_by_c  = {r.concurrency: r for r in full_results}
    split_by_c = {r.concurrency: r for r in split_results}

    for c in sorted(full_by_c):
        if c not in split_by_c:
            continue
        f = full_by_c[c]
        s = split_by_c[c]
        gain     = s.throughput_qps / f.throughput_qps if f.throughput_qps > 0 else 0
        lat_save = f.p95_latency_ms - s.p95_latency_ms
        print(f"  {c:>12} {f.throughput_qps:>10.2f} {s.throughput_qps:>11.2f} "
              f"{gain:>7.2f}x {f.p95_latency_ms:>10.1f} {s.p95_latency_ms:>11.1f} "
              f"{lat_save:>+10.1f}ms")

    # Find crossover — where split starts clearly winning
    crossover = None
    for c in sorted(full_by_c):
        if c not in split_by_c:
            continue
        if split_by_c[c].throughput_qps > full_by_c[c].throughput_qps * 1.1:
            crossover = c
            break

    print(f"\n  Throughput gain = split QPS / full QPS (>1.0 means split server wins)")
    if crossover:
        print(f"  Split server pulls ahead at concurrency={crossover}")
    print(f"  Lat Saved = reduction in server-side P95 latency per request")
    print(f"{'='*92}")


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Server throughput load test: full model vs split tail"
    )
    parser.add_argument("--mode",        default="local",
                        choices=["local", "grpc"],
                        help="local=same machine simulation, grpc=real server")
    parser.add_argument("--split-layer", type=int, default=6)
    parser.add_argument("--device",      default=None)
    parser.add_argument("--samples",     type=int, default=100,
                        help="Number of pre-tokenized requests to cycle through")
    parser.add_argument("--concurrency", default="1,2,4,8",
                        help="Comma-separated concurrency levels to sweep")
    parser.add_argument("--duration",    type=float, default=10.0,
                        help="Seconds to run at each concurrency level")
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
    concurrency_levels = [int(x) for x in args.concurrency.split(",")]

    print(f"[LoadTest] device={device} | split_layer={args.split_layer} | "
          f"concurrency={concurrency_levels} | duration={args.duration}s")

    # ── Load sentences ────────────────────────────────────────────────────────
    from Models import load_bert_tokenizer
    tokenizer = load_bert_tokenizer()

    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", args.wikitext, split="test")
        sentences = [
            r["text"].strip() for r in ds
            if len(r["text"].strip()) > 20
            and not r["text"].strip().startswith("=")
        ][:args.samples]
    except Exception:
        sentences = [
            "The tower is 324 metres tall and the tallest structure in Paris.",
            "Scientists discovered a new species of bird in the Amazon rainforest.",
            "The economy grew at three percent during the second quarter of the year.",
            "Engineers designed a bridge capable of withstanding severe weather.",
            "The ancient manuscript was preserved in a climate-controlled vault.",
        ] * (args.samples // 5 + 1)
        sentences = sentences[:args.samples]

    print(f"[LoadTest] Prepared {len(sentences)} sentences.")

    if args.mode == "local":
        # ── Local simulation ──────────────────────────────────────────────────
        # server_fn for "full" = run entire model
        # server_fn for "split" = run tail only (head already done by edge)
        #
        # We pre-run head once to get the hidden states, then the "server"
        # just runs the tail repeatedly on those cached hidden states.
        # This correctly simulates server load: server only sees tail work.

        from Models import BERTFull, load_bert, load_bert_head, load_bert_tail

        print("\n[LoadTest] Loading models...")
        full_model = BERTFull(load_bert(device))
        full_model.eval()

        head = load_bert_head(args.save_dir, args.split_layer, device)
        tail = load_bert_tail(args.save_dir, args.split_layer, device)
        head.eval(); tail.eval()

        # Pre-tokenize for full model requests
        full_requests = prepare_requests(tokenizer, sentences, device)

        # Pre-compute hidden states — these arrive at the server from the edge.
        # The server never sees input_ids in the split scenario.
        print("[LoadTest] Pre-computing hidden states (simulating edge device)...")
        hidden_requests = []
        with torch.no_grad():
            for input_ids, attn_mask in full_requests:
                hidden = head(input_ids, attn_mask)
                hidden_requests.append((hidden, attn_mask))

        # Server functions
        def full_server_fn(input_ids, attn_mask):
            return full_model(input_ids, attn_mask)

        def split_server_fn(hidden, attn_mask):
            return tail(hidden, attn_mask)

        print(f"\n[LoadTest] === FULL SERVER (runs entire model) ===")
        full_results = run_sweep(
            "full_server", 0, full_server_fn,
            full_requests, concurrency_levels, args.duration,
        )

        print(f"\n[LoadTest] === SPLIT SERVER (runs tail only, layers "
              f"{args.split_layer}–end) ===")
        split_results = run_sweep(
            "split_server", args.split_layer, split_server_fn,
            hidden_requests, concurrency_levels, args.duration,
        )

        print_comparison(full_results, split_results)

        # Print individual summaries for the highest concurrency level
        print(full_results[-1].summary())
        print(split_results[-1].summary())

    elif args.mode == "grpc":
        # ── Real gRPC load test ───────────────────────────────────────────────
        # Each worker thread runs head locally and sends hidden state to server.
        # Server runs tail. We measure end-to-end from client perspective,
        # but the key metric is server-side throughput reported by the server.

        from Models import load_bert_head
        from communication.client import SplitInferenceClient

        head = load_bert_head(args.save_dir, args.split_layer, device)
        head.eval()

        full_requests = prepare_requests(tokenizer, sentences, device)

        # Pre-compute hiddens for the split scenario
        print("[LoadTest] Pre-computing hidden states...")
        hidden_requests = []
        with torch.no_grad():
            for input_ids, attn_mask in full_requests:
                hidden = head(input_ids, attn_mask)
                hidden_requests.append((hidden, attn_mask))

        # Each worker needs its own gRPC client (not thread-safe to share)
        def make_grpc_fn(split_layer):
            client = SplitInferenceClient(args.server_host, args.server_port)
            def fn(hidden, attn_mask):
                return client.run_tail(hidden, "bert", split_layer, attn_mask)
            return fn

        print(f"\n[LoadTest] === GRPC SPLIT SERVER (layer {args.split_layer}) ===")
        # For gRPC we do a simple sequential sweep, one concurrency at a time
        split_results = []
        for c in concurrency_levels:
            # Create one gRPC fn per worker
            fns = [make_grpc_fn(args.split_layer) for _ in range(c)]
            # For gRPC, each worker uses its own client fn
            # We pass the same fn list via closure
            fn_idx = [0]
            fn_lock = threading.Lock()
            def grpc_server_fn(hidden, attn_mask):
                with fn_lock:
                    idx = fn_idx[0] % len(fns)
                    fn_idx[0] += 1
                    fn = fns[idx]
                result = fn(hidden, attn_mask)
                return result.logits

            r = run_load_test(
                "grpc_split", args.split_layer, grpc_server_fn,
                hidden_requests, c, args.duration,
            )
            print(f"  concurrency={c:>3}  QPS={r.throughput_qps:>8.2f}  "
                  f"p50={r.p50_latency_ms:>7.1f}ms  "
                  f"p95={r.p95_latency_ms:>7.1f}ms")
            split_results.append(r)

        print("\n[LoadTest] Note: for fair comparison, run a full-model server")
        print("  and compare its QPS at each concurrency level.")

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(
        args.output_dir,
        f"loadtest_{args.mode}_split{args.split_layer}.json"
    )
    out = {"config": vars(args)}
    if args.mode == "local":
        out["full_server"]  = [r.to_dict() for r in full_results]
        out["split_server"] = [r.to_dict() for r in split_results]
    else:
        out["split_server"] = [r.to_dict() for r in split_results]

    with open(save_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[LoadTest] Results saved → {save_path}")


if __name__ == "__main__":
    main()