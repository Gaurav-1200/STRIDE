"""
metrics.py — Helpers for timing, memory, and aggregating profiling results.
"""

import time
import json
import os
import torch
import psutil
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class LayerMetrics:
    layer_name: str
    flops: float = 0.0          # Floating point ops (from fvcore / thop)
    latency_ms: float = 0.0     # Wall-clock time for this layer
    mem_before_mb: float = 0.0  # GPU/RAM allocated before layer
    mem_after_mb: float = 0.0   # GPU/RAM allocated after layer
    mem_delta_mb: float = 0.0   # Difference (can be negative due to GC)


@dataclass
class RunMetrics:
    run_type: str                          # "baseline" | "split_client" | "split_server"
    model_name: str
    split_layer: int
    device: str

    # E2E timing
    ttft_ms: float = 0.0                   # Time-to-first-token (generation only)
    e2e_latency_ms: float = 0.0            # Total wall-clock time

    # Compute
    total_flops: float = 0.0

    # Memory (this device only)
    peak_mem_mb: float = 0.0
    total_mem_mb: float = 0.0             # Sum across all layers

    # Communication (split runs only)
    tensor_size_bytes: int = 0
    comm_latency_ms: float = 0.0

    # Per-layer breakdown
    layer_metrics: List[LayerMetrics] = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        return d

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[Metrics] Saved → {path}")

    def summary(self) -> str:
        lines = [
            f"\n{'='*55}",
            f"  Run Type    : {self.run_type}",
            f"  Model       : {self.model_name}",
            f"  Split Layer : {self.split_layer}",
            f"  Device      : {self.device}",
            f"  E2E Latency : {self.e2e_latency_ms:.1f} ms",
        ]
        if self.ttft_ms > 0:
            lines.append(f"  TTFT        : {self.ttft_ms:.1f} ms")
        lines += [
            f"  Total FLOPs : {self.total_flops / 1e9:.2f} GFLOPs",
            f"  Peak Memory : {self.peak_mem_mb:.1f} MB",
        ]
        if self.tensor_size_bytes > 0:
            lines.append(f"  Tensor Size : {self.tensor_size_bytes / 1e6:.2f} MB")
            lines.append(f"  Comm Latency: {self.comm_latency_ms:.1f} ms")
        lines.append(f"{'='*55}")
        return "\n".join(lines)


# ── Context Managers ──────────────────────────────────────────────────────────

@contextmanager
def timer(label: str = "", verbose: bool = False):
    """Simple wall-clock timer. Yields a dict with key 'elapsed_ms'."""
    result = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed_ms"] = (time.perf_counter() - start) * 1000
        if verbose:
            print(f"[Timer] {label}: {result['elapsed_ms']:.2f} ms")


def get_memory_mb(device: str) -> float:
    """Return currently allocated memory in MB for the given device."""
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e6
    else:
        # CPU: use process RSS as a proxy
        return psutil.Process(os.getpid()).memory_info().rss / 1e6


def get_peak_memory_mb(device: str) -> float:
    """Return peak allocated memory in MB (GPU only; CPU returns current)."""
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    return get_memory_mb(device)


def reset_peak_memory(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ── Comparison Utility ────────────────────────────────────────────────────────

def compare_runs(baseline: RunMetrics, split_client: RunMetrics,
                 split_server: Optional[RunMetrics] = None) -> Dict:
    """
    Compare baseline vs. split run and compute reduction percentages.
    split_server metrics can be passed in separately if collected remotely.
    """
    combined_latency = split_client.e2e_latency_ms + split_client.comm_latency_ms
    combined_flops = split_client.total_flops + (
        split_server.total_flops if split_server else 0.0
    )
    server_flop_reduction = (
        ((baseline.total_flops - split_server.total_flops) / baseline.total_flops * 100)
        if split_server and baseline.total_flops > 0 else None
    )

    report = {
        "latency_reduction_pct": (
            (baseline.e2e_latency_ms - combined_latency) /
            baseline.e2e_latency_ms * 100
        ) if baseline.e2e_latency_ms > 0 else None,
        "flop_reduction_on_server_pct": server_flop_reduction,
        "baseline_peak_mem_mb": baseline.peak_mem_mb,
        "client_peak_mem_mb": split_client.peak_mem_mb,
        "server_peak_mem_mb": split_server.peak_mem_mb if split_server else None,
        "comm_overhead_ms": split_client.comm_latency_ms,
        "tensor_payload_mb": split_client.tensor_size_bytes / 1e6,
    }
    return report