"""
layer_profiler.py — Per-layer profiling: FLOPs, latency, memory.

Design principle: This profiler is NON-INVASIVE. It wraps model layers with
hooks so you don't need to modify model code. This lets us reuse it across
GPT-2, BERT, and future LLMs with zero changes.

Usage:
    profiler = LayerProfiler(model, device="cuda")
    profiler.attach()
    with profiler.record():
        output = model(input_ids)
    metrics = profiler.get_metrics()
    profiler.detach()
"""

import time
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from utils.Metrics import LayerMetrics, get_memory_mb

try:
    from fvcore.nn import FlopCountAnalysis
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False
    print("[Profiler] fvcore not found. FLOPs will be estimated via param count fallback.")


class LayerProfiler:
    """
    Attaches forward hooks to every named module in the model.
    Captures: latency, memory delta, and estimated FLOPs per layer.
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self._hooks = []
        self._layer_records: List[LayerMetrics] = []
        self._recording = False

    # ── Hook Machinery ────────────────────────────────────────────────────────

    def attach(self):
        """Register forward hooks on all named modules."""
        self.detach()  # Clear any previous hooks
        for name, module in self.model.named_modules():
            # Skip the top-level container to avoid double-counting
            if name == "":
                continue
            pre = module.register_forward_pre_hook(self._pre_hook(name))
            post = module.register_forward_hook(self._post_hook(name))
            self._hooks.extend([pre, post])

    def detach(self):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _pre_hook(self, name: str):
        def hook(module, input):
            if not self._recording:
                return
            # Store start time and memory snapshot in the module itself
            module._profiler_start_time = time.perf_counter()
            module._profiler_mem_before = get_memory_mb(self.device)
        return hook

    def _post_hook(self, name: str):
        def hook(module, input, output):
            if not self._recording:
                return
            elapsed_ms = (time.perf_counter() - module._profiler_start_time) * 1000
            mem_after = get_memory_mb(self.device)
            mem_before = getattr(module, "_profiler_mem_before", mem_after)

            lm = LayerMetrics(
                layer_name=name,
                latency_ms=elapsed_ms,
                mem_before_mb=mem_before,
                mem_after_mb=mem_after,
                mem_delta_mb=mem_after - mem_before,
                flops=self._estimate_flops(module, input, output),
            )
            self._layer_records.append(lm)
        return hook

    # ── FLOPs Estimation ──────────────────────────────────────────────────────

    def _estimate_flops(self, module: nn.Module, input, output) -> float:
        """
        Estimate FLOPs for a single layer forward pass.
        Uses fvcore when available; falls back to a simple parameter-count proxy.

        Note: fvcore's FlopCountAnalysis works on full models. For per-layer
        estimation we use the parameter proxy here, and separately run a full-
        model FlopCountAnalysis in profiler.get_total_flops().
        """
        try:
            if isinstance(module, nn.Linear):
                # FLOPs for Linear(in, out): 2 * in * out per token
                if isinstance(input, tuple) and len(input) > 0:
                    x = input[0]
                    if x.dim() >= 2:
                        # batch * seq_len * (2 * in * out)
                        tokens = x.shape[0] * (x.shape[1] if x.dim() == 3 else 1)
                        return float(2 * module.in_features * module.out_features * tokens)
            elif isinstance(module, nn.LayerNorm):
                if isinstance(input, tuple) and len(input) > 0:
                    x = input[0]
                    return float(5 * x.numel())  # mean, var, norm, scale, shift
        except Exception:
            pass
        # Fallback: rough proxy via param count (2 FLOPs per param per token)
        return float(sum(p.numel() for p in module.parameters()) * 2)

    # ── Context Manager ───────────────────────────────────────────────────────

    class _RecordContext:
        def __init__(self, profiler):
            self._p = profiler

        def __enter__(self):
            self._p._layer_records = []
            self._p._recording = True
            return self

        def __exit__(self, *args):
            self._p._recording = False

    def record(self):
        return self._RecordContext(self)

    # ── Results ───────────────────────────────────────────────────────────────

    def get_metrics(self) -> List[LayerMetrics]:
        return list(self._layer_records)

    def get_total_flops_fvcore(self, example_input: dict) -> Optional[float]:
        """
        Run fvcore FlopCountAnalysis on the full model for an accurate total.
        Call this ONCE after a forward pass (not inside the timed loop).
        """
        if not FVCORE_AVAILABLE:
            return None
        try:
            # fvcore needs a tuple of positional args
            flops = FlopCountAnalysis(self.model, (example_input["input_ids"],))
            flops.unsupported_ops_warnings(False)
            flops.uncalled_modules_warnings(False)
            return float(flops.total())
        except Exception as e:
            print(f"[Profiler] fvcore FLOPs failed: {e}")
            return None

    def print_layer_table(self, top_n: int = 20):
        """Pretty-print the top-N layers by latency."""
        # records = sorted(self._layer_records, key=lambda r: r.latency_ms, reverse=True)
        records = self._layer_records

        print(f"\n{'Layer':<50} {'Latency(ms)':>12} {'FLOPs(M)':>10} {'ΔMem(MB)':>10}")
        print("-" * 85)
        for r in records:
            name = r.layer_name[-49:] if len(r.layer_name) > 49 else r.layer_name
            print(f"{name:<50} {r.latency_ms:>12.2f} {r.flops/1e6:>10.2f} {r.mem_delta_mb:>10.2f}")