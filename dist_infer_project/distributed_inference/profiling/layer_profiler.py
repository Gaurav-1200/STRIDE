from __future__ import annotations

"""
@@@CHANGE@@@
layer_profiler.py — Per-layer profiling: FLOPs, latency, memory.

Adapted from the user's profiler so it is self-contained inside this repo and can
be consumed directly by the partition planner.
"""

import time
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import torch
import torch.nn as nn

try:
    from fvcore.nn import FlopCountAnalysis
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False


@dataclass
class LayerMetrics:
    """@@@CHANGE@@@ Standardized profiler payload for planner consumption."""
    layer_name: str
    latency_ms: float
    mem_before_mb: float
    mem_after_mb: float
    mem_delta_mb: float
    flops: float

    def to_dict(self) -> Dict[str, float | str]:
        return asdict(self)


def get_memory_mb(device: str | torch.device) -> float:
    """@@@CHANGE@@@ Self-contained memory helper replacing utils.Metrics dependency."""
    device_str = str(device)
    if device_str.startswith("cuda") and torch.cuda.is_available():
        idx = torch.device(device_str).index
        idx = torch.cuda.current_device() if idx is None else idx
        return torch.cuda.memory_allocated(idx) / 1e6
    return 0.0


class LayerProfiler:
    """
    @@@CHANGE@@@
    Attaches forward hooks to named modules.
    Captures latency, memory delta and estimated FLOPs.

    This remains non-invasive, but now also supports optional layer-name filtering
    so the planner can aggregate only transformer-block entries when desired.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        include_modules: Optional[List[str]] = None,
    ):
        self.model = model
        self.device = device
        self.include_modules = include_modules
        self._hooks = []
        self._layer_records: List[LayerMetrics] = []
        self._recording = False

    def attach(self):
        """Register hooks on all named modules or only a selected subset."""
        self.detach()
        for name, module in self.model.named_modules():
            if name == "":
                continue
            if self.include_modules is not None and name not in self.include_modules:
                continue
            pre = module.register_forward_pre_hook(self._pre_hook(name))
            post = module.register_forward_hook(self._post_hook(name))
            self._hooks.extend([pre, post])

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _pre_hook(self, name: str):
        def hook(module, inputs):
            if not self._recording:
                return
            if str(self.device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
            module._profiler_start_time = time.perf_counter()
            module._profiler_mem_before = get_memory_mb(self.device)
        return hook

    def _post_hook(self, name: str):
        def hook(module, inputs, output):
            if not self._recording:
                return
            if str(self.device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
            elapsed_ms = (time.perf_counter() - module._profiler_start_time) * 1000
            mem_after = get_memory_mb(self.device)
            mem_before = getattr(module, "_profiler_mem_before", mem_after)
            self._layer_records.append(
                LayerMetrics(
                    layer_name=name,
                    latency_ms=elapsed_ms,
                    mem_before_mb=mem_before,
                    mem_after_mb=mem_after,
                    mem_delta_mb=mem_after - mem_before,
                    flops=self._estimate_flops(module, inputs, output),
                )
            )
        return hook

    def _estimate_flops(self, module: nn.Module, inputs, output) -> float:
        try:
            if isinstance(module, nn.Linear):
                if isinstance(inputs, tuple) and len(inputs) > 0:
                    x = inputs[0]
                    if hasattr(x, "dim") and x.dim() >= 2:
                        tokens = x.shape[0] * (x.shape[1] if x.dim() == 3 else 1)
                        return float(2 * module.in_features * module.out_features * tokens)
            elif isinstance(module, nn.LayerNorm):
                if isinstance(inputs, tuple) and len(inputs) > 0:
                    x = inputs[0]
                    return float(5 * x.numel())
        except Exception:
            pass
        return float(sum(p.numel() for p in module.parameters()) * 2)

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

    def get_metrics(self) -> List[LayerMetrics]:
        return list(self._layer_records)

    def get_total_flops_fvcore(self, example_input: dict) -> Optional[float]:
        if not FVCORE_AVAILABLE:
            return None
        try:
            flops = FlopCountAnalysis(self.model, (example_input["input_ids"],))
            flops.unsupported_ops_warnings(False)
            flops.uncalled_modules_warnings(False)
            return float(flops.total())
        except Exception:
            return None

    def print_layer_table(self):
        print(f"\n{'Layer':<50} {'Latency(ms)':>12} {'FLOPs(M)':>10} {'ΔMem(MB)':>10}")
        print("-" * 85)
        for r in self._layer_records:
            name = r.layer_name[-49:] if len(r.layer_name) > 49 else r.layer_name
            print(f"{name:<50} {r.latency_ms:>12.2f} {r.flops/1e6:>10.2f} {r.mem_delta_mb:>10.2f}")
