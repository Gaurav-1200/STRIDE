"""
device_profiler.py — Device-level stats: GPU memory, power state, CPU usage.

This feeds into the DAG cost model in Phase 4/5. For now it just snapshots
device health so you have the data when you need it.
"""

import torch
import psutil
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class PowerState(Enum):
    """Discrete power states as described in the proposal."""
    CHARGING = "charging"
    IN_USE = "in_use"
    IDLE = "idle"
    UNKNOWN = "unknown"


@dataclass
class DeviceSnapshot:
    device_id: str                   # e.g. "cuda:0" or "cpu"
    power_state: PowerState
    gpu_mem_allocated_mb: float = 0.0
    gpu_mem_reserved_mb: float = 0.0
    gpu_mem_total_mb: float = 0.0
    cpu_percent: float = 0.0
    ram_used_mb: float = 0.0
    ram_total_mb: float = 0.0

    def to_dict(self):
        d = {k: v for k, v in self.__dict__.items()}
        d["power_state"] = self.power_state.value
        return d


class DeviceProfiler:
    def __init__(self, device: str = "cpu"):
        self.device = device

    def snapshot(self) -> DeviceSnapshot:
        """Take a point-in-time snapshot of device resource usage."""
        gpu_alloc = gpu_reserved = gpu_total = 0.0

        if self.device.startswith("cuda") and torch.cuda.is_available():
            idx = torch.cuda.current_device()
            gpu_alloc = torch.cuda.memory_allocated(idx) / 1e6
            gpu_reserved = torch.cuda.memory_reserved(idx) / 1e6
            gpu_total = torch.cuda.get_device_properties(idx).total_memory / 1e6

        vm = psutil.virtual_memory()
        cpu_pct = psutil.cpu_percent(interval=0.1)

        # Power state heuristic for laptops (battery info if available)
        power_state = self._detect_power_state()

        return DeviceSnapshot(
            device_id=self.device,
            power_state=power_state,
            gpu_mem_allocated_mb=gpu_alloc,
            gpu_mem_reserved_mb=gpu_reserved,
            gpu_mem_total_mb=gpu_total,
            cpu_percent=cpu_pct,
            ram_used_mb=vm.used / 1e6,
            ram_total_mb=vm.total / 1e6,
        )

    def _detect_power_state(self) -> PowerState:
        """
        Heuristic power state detection.
        - On systems with battery: check if plugged in.
        - Otherwise assume IN_USE (server / desktop scenario).
        """
        try:
            battery = psutil.sensors_battery()
            if battery is None:
                return PowerState.IN_USE   # No battery → desktop/server
            if battery.power_plugged:
                return PowerState.CHARGING
            cpu_pct = psutil.cpu_percent(interval=None)
            return PowerState.IN_USE if cpu_pct > 10 else PowerState.IDLE
        except Exception:
            return PowerState.UNKNOWN

    def print_snapshot(self):
        s = self.snapshot()
        print(f"\n── Device Snapshot ({s.device_id}) ─────────────────────")
        print(f"  Power State : {s.power_state.value}")
        if s.gpu_mem_total_mb > 0:
            pct = s.gpu_mem_allocated_mb / s.gpu_mem_total_mb * 100
            print(f"  GPU Memory  : {s.gpu_mem_allocated_mb:.0f} / {s.gpu_mem_total_mb:.0f} MB ({pct:.1f}%)")
        print(f"  CPU Usage   : {s.cpu_percent:.1f}%")
        ram_pct = s.ram_used_mb / s.ram_total_mb * 100 if s.ram_total_mb > 0 else 0
        print(f"  RAM         : {s.ram_used_mb:.0f} / {s.ram_total_mb:.0f} MB ({ram_pct:.1f}%)")
        print(f"────────────────────────────────────────────────────")