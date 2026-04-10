from __future__ import annotations

"""
@@@CHANGE@@@
device_profiler.py — Device-level stats: GPU memory, power state, CPU usage.

Adapted from the user's profiler and kept drop-in for the planner.
"""

import psutil
import torch
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict


class PowerState(Enum):
    CHARGING = "charging"
    IN_USE = "in_use"
    IDLE = "idle"
    UNKNOWN = "unknown"


@dataclass
class DeviceSnapshot:
    device_id: str
    power_state: PowerState
    gpu_mem_allocated_mb: float = 0.0
    gpu_mem_reserved_mb: float = 0.0
    gpu_mem_total_mb: float = 0.0
    cpu_percent: float = 0.0
    ram_used_mb: float = 0.0
    ram_total_mb: float = 0.0

    def to_dict(self) -> Dict[str, float | str]:
        data = asdict(self)
        data["power_state"] = self.power_state.value
        return data


class DeviceProfiler:
    def __init__(self, device: str = "cpu"):
        self.device = device

    def snapshot(self) -> DeviceSnapshot:
        gpu_alloc = gpu_reserved = gpu_total = 0.0
        if self.device.startswith("cuda") and torch.cuda.is_available():
            idx = torch.device(self.device).index
            idx = torch.cuda.current_device() if idx is None else idx
            gpu_alloc = torch.cuda.memory_allocated(idx) / 1e6
            gpu_reserved = torch.cuda.memory_reserved(idx) / 1e6
            gpu_total = torch.cuda.get_device_properties(idx).total_memory / 1e6

        vm = psutil.virtual_memory()
        cpu_pct = psutil.cpu_percent(interval=0.1)
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
        try:
            battery = psutil.sensors_battery()
            if battery is None:
                return PowerState.IN_USE
            if battery.power_plugged:
                return PowerState.CHARGING
            cpu_pct = psutil.cpu_percent(interval=None)
            return PowerState.IN_USE if cpu_pct > 10 else PowerState.IDLE
        except Exception:
            return PowerState.UNKNOWN
