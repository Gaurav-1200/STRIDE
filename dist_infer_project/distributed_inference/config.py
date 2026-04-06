from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml


@dataclass
class WorkerSpec:
    name: str
    host: str
    port: int
    partition_dir: str


@dataclass
class ClusterConfig:
    model_id: str
    dtype: str
    max_new_tokens: int
    workers: List[WorkerSpec]

    @staticmethod
    def load(path: str | Path) -> "ClusterConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        workers = [WorkerSpec(**w) for w in raw["workers"]]
        return ClusterConfig(
            model_id=raw["model_id"],
            dtype=raw.get("dtype", "float16"),
            max_new_tokens=raw.get("max_new_tokens", 64),
            workers=workers,
        )
