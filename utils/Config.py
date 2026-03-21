"""
config.py — Central configuration for split inference experiments.
Edit this file to change split points, server address, model choice, etc.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InferenceConfig:
    # ── Model ─────────────────────────────────────────────────────────────────
    model_name: str = "gpt2"          # "gpt2" | "bert-base-uncased"
    max_new_tokens: int = 50           # Only relevant for GPT-2 generation
    max_length: int = 128              # Max sequence length

    # ── Split ─────────────────────────────────────────────────────────────────
    # For GPT-2 Small: 12 transformer blocks (layers 0–11).
    # split_layer=6 means layers 0–5 run on the client, 6–11 on the server.
    # split_layer=0 means everything runs on the server (useful for testing).
    split_layer: int = 6

    # ── Network ───────────────────────────────────────────────────────────────
    server_host: str = "127.0.0.1"    # IP of the machine running server.py
    server_port: int = 9999
    socket_timeout: int = 60          # seconds

    # ── Device ────────────────────────────────────────────────────────────────
    # "cuda" if GPU available, else "cpu". Each machine auto-detects.
    device: Optional[str] = None      # None = auto-detect

    # ── Profiling ─────────────────────────────────────────────────────────────
    profile_layers: bool = True        # Capture per-layer FLOPs + memory
    warmup_runs: int = 2               # Discarded runs before measurement
    timed_runs: int = 5                # Averaged over this many runs

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir: str = "./results"


# Singleton used across scripts
DEFAULT_CONFIG = InferenceConfig()