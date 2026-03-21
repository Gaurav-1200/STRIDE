"""
tensor_utils.py — Shared helpers for tensor ↔ protobuf bytes conversion.

Both server.py and client.py import from here so serialization logic
lives in exactly one place.

Design: we use torch.save/load with a BytesIO buffer.
  - Preserves dtype, layout, and device-agnosticism.
  - More compact than hex encoding (was used in the socket version).
  - For float16/bfloat16 models (Phase 3) this matters a lot — half the bytes.
"""

import io
import torch
from typing import Optional


def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """Serialize a tensor to raw bytes using torch.save."""
    buf = io.BytesIO()
    torch.save(tensor.cpu().contiguous(), buf)
    return buf.getvalue()


def bytes_to_tensor(data: bytes, device: str = "cpu") -> torch.Tensor:
    """Deserialize a tensor from raw bytes."""
    buf = io.BytesIO(data)
    tensor = torch.load(buf, weights_only=True)
    return tensor.to(device)


def tensor_size_mb(tensor: torch.Tensor) -> float:
    """Return the in-memory size of a tensor in MB."""
    return tensor.element_size() * tensor.nelement() / 1e6


def dtype_string(tensor: torch.Tensor) -> str:
    return str(tensor.dtype).replace("torch.", "")


def optional_mask_to_bytes(mask: Optional[torch.Tensor]) -> bytes:
    """Serialize attention mask, or return empty bytes if None."""
    if mask is None:
        return b""
    return tensor_to_bytes(mask)


def bytes_to_optional_mask(data: bytes, device: str = "cpu") -> Optional[torch.Tensor]:
    """Deserialize attention mask, or return None if data is empty."""
    if not data:
        return None
    return bytes_to_tensor(data, device)