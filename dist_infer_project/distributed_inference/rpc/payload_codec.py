from __future__ import annotations

import torch


def tensor_to_payload(tensor, pb_cls):
    if tensor is None:
        return pb_cls(dtype="", shape=[], data=b"")
    arr = tensor.detach().cpu().contiguous()
    return pb_cls(dtype=str(arr.numpy().dtype), shape=list(arr.shape), data=arr.numpy().tobytes())


def payload_to_tensor(payload):
    if not payload.dtype:
        return None
    import numpy as np

    np_dtype = np.dtype(payload.dtype)
    arr = np.frombuffer(payload.data, dtype=np_dtype).reshape(payload.shape)
    return torch.from_numpy(arr.copy())
