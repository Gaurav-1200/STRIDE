"""
client.py — gRPC client (edge / head side) for split inference.

Wraps the generated gRPC stub with a clean API that run_split.py uses.
Handles:
  - Channel creation with appropriate message size limits
  - Health checking before benchmarking
  - Unary call (RunTail) for Phase 1 & 2
  - Streaming call (RunTailStreaming) for Phase 3 TTFT measurement
  - Accurate communication latency: round-trip minus server compute time
"""

import os
import sys
import time
import torch

import grpc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from communication import inference_pb2, inference_pb2_grpc
from communication.tensor_utils import (
    tensor_to_bytes, bytes_to_tensor,
    optional_mask_to_bytes, tensor_size_mb, dtype_string
)
from utils.Metrics import timer


# ── Channel Factory ───────────────────────────────────────────────────────────

def make_channel(host: str, port: int) -> grpc.Channel:
    """Create a gRPC channel with large message size limits."""
    options = [
        ("grpc.max_send_message_length",        500 * 1024 * 1024),
        ("grpc.max_receive_message_length",     500 * 1024 * 1024),
        ("grpc.keepalive_time_ms",              30_000),
        ("grpc.keepalive_timeout_ms",           10_000),
        ("grpc.keepalive_permit_without_calls", True),
    ]
    return grpc.insecure_channel(f"{host}:{port}", options=options)


# ── Client Class ──────────────────────────────────────────────────────────────

class SplitInferenceClient:
    """
    gRPC client for split inference.

    Usage:
        client = SplitInferenceClient("192.168.1.100", 50051)
        client.health_check()   # verify server is up

        # Unary call (Phase 1 & 2)
        result = client.run_tail(hidden, model_name="gpt2", split_layer=6)
        logits          = result.logits
        server_metrics  = result.server_metrics   # dict
        comm_latency_ms = result.comm_latency_ms
        tensor_size_mb  = result.tensor_size_mb

        # Streaming call (Phase 3)
        for token in client.run_tail_streaming(hidden, model_name="gpt2", ...):
            print(token.token_text, end="", flush=True)
    """

    def __init__(self, host: str, port: int, timeout: int = 60):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._channel = make_channel(host, port)
        self._stub = inference_pb2_grpc.InferenceServiceStub(self._channel)

    # ── Health Check ──────────────────────────────────────────────────────────

    def health_check(self, verbose: bool = True) -> bool:
        """Ping the server. Returns True if healthy."""
        try:
            resp = self._stub.HealthCheck(
                inference_pb2.HealthRequest(),
                timeout=10,
            )
            if verbose:
                print(f"[Client] Server healthy ✓ | model={resp.model_name} "
                      f"split_layer={resp.split_layer} device={resp.device} "
                      f"gpu_free={resp.gpu_mem_free_mb:.0f}MB")
            return resp.healthy
        except grpc.RpcError as e:
            print(f"[Client] Health check failed: {e.details()}")
            return False

    # ── Unary RPC ─────────────────────────────────────────────────────────────

    def run_tail(
        self,
        hidden: torch.Tensor,
        model_name: str,
        split_layer: int,
        attention_mask: torch.Tensor = None,
    ) -> "TailResult":
        """
        Send hidden state to server, get logits back (unary).

        Returns a TailResult with:
          .logits           : torch.Tensor
          .server_metrics   : dict (latency, FLOPs, memory, per-layer)
          .comm_latency_ms  : float  (round-trip minus server compute)
          .tensor_size_mb   : float  (bytes sent to server)
        """
        # Serialize
        hidden_bytes = tensor_to_bytes(hidden)
        mask_bytes = optional_mask_to_bytes(attention_mask)

        request = inference_pb2.TailRequest(
            hidden_state=hidden_bytes,
            shape=list(hidden.shape),
            dtype=dtype_string(hidden),
            model_name=model_name,
            split_layer=split_layer,
            attention_mask=mask_bytes,
        )

        # Round-trip timing
        with timer("grpc_round_trip") as t:
            response = self._stub.RunTail(request, timeout=self.timeout)

        # Communication time = round-trip − server inference time
        server_ms = response.metrics.e2e_latency_ms
        comm_ms = max(t["elapsed_ms"] - server_ms, 0.0)

        logits = bytes_to_tensor(response.logits)
        server_metrics = self._proto_metrics_to_dict(response.metrics)

        return TailResult(
            logits=logits,
            server_metrics=server_metrics,
            comm_latency_ms=comm_ms,
            tensor_size_mb=len(hidden_bytes) / 1e6,
        )

    # ── Streaming RPC ─────────────────────────────────────────────────────────

    def run_tail_streaming(
        self,
        hidden: torch.Tensor,
        model_name: str,
        split_layer: int,
        max_new_tokens: int = 50,
    ):
        """
        Generator: yields TokenResult for each generated token.

        Usage:
            ttft_ms = None
            for token in client.run_tail_streaming(hidden, "gpt2", 6):
                if token.is_first_token:
                    ttft_ms = token.ttft_ms   # set by the generator
                print(token.text, end="", flush=True)
        """
        hidden_bytes = tensor_to_bytes(hidden)
        request = inference_pb2.TailRequest(
            hidden_state=hidden_bytes,
            shape=list(hidden.shape),
            dtype=dtype_string(hidden),
            model_name=model_name,
            split_layer=split_layer,
            max_new_tokens=max_new_tokens,
        )

        stream_start = time.perf_counter()

        for response in self._stub.RunTailStreaming(request, timeout=self.timeout):
            elapsed_ms = (time.perf_counter() - stream_start) * 1000
            yield TokenResult(
                token_id=response.token_id,
                text=response.token_text,
                is_first_token=response.is_first_token,
                is_last_token=response.is_last_token,
                step_latency_ms=response.step_latency_ms,
                ttft_ms=elapsed_ms if response.is_first_token else None,
                server_metrics=(
                    self._proto_metrics_to_dict(response.metrics)
                    if response.is_last_token else None
                ),
            )

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def close(self):
        self._channel.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _proto_metrics_to_dict(metrics: inference_pb2.ServerMetrics) -> dict:
        return {
            "e2e_latency_ms":  metrics.e2e_latency_ms,
            "total_flops":     metrics.total_flops,
            "peak_mem_mb":     metrics.peak_mem_mb,
            "device":          metrics.device,
            "power_state":     metrics.power_state,
            "layer_metrics": [
                {
                    "layer_name":  lm.layer_name,
                    "latency_ms":  lm.latency_ms,
                    "flops":       lm.flops,
                    "mem_delta_mb": lm.mem_delta_mb,
                }
                for lm in metrics.layer_metrics
            ],
        }


# ── Result Types ──────────────────────────────────────────────────────────────

class TailResult:
    __slots__ = ["logits", "server_metrics", "comm_latency_ms", "tensor_size_mb"]

    def __init__(self, logits, server_metrics, comm_latency_ms, tensor_size_mb):
        self.logits = logits
        self.server_metrics = server_metrics
        self.comm_latency_ms = comm_latency_ms
        self.tensor_size_mb = tensor_size_mb


class TokenResult:
    __slots__ = ["token_id", "text", "is_first_token", "is_last_token",
                 "step_latency_ms", "ttft_ms", "server_metrics"]

    def __init__(self, token_id, text, is_first_token, is_last_token,
                 step_latency_ms, ttft_ms, server_metrics):
        self.token_id = token_id
        self.text = text
        self.is_first_token = is_first_token
        self.is_last_token = is_last_token
        self.step_latency_ms = step_latency_ms
        self.ttft_ms = ttft_ms
        self.server_metrics = server_metrics