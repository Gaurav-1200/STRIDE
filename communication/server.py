"""
server.py — gRPC server (cloud / tail side) for split inference.

Implements the InferenceService defined in proto/inference.proto:
  - RunTail          : unary RPC  → used in Phase 1 & 2
  - RunTailStreaming  : server-streaming RPC → used in Phase 3 for true TTFT
  - HealthCheck       : liveness probe

Usage:
    # First time only — compile the proto:
    python communication/proto/generate_proto.py

    # Start the server (machine B):
    python communication/server.py --model gpt2 --split-layer 6 --port 50051

The client (machine A) connects via run_split.py or directly via client.py.
"""

import argparse
import os
import sys
import time
import threading
import torch

import grpc
from concurrent import futures

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from communication import inference_pb2, inference_pb2_grpc
from communication.tensor_utils import (
    bytes_to_tensor, tensor_to_bytes, bytes_to_optional_mask
)
from profiler.LayerProfiler import LayerProfiler
from profiler.DeviceProfiler import DeviceProfiler
from utils.Metrics import get_peak_memory_mb, reset_peak_memory, timer


# ── Model Loader ──────────────────────────────────────────────────────────────

def load_tail(model_name: str, split_layer: int, device: str, save_dir: str):
    if model_name in ("bert", "bert-base-uncased"):
        from Models import BERTTail,load_bert_tail
        base = load_bert_tail(save_dir, split_layer, device)
        return BERTTail(base,split_layer)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ── Metrics Builder ───────────────────────────────────────────────────────────

def build_server_metrics_proto(elapsed_ms, total_flops, peak_mem_mb,
                                device, power_state, layer_records):
    layer_protos = [
        inference_pb2.LayerMetric(
            layer_name=lm.layer_name,
            latency_ms=lm.latency_ms,
            flops=lm.flops,
            mem_delta_mb=lm.mem_delta_mb,
        )
        for lm in layer_records
    ]
    return inference_pb2.ServerMetrics(
        e2e_latency_ms=elapsed_ms,
        total_flops=total_flops,
        peak_mem_mb=peak_mem_mb,
        device=device,
        power_state=power_state,
        layer_metrics=layer_protos,
    )


# ── gRPC Servicer ─────────────────────────────────────────────────────────────

class InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
    """
    Implements all RPCs defined in inference.proto.

    Thread-safety: gRPC runs each RPC in its own thread from the thread pool.
    The tail model and profilers are shared — we serialize with a lock.
    PyTorch inference with no_grad is thread-safe, but the profiler hooks
    are not reentrant, hence the lock.
    """

    def __init__(self, model_name: str, split_layer: int, device: str, save_dir: str):
        self.model_name = model_name
        self.split_layer = split_layer
        self.device = device
        self._lock = threading.Lock()

        print(f"[Server] Loading tail: {model_name} (layers {split_layer}→end) on {device}")
        self.tail = load_tail(model_name, split_layer, device, save_dir)
        self.tail.eval()

        self.layer_profiler = LayerProfiler(self.tail, device)
        self.dev_profiler = DeviceProfiler(device)

        print("[Server] Ready.")
        self.dev_profiler.print_snapshot()

    # ── HealthCheck ───────────────────────────────────────────────────────────

    def HealthCheck(self, request, context):
        snap = self.dev_profiler.snapshot()
        gpu_free = (
            snap.gpu_mem_total_mb - snap.gpu_mem_allocated_mb
            if snap.gpu_mem_total_mb > 0 else -1.0
        )
        return inference_pb2.HealthResponse(
            healthy=True,
            model_name=self.model_name,
            split_layer=self.split_layer,
            device=self.device,
            gpu_mem_free_mb=gpu_free,
        )

    # ── RunTail (Unary) ───────────────────────────────────────────────────────

    def RunTail(self, request, context):
        """
        Unary RPC: hidden state → tail layers → logits.
        Used for Phase 1 (GPT-2 forward pass) and Phase 2 (BERT MLM).
        """
        print(f"[Server] RunTail | shape={list(request.shape)} "
              f"dtype={request.dtype} model={request.model_name}")

        with self._lock:
            hidden = bytes_to_tensor(request.hidden_state, self.device)
            attn_mask = bytes_to_optional_mask(request.attention_mask, self.device)

            reset_peak_memory(self.device)
            self.layer_profiler.attach()

            with timer("tail") as t:
                with self.layer_profiler.record():
                    with torch.no_grad():
                        if request.model_name in ("bert", "bert-base-uncased"):
                            logits = self.tail(hidden, attn_mask)
                        else:
                            logits = self.tail(hidden)

            self.layer_profiler.detach()
            layer_records = self.layer_profiler.get_metrics()

        peak_mem = get_peak_memory_mb(self.device)
        total_flops = sum(lm.flops for lm in layer_records)
        snap = self.dev_profiler.snapshot()

        print(f"[Server] Done | {t['elapsed_ms']:.1f} ms | "
              f"{total_flops/1e9:.2f} GFLOPs | {peak_mem:.0f} MB")
        self.layer_profiler.print_layer_table(top_n=8)

        metrics_proto = build_server_metrics_proto(
            elapsed_ms=t["elapsed_ms"],
            total_flops=total_flops,
            peak_mem_mb=peak_mem,
            device=self.device,
            power_state=snap.power_state.value,
            layer_records=layer_records,
        )

        return inference_pb2.TailResponse(
            logits=tensor_to_bytes(logits),
            logits_shape=list(logits.shape),
            metrics=metrics_proto,
        )

    # ── RunTailStreaming (Server-streaming) ───────────────────────────────────

    def RunTailStreaming(self, request, context):
        """
        Server-streaming RPC for autoregressive GPT-2 token generation.
        Yields one TokenResponse per token — client measures TTFT on arrival
        of the first yielded message.

        Note: This is a greedy, KV-cache-free implementation for Phase 1.
        Phase 3 will add proper KV-cache so each step is O(seq) not O(seq²).
        """
        if request.model_name not in ("gpt2",):
            context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                "Streaming only supported for GPT-2. Use RunTail for BERT."
            )
            return

        max_new_tokens = request.max_new_tokens or 50

        # Load tokenizer once for token text decoding
        try:
            from models.gpt2_splittable import load_tokenizer
            tokenizer = load_tokenizer()
        except Exception:
            tokenizer = None

        with self._lock:
            hidden = bytes_to_tensor(request.hidden_state, self.device)
            tail_blocks = self.tail.h
            ln_f = self.tail.ln_f
            lm_head = self.tail.lm_head

            step_latencies = []
            is_first = True

            for step in range(max_new_tokens):
                if not context.is_active():
                    break

                t0 = time.perf_counter()
                with torch.no_grad():
                    h = hidden
                    for block in tail_blocks:
                        h = block(h)[0]
                    h = ln_f(h)
                    logits = lm_head(h)

                next_id = int(logits[0, -1, :].argmax())
                step_ms = (time.perf_counter() - t0) * 1000
                step_latencies.append(step_ms)

                token_text = ""
                if tokenizer:
                    try:
                        token_text = tokenizer.decode([next_id])
                    except Exception:
                        token_text = f"<{next_id}>"

                eos = 50256  # GPT-2 EOS token id
                is_last = (step == max_new_tokens - 1) or (next_id == eos)

                # Only attach full metrics to the final token
                metrics_proto = inference_pb2.ServerMetrics()
                if is_last:
                    snap = self.dev_profiler.snapshot()
                    metrics_proto = inference_pb2.ServerMetrics(
                        e2e_latency_ms=sum(step_latencies),
                        peak_mem_mb=get_peak_memory_mb(self.device),
                        device=self.device,
                        power_state=snap.power_state.value,
                    )

                yield inference_pb2.TokenResponse(
                    token_id=next_id,
                    token_text=token_text,
                    is_first_token=is_first,
                    is_last_token=is_last,
                    step_latency_ms=step_ms,
                    metrics=metrics_proto,
                )

                is_first = False
                if is_last:
                    break

                # Phase 1 limitation: no KV-cache, so we only generate 1 token
                # per hidden state sent. Full autoregression added in Phase 3.
                break


# ── Server Startup ────────────────────────────────────────────────────────────

def serve(model_name: str, split_layer: int, port: int,
          device: str, max_workers: int = 4, save_dir: str = None):

    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), "assets", "BERT")
    
    servicer = InferenceServicer(model_name, split_layer, device, save_dir)

    # 500 MB message limit — needed for large hidden states / logits
    options = [
        ("grpc.max_send_message_length",        500 * 1024 * 1024),
        ("grpc.max_receive_message_length",     500 * 1024 * 1024),
        ("grpc.keepalive_time_ms",              30_000),
        ("grpc.keepalive_timeout_ms",           10_000),
        ("grpc.keepalive_permit_without_calls", True),
    ]

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=options,
    )
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()

    print(f"\n[Server] gRPC listening on :{port}")
    print(f"[Server] model={model_name} split_layer={split_layer} device={device}\n")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down...")
        server.stop(grace=5)


def main():
    parser = argparse.ArgumentParser(description="gRPC Split Inference Server")
    parser.add_argument("--model",       default="bert", choices=["bert"])
    parser.add_argument("--split-layer", type=int, default=6)
    parser.add_argument("--port",        type=int, default=50051)
    parser.add_argument("--device",      default=None)
    parser.add_argument("--workers",     type=int, default=4)
    parser.add_argument("--save-dir",   type=str, default="assets/BERT")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    serve(args.model, args.split_layer, args.port, device, args.workers, args.save_dir)


if __name__ == "__main__":
    main()