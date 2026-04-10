from __future__ import annotations

"""
@@@CHANGE@@@
network_profiler.py — gRPC-based network profiling for per-link RTT and transfer throughput.

The profiler uses the same worker service already required for inference, so the
numbers reflect the real stack: protobuf serialization + gRPC + local network.
"""

import statistics
import time
from dataclasses import dataclass, asdict
from typing import Dict, List

import grpc
import numpy as np

from distributed_inference.rpc import inference_pb2, inference_pb2_grpc


@dataclass
class NetworkSnapshot:
    target: str
    payload_bytes: int
    median_rtt_ms: float
    p95_rtt_ms: float
    throughput_mbps: float
    samples: int

    def to_dict(self) -> Dict[str, float | int | str]:
        return asdict(self)


class NetworkProfiler:
    """@@@CHANGE@@@ Probe one remote worker using Ping RPC."""

    def __init__(self, host: str, port: int, timeout_s: float = 10.0):
        self.target = f"{host}:{port}"
        self.channel = grpc.insecure_channel(self.target)
        self.stub = inference_pb2_grpc.PartitionWorkerStub(self.channel)
        self.timeout_s = timeout_s

    def snapshot(self, payload_bytes: int = 1 << 20, samples: int = 5) -> NetworkSnapshot:
        latencies_ms: List[float] = []
        blob = np.random.bytes(payload_bytes)

        for _ in range(samples):
            start = time.perf_counter()
            resp = self.stub.Ping(
                inference_pb2.PingRequest(payload=blob, echo_payload=True),
                timeout=self.timeout_s,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if resp.ack_bytes != payload_bytes:
                raise RuntimeError(
                    f"Ping payload mismatch for {self.target}: sent={payload_bytes}, ack={resp.ack_bytes}"
                )
            latencies_ms.append(elapsed_ms)

        median_rtt = statistics.median(latencies_ms)
        p95_rtt = max(latencies_ms) if len(latencies_ms) < 20 else float(np.percentile(latencies_ms, 95))
        throughput_mbps = (payload_bytes * 8 / 1e6) / max(median_rtt / 1000.0, 1e-9)
        return NetworkSnapshot(
            target=self.target,
            payload_bytes=payload_bytes,
            median_rtt_ms=median_rtt,
            p95_rtt_ms=p95_rtt,
            throughput_mbps=throughput_mbps,
            samples=samples,
        )
