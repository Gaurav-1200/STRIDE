from __future__ import annotations

"""@@@CHANGE@@@ Gather device and network profiles for all workers."""

import argparse
import json
from pathlib import Path

from distributed_inference.config import ClusterConfig
from distributed_inference.profiling.network_profiler import NetworkProfiler
from distributed_inference.rpc import inference_pb2, inference_pb2_grpc
import grpc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--payload-bytes", type=int, default=1 << 20)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    cfg = ClusterConfig.load(args.config)
    records = {"model_id": cfg.model_id, "workers": []}

    for worker in cfg.workers:
        channel = grpc.insecure_channel(f"{worker.host}:{worker.port}")
        stub = inference_pb2_grpc.PartitionWorkerStub(channel)
        device_resp = stub.ProfileDevice(inference_pb2.DeviceProfileRequest())
        net_prof = NetworkProfiler(worker.host, worker.port)
        net_snapshot = net_prof.snapshot(payload_bytes=args.payload_bytes, samples=args.samples)
        records["workers"].append(
            {
                "name": worker.name,
                "host": worker.host,
                "port": worker.port,
                "partition_dir": worker.partition_dir,
                "layer_indices": list(device_resp.layer_indices),
                "device_snapshot": {
                    "device_id": device_resp.device_id,
                    "power_state": device_resp.power_state,
                    "gpu_mem_allocated_mb": device_resp.gpu_mem_allocated_mb,
                    "gpu_mem_reserved_mb": device_resp.gpu_mem_reserved_mb,
                    "gpu_mem_total_mb": device_resp.gpu_mem_total_mb,
                    "cpu_percent": device_resp.cpu_percent,
                    "ram_used_mb": device_resp.ram_used_mb,
                    "ram_total_mb": device_resp.ram_total_mb,
                },
                "network_snapshot": net_snapshot.to_dict(),
            }
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Wrote cluster profile to {out_path}")


if __name__ == "__main__":
    main()
