from __future__ import annotations

import argparse
import logging
from concurrent import futures

import grpc

from distributed_inference.runtime_partition import RuntimePartition
from distributed_inference.rpc.payload_codec import payload_to_tensor, tensor_to_payload
from distributed_inference.rpc import inference_pb2, inference_pb2_grpc


class PartitionWorkerService(inference_pb2_grpc.PartitionWorkerServicer):
    def __init__(self, partition_dir: str, device: str):
        self.partition = RuntimePartition(partition_dir=partition_dir, device=device)
        self.worker_name = partition_dir

    def Forward(self, request, context):
        input_ids = None
        hidden_states = payload_to_tensor(request.hidden_states)
        attention_mask = payload_to_tensor(request.attention_mask)
        print(f"I am in forward ID: {request.request_id}, Step: {request.step}hidden_state: {hidden_states.shape}, attention_mask:{attention_mask.shape}")


        hidden_states, logits = self.partition.forward_partition(
            input_ids=input_ids,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            return_logits=request.return_logits,
        )

        return inference_pb2.ForwardResponse(
            request_id=request.request_id,
            step=request.step,
            hidden_states=tensor_to_payload(hidden_states, inference_pb2.TensorPayload),
            logits=tensor_to_payload(logits, inference_pb2.TensorPayload),
        )

    def Health(self, request, context):
        return inference_pb2.HealthResponse(
            status="ok",
            worker_name=self.worker_name,
            layer_indices=self.partition.meta.layer_indices,
        )


def serve() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition-dir", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-workers", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.max_workers))
    inference_pb2_grpc.add_PartitionWorkerServicer_to_server(
        PartitionWorkerService(partition_dir=args.partition_dir, device=args.device), server
    )
    server.add_insecure_port(f"{args.host}:{args.port}")
    server.start()
    logging.info("Partition worker started on %s:%s", args.host, args.port)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
