from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import List

import grpc
import torch
from transformers import AutoTokenizer

from distributed_inference.config import ClusterConfig
from distributed_inference.runtime_partition import RuntimePartition
from distributed_inference.rpc.payload_codec import payload_to_tensor, tensor_to_payload
from distributed_inference.rpc import inference_pb2, inference_pb2_grpc


class ClusterCoordinator:
    def __init__(self, config_path: str, local_first_partition_dir: str, options:dict, device: str = "cuda"):
        self.cfg = ClusterConfig.load(config_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id, trust_remote_code=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.first_partition = RuntimePartition(local_first_partition_dir, device=device)
        self.stubs = []
        for w in self.cfg.workers[1:]:
            channel = grpc.insecure_channel(f"{w.host}:{w.port}",options=options)
            self.stubs.append(inference_pb2_grpc.PartitionWorkerStub(channel))

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        print("I got a request!!")
        max_new_tokens = max_new_tokens or self.cfg.max_new_tokens
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        generated = input_ids.clone()
        for step in range(max_new_tokens):
            hidden_states, _ = self.first_partition.forward_partition(
                input_ids=generated,
                hidden_states=None,
                attention_mask=attention_mask,
                return_logits=False,
            )

            request_id = str(uuid.uuid4())
            logits = None

            for i, stub in enumerate(self.stubs):
                is_last_node = (i == len(self.stubs) - 1)
                req = inference_pb2.ForwardRequest(
                    request_id=request_id,
                    step=step,
                    is_prefill=(step == 0),
                    return_logits=is_last_node,
                    hidden_states=tensor_to_payload(hidden_states, inference_pb2.TensorPayload),
                    attention_mask=tensor_to_payload(attention_mask, inference_pb2.TensorPayload),
                )
                print(f"Sending Step {step} to Node {i}...  EOS ID: {self.tokenizer.eos_token_id}")
                resp = stub.Forward(req)
                hidden_states = payload_to_tensor(resp.hidden_states)
                if is_last_node:
                    logits = payload_to_tensor(resp.logits)

            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token.cpu()], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype)], dim=1
            )
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(generated[0], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--local-first-partition-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    MAX_MESSAGE_LENGTH = 128 * 1024 * 1024 

    options = [
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
    ]

    coordinator = ClusterCoordinator(
        config_path=args.config,
        local_first_partition_dir=args.local_first_partition_dir,
        options=options,
        device=args.device,
    )
    out = coordinator.generate(prompt=args.prompt, max_new_tokens=args.max_new_tokens)
    print("---",out)


if __name__ == "__main__":
    main()
