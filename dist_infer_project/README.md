# Distributed transformer inference over gRPC

This repo is a clean baseline for **layer-partitioned inference** across multiple machines on the same network.

## What it does

- Splits a decoder-only Hugging Face CausalLM into **contiguous layer partitions**.
- Saves each partition as its own `weights.safetensors` file.
- Runs one **partition worker** per machine.
- Uses a **coordinator** to pass hidden states sequentially over **gRPC** from one machine to the next.
- Lets each machine load **only the layers assigned to it** at runtime.

## Important scope decision

This is a **correctness-first baseline**, not the final high-throughput version.

For simplicity and portability, the current decode loop recomputes the whole current sequence at each generation step, instead of keeping per-partition KV cache state. That choice keeps the code standardized across model families, while still giving you a working distributed inference skeleton. Hugging Face documents that cache formats differ across models, so keeping KV-cache logic isolated behind a future extension point is the safer base design. citeturn150824search1turn150824search3turn150824search17

## Why this layout is reasonable

gRPC is designed for efficient service-to-service RPC and uses protocol buffers for service definitions and message interchange. Python support and code generation are part of the official toolchain via `grpcio-tools`. citeturn150824search4turn150824search2turn150824search18

## Supported model family shape

This scaffold currently targets decoder-only Hugging Face models that expose:

- `model.embed_tokens`
- `model.layers`
- `model.norm`
- `lm_head`

That generally covers Llama/Mistral/Qwen2/Qwen3/Gemma-style CausalLM wrappers. It is **not** yet universal for every transformer architecture.

## Repo layout

```text
proto/inference.proto                gRPC service definition
distributed_inference/export_partitions.py
                                     one-time exporter from full HF model to partition files
distributed_inference/server.py      worker process for one partition
distributed_inference/coordinator.py coordinator that runs end-to-end generation
distributed_inference/runtime_partition.py
                                     runtime-only partition loader and forward executor
examples/cluster.yaml                sample cluster config
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/generate_proto.sh
```

## Step 1: export runtime partitions

Run this once on a machine that can access the full model weights:

```bash
python -m distributed_inference.export_partitions \
  --model-id Qwen/Qwen2.5-3B-Instruct \
  --output-dir /models/qwen2_5_3b \
  --layer-count 4 16 16 \
  --num-workers 3 \
  --dtype float16
```

This creates:

```text
/models/qwen2_5_3b/
  tokenizer/
  plan.json
  partition_0/
    meta.json
    weights.safetensors
  partition_1/
    meta.json
    weights.safetensors
  partition_2/
    meta.json
    weights.safetensors
```

Copy each `partition_i` directory to the corresponding machine.

## Step 2: start workers on remote machines

On node 1:

```bash
python -m distributed_inference.server \
  --partition-dir /models/qwen2_5_3b/partition_1 \
  --port 50051 \
  --device cuda
```

On node 2:

```bash
python -m distributed_inference.server \
  --partition-dir /models/qwen2_5_3b/partition_2 \
  --port 50051 \
  --device cuda
```

## Step 3: run the coordinator on the first node

The coordinator locally hosts partition 0 and calls the other partitions over gRPC.

```bash
python -m distributed_inference.coordinator \
  --config examples/cluster.yaml \
  --local-first-partition-dir /models/qwen2_5_3b/partition_0 \
  --prompt "Explain why attention masks are needed in causal language models." \
  --device cuda
```

## Design notes

### 1. Why export partitions first?

At runtime, you explicitly wanted each machine to load **only its own assigned layers**. The cleanest way to enforce that is to pre-export layer subsets into per-machine safetensor files, then load only those files on each worker.

### 2. Why contiguous partitions?

For a first implementation, contiguous layer blocks are simpler to reason about, easier to debug, and align with a forward-pass pipeline.

### 3. Why not KV cache yet?

Because cache plumbing is one of the first places cross-model standardization breaks. Hugging Face’s docs explicitly note that cache formats vary by model family. citeturn150824search1turn150824search3turn150824search17

### 4. Why gRPC?

Official gRPC documentation positions it as a high-performance RPC system for service connectivity, with built-in support for load balancing, health checking, tracing, and authentication. That makes it a good fit for machine-to-machine forwarding of hidden states. citeturn150824search10turn150824search12turn150824search16

## What you should build next

1. **Per-partition KV cache**
   - Keep cache state on each worker keyed by `request_id`.
   - Send only the latest token on decode after prefill.

2. **Binary framing optimization**
   - Consider FP16 transfer payloads and optional tensor compression.

3. **Failure handling**
   - Add request timeouts, retries, worker heartbeats, and resume logic.

4. **Batching**
   - Extend the coordinator to group requests with the same decode step.

5. **Topology-aware placement**
   - Put attention-heavy later layers on stronger GPUs if machines are heterogeneous.

## Limitations you should be aware of

- This is a **baseline**, so end-to-end latency can still be worse than a strong single-GPU setup if the network is slow.
- Partition export currently uses a full-model load one time during preprocessing.
- The runtime path currently focuses on **decoder-only text generation**.
- Attention-mask handling is written for a standardized baseline, but some model families may require extra position handling or rotary-embedding details for maximum compatibility.

## Recommended next refactor

The next serious step is to add a `CacheManager` interface and make `RuntimePartition.forward_partition()` able to switch between:

- `prefill_full_sequence()`
- `decode_one_token_with_cache()`

That keeps your API stable while moving the heavy optimization behind the partition abstraction.
