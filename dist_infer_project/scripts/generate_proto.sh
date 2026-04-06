#!/usr/bin/env bash
set -euo pipefail
python -m grpc_tools.protoc \
  -I ./proto \
  --python_out=./distributed_inference/rpc \
  --grpc_python_out=./distributed_inference/rpc \
  ./proto/inference.proto
