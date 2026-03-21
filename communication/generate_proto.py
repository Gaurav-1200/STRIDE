#!/usr/bin/env python3
"""
generate_proto.py — Compile inference.proto into Python gRPC stubs.

Run this once before using the communication layer:
    python communication/proto/generate_proto.py

Outputs (in communication/proto/):
    inference_pb2.py       ← protobuf message classes
    inference_pb2_grpc.py  ← gRPC service stubs (server + client)
"""

import subprocess
import sys
import os

PROTO_DIR = os.path.dirname(os.path.abspath(__file__))
PROTO_FILE = os.path.join(PROTO_DIR, "inference.proto")

# Output goes into the same proto/ directory so imports are clean
OUT_DIR = PROTO_DIR


def main():
    print("[Proto] Compiling inference.proto ...")

    cmd = [
        sys.executable, "-m", "grpc_tools.protoc",
        f"--proto_path={PROTO_DIR}",
        f"--python_out={OUT_DIR}",
        f"--grpc_python_out={OUT_DIR}",
        PROTO_FILE,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("[Proto] ✗ Compilation failed:")
        print(result.stderr)
        sys.exit(1)

    # Fix relative import in generated grpc file (grpc_tools quirk)
    grpc_file = os.path.join(OUT_DIR, "inference_pb2_grpc.py")
    if os.path.exists(grpc_file):
        with open(grpc_file) as f:
            content = f.read()
        # grpc_tools generates: import inference_pb2 as inference__pb2
        # We need:              from . import inference_pb2 as inference__pb2
        fixed = content.replace(
            "import inference_pb2 as inference__pb2",
            "from . import inference_pb2 as inference__pb2",
        )
        with open(grpc_file, "w") as f:
            f.write(fixed)

    print("[Proto] ✓ Generated:")
    print(f"         {OUT_DIR}/inference_pb2.py")
    print(f"         {OUT_DIR}/inference_pb2_grpc.py")
    print("\nNext steps:")
    print("  Server: python communication/server.py --model gpt2 --split-layer 6")
    print("  Client: python experiments/run_split.py --model gpt2 --split-layer 6 \\")
    print("              --server-host <IP>")


if __name__ == "__main__":
    main()