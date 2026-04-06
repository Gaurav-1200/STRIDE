from __future__ import annotations

import argparse

from distributed_inference.partitioner import export_partitions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--layer-counts", type=int, nargs='+' , required=True)
    parser.add_argument("--dtype", default="float16")
    args = parser.parse_args()
    export_partitions(
        model_id=args.model_id,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        layer_counts=args.layer_counts,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
