"""
run_split.py — Run split model across 2 machines using gRPC and collect metrics.

Usage (client/edge machine):
    python experiments/run_split.py --model gpt2 --split-layer 6 \
        --server-host 192.168.1.50 --server-port 50051

The server must already be running on machine B:
    python communication/server.py --model gpt2 --split-layer 6 --port 50051

Output:
    results/split_gpt2_layer6_client.json    ← client-side metrics
    results/split_gpt2_layer6_server.json    ← server metrics
    results/split_gpt2_layer6_comparison.json ← vs baseline
"""

import argparse
import json
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profiler.LayerProfiler import LayerProfiler
from profiler.DeviceProfiler import DeviceProfiler
from communication.client import SplitInferenceClient
from utils.Metrics import RunMetrics, get_peak_memory_mb, reset_peak_memory, timer
from utils.Config import InferenceConfig


SAMPLE_PROMPTS = {
    "gpt2": "The future of artificial intelligence is",
    "bert": "The capital of [MASK] is Paris.",
}


def load_head(model_name: str, split_layer: int, device: str,save_dir:str):
    if model_name == "bert": 
        from Models import load_bert_tokenizer, load_bert_head
        head = load_bert_head(save_dir, split_layer, device)  # already a BERTHead
        return head, load_bert_tokenizer()
    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_split(cfg: InferenceConfig) -> RunMetrics:
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  SPLIT RUN | model={cfg.model_name} | split_layer={cfg.split_layer}")
    print(f"  Client device={device} | Server={cfg.server_host}:{cfg.server_port}")
    print(f"{'='*60}")

    head, tokenizer = load_head(cfg.model_name, cfg.split_layer, device,save_dir="assets/BERT")
    head.eval()

    prompt = SAMPLE_PROMPTS.get(cfg.model_name, "Hello world")
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # ── gRPC Client ───────────────────────────────────────────────────────────
    grpc_client = SplitInferenceClient(
        cfg.server_host, cfg.server_port, timeout=cfg.socket_timeout
    )
    # Verify server is up before spending time loading models
    if not grpc_client.health_check():
        print("[Split] ERROR: Server unreachable. Start server.py on machine B first.")
        return None

    dev_profiler = DeviceProfiler(device)
    dev_profiler.print_snapshot()
    layer_profiler = LayerProfiler(head, device)

    client_metrics = RunMetrics(
        run_type="split_client",
        model_name=cfg.model_name,
        split_layer=cfg.split_layer,
        device=device,
    )

    def run_head(profile: bool = False) -> torch.Tensor:
        with torch.no_grad():
            if profile:
                with layer_profiler.record():
                    return head(input_ids) if cfg.model_name == "gpt2" \
                        else head(input_ids, attention_mask)
            return head(input_ids) if cfg.model_name == "gpt2" \
                else head(input_ids, attention_mask)

    # ── Warmup ────────────────────────────────────────────────────────────────
    print(f"\n[Split] Warming up ({cfg.warmup_runs} runs)...")
    for _ in range(cfg.warmup_runs):
        hidden = run_head()
        grpc_client.run_tail(
            hidden, cfg.model_name, cfg.split_layer, attention_mask
        )

    # ── Timed Runs ────────────────────────────────────────────────────────────
    print(f"[Split] Measuring ({cfg.timed_runs} runs)...")
    latencies_e2e = []
    comm_latencies = []
    tensor_size_mb = 0.0
    server_metrics_last = {}
    reset_peak_memory(device)

    if cfg.profile_layers:
        layer_profiler.attach()

    for i in range(cfg.timed_runs):
        with timer("e2e") as t_e2e:
            hidden = run_head(profile=(cfg.profile_layers and i == 0))
            result = grpc_client.run_tail(
                hidden, cfg.model_name, cfg.split_layer, attention_mask
            )

        latencies_e2e.append(t_e2e["elapsed_ms"])
        comm_latencies.append(result.comm_latency_ms)
        tensor_size_mb = result.tensor_size_mb
        server_metrics_last = result.server_metrics

    if cfg.profile_layers:
        layer_profiler.detach()

    # ── TTFT (GPT-2 only — uses streaming RPC) ────────────────────────────────
    ttft_ms = 0.0
    if cfg.model_name == "gpt2":
        print("[Split] Measuring TTFT via streaming RPC...")
        hidden = run_head()
        for token in grpc_client.run_tail_streaming(
            hidden, cfg.model_name, cfg.split_layer, max_new_tokens=1
        ):
            if token.is_first_token and token.ttft_ms is not None:
                ttft_ms = token.ttft_ms
                print(f"[Split] TTFT: {ttft_ms:.1f} ms | first token: '{token.text}'")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    layer_records = layer_profiler.get_metrics()
    total_client_flops = sum(lm.flops for lm in layer_records)
    peak_mem = get_peak_memory_mb(device)

    client_metrics.e2e_latency_ms = sum(latencies_e2e) / len(latencies_e2e)
    client_metrics.ttft_ms = ttft_ms
    client_metrics.total_flops = total_client_flops
    client_metrics.peak_mem_mb = peak_mem
    client_metrics.comm_latency_ms = sum(comm_latencies) / len(comm_latencies)
    client_metrics.tensor_size_bytes = int(tensor_size_mb * 1e6)
    client_metrics.layer_metrics = layer_records

    print(client_metrics.summary())
    print(f"\n[Split] Server-side summary:")
    print(f"  Server Latency : {server_metrics_last.get('e2e_latency_ms', 0):.1f} ms")
    print(f"  Server FLOPs   : {server_metrics_last.get('total_flops', 0)/1e9:.2f} GFLOPs")
    print(f"  Server Peak Mem: {server_metrics_last.get('peak_mem_mb', 0):.1f} MB")
    print(f"  Power State    : {server_metrics_last.get('power_state', 'unknown')}")

    if cfg.profile_layers:
        layer_profiler.print_layer_table(top_n=10)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(cfg.output_dir, exist_ok=True)
    tag = f"split_{cfg.model_name}_layer{cfg.split_layer}"

    client_metrics.save(os.path.join(cfg.output_dir, f"{tag}_client.json"))

    server_save = os.path.join(cfg.output_dir, f"{tag}_server.json")
    with open(server_save, "w") as f:
        json.dump(server_metrics_last, f, indent=2)
    print(f"[Metrics] Server metrics saved → {server_save}")

    # ── Comparison vs baseline ────────────────────────────────────────────────
    baseline_path = os.path.join(cfg.output_dir, f"baseline_{cfg.model_name}.json")
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline_data = json.load(f)

        baseline_lat   = baseline_data.get("e2e_latency_ms", 0)
        baseline_flops = baseline_data.get("total_flops", 0)
        server_flops   = server_metrics_last.get("total_flops", 0)
        combined_lat   = client_metrics.e2e_latency_ms

        comparison = {
            "model":                      cfg.model_name,
            "split_layer":                cfg.split_layer,
            "baseline_e2e_ms":            baseline_lat,
            "split_e2e_ms":               combined_lat,
            "latency_reduction_pct":      (baseline_lat - combined_lat) / baseline_lat * 100
                                          if baseline_lat > 0 else None,
            "server_flop_reduction_pct":  (baseline_flops - server_flops) / baseline_flops * 100
                                          if baseline_flops > 0 else None,
            "comm_overhead_ms":           client_metrics.comm_latency_ms,
            "tensor_payload_mb":          tensor_size_mb,
            "baseline_peak_mem_mb":       baseline_data.get("peak_mem_mb", 0),
            "client_peak_mem_mb":         client_metrics.peak_mem_mb,
            "server_peak_mem_mb":         server_metrics_last.get("peak_mem_mb", 0),
        }

        cmp_path = os.path.join(cfg.output_dir, f"{tag}_comparison.json")
        with open(cmp_path, "w") as f:
            json.dump(comparison, f, indent=2)

        print(f"\n[Comparison] vs Baseline:")
        for k, v in comparison.items():
            print(f"  {k:<38} {f'{v:.2f}' if isinstance(v, float) else v}")

    grpc_client.close()
    return client_metrics


def main():
    parser = argparse.ArgumentParser(description="gRPC split inference client (head side)")
    parser.add_argument("--model",        default="bert", choices=["gpt2", "bert"])
    parser.add_argument("--split-layer",  type=int, default=6)
    parser.add_argument("--server-host",  default="192.168.31.xx")
    parser.add_argument("--server-port",  type=int, default=50051)
    parser.add_argument("--device",       default="cuda")
    parser.add_argument("--runs",         type=int, default=5)
    parser.add_argument("--warmup",       type=int, default=2)
    parser.add_argument("--output-dir",   default="./results")
    args = parser.parse_args()

    cfg = InferenceConfig(
        model_name=args.model,
        split_layer=args.split_layer,
        server_host=args.server_host,
        server_port=args.server_port,
        device=args.device,
        timed_runs=args.runs,
        warmup_runs=args.warmup,
        output_dir=args.output_dir,
    )
    run_split(cfg)


if __name__ == "__main__":
    main()