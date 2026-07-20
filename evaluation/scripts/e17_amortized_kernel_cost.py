# evaluation/scripts/e17_amortized_kernel_cost.py
# Amortized end-to-end cost analysis: comparing SA-FFN with partial columns + missed transfers
# vs. standard full-expert Dense FFN + full expert weight transfers.
import os
import sys
import json
import math
import torch
import torch.nn.functional as F

sys.path.append("/home/palakm/MoEServingSim/vllm_integration")
from sa_ffn_triton import sa_ffn_forward

MODELS = {
    "qwen3_30b": {
        "hidden_size": 2048, "intermediate": 768,
        "cache_size": 128, "miss_size": 16,
    },
    "deepseek_v2_lite": {
        "hidden_size": 2048, "intermediate": 1408,
        "cache_size": 256, "miss_size": 32,
    }
}

def timed(fn, device, iters=1000):
    torch.cuda.synchronize(device)
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    for _ in range(50):
        fn()
    torch.cuda.synchronize(device)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize(device)
    return s.elapsed_time(e) / iters

def benchmark_amortized(model_name: str, spec: dict, device: str = "cuda:0"):
    print(f"\nBenchmarking Amortized Kernel Cost for {model_name}...")
    H = spec["hidden_size"]
    I = spec["intermediate"]
    C = spec["cache_size"]
    M = spec["miss_size"]
    
    batches = [1, 4, 8, 16]
    link_speeds_gbs = [16.0, 64.0] # Gen4 (16 GB/s), Gen5 (64 GB/s)
    
    # Pre-allocate weights on GPU
    # Cached portion
    W_gate_c = torch.randn(C, H, dtype=torch.bfloat16, device=device)
    W_up_c = torch.randn(C, H, dtype=torch.bfloat16, device=device)
    W_down_c = torch.randn(H, C, dtype=torch.bfloat16, device=device)
    
    # Missed/streamed portion
    W_gate_m = torch.randn(M, H, dtype=torch.bfloat16, device=device)
    W_up_m = torch.randn(M, H, dtype=torch.bfloat16, device=device)
    W_down_m = torch.randn(H, M, dtype=torch.bfloat16, device=device)
    
    # Dense FFN (loads and computes on the FULL expert size I)
    W_gate_full = torch.randn(I, H, dtype=torch.bfloat16, device=device)
    W_up_full = torch.randn(I, H, dtype=torch.bfloat16, device=device)
    W_down_full = torch.randn(H, I, dtype=torch.bfloat16, device=device)
    
    results = {}
    
    for B in batches:
        x = torch.randn(B, H, dtype=torch.bfloat16, device=device)
        
        # 1. Measure FFN execution time (SA-FFN vs. Full Dense FFN)
        def run_full_dense():
            g = torch.matmul(x, W_gate_full.t())
            u = torch.matmul(x, W_up_full.t())
            return torch.matmul(F.silu(g) * u, W_down_full.t())
            
        def run_sa_ffn():
            return sa_ffn_forward(
                x, W_gate_c, W_up_c, W_down_c,
                W_gate_m, W_up_m, W_down_m
            )
            
        t_dense_ms = timed(run_full_dense, device)
        t_sa_ms = timed(run_sa_ffn, device)
        
        results[B] = []
        
        for bw in link_speeds_gbs:
            # 2. Calculate weight transfer sizes (each parameter is BF16 -> 2 bytes)
            # Full expert transfer payload: gate, up, down matrices of size I x H
            full_expert_bytes = 3 * I * H * 2
            # Missed columns transfer payload: gate, up, down matrices of size M x H
            miss_column_bytes = 3 * M * H * 2
            
            # Achieve transfer latency (ms)
            t_transfer_full_ms = (full_expert_bytes / (bw * 1e9)) * 1000.0
            t_transfer_miss_ms = (miss_column_bytes / (bw * 1e9)) * 1000.0
            
            # Total system latencies
            t_total_dense_ms = t_dense_ms + t_transfer_full_ms
            t_total_sa_ms = t_sa_ms + t_transfer_miss_ms
            
            speedup = t_total_dense_ms / t_total_sa_ms
            
            # Record results
            results[B].append({
                "link_speed_gbps": bw,
                "dense_compute_ms": t_dense_ms,
                "sa_ffn_compute_ms": t_sa_ms,
                "full_transfer_ms": t_transfer_full_ms,
                "miss_transfer_ms": t_transfer_miss_ms,
                "total_dense_system_ms": t_total_dense_ms,
                "total_sa_system_ms": t_total_sa_ms,
                "net_speedup": speedup
            })
            
            print(f"  Batch: {B:<2} | Link: {bw} GB/s | Dense Sys: {t_total_dense_ms:.4f} ms | SA-FFN Sys: {t_total_sa_ms:.4f} ms | Net Speedup: {speedup:.2f}x")
            
    out_dir = f"/home/palakm/MoEServingSim/evaluation/results/e17_amortized/{model_name}"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "amortized_kernel_results.json"), "w") as f:
        json.dump(results, f, indent=4)

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    for name, spec in MODELS.items():
        benchmark_amortized(name, spec, device)

if __name__ == "__main__":
    main()
