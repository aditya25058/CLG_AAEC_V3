import torch
import time
import numpy as np

def benchmark_fused_pipeline(device):
    print("\n=== Benchmarking Execution Pipeline Overhead (Single Token B=1) ===")
    
    hidden_dim = 4096
    ffn_dim = 768
    cache_size = 128
    miss_size = 32
    precision = torch.float16
    
    # B=1 (Single Token Decoding Phase)
    seq_len = 1
    
    # Tensors
    X = torch.randn(seq_len, hidden_dim, dtype=precision, device=device)
    W_cached = torch.randn(cache_size, hidden_dim, dtype=precision, device=device)
    W_missed = torch.randn(miss_size, hidden_dim, dtype=precision, device=device)
    W_full = torch.randn(cache_size + miss_size, hidden_dim, dtype=precision, device=device)
    
    # Warmup
    for _ in range(50):
        # Two-pass operations
        y1 = torch.matmul(X, W_cached.t())
        y2 = torch.matmul(X, W_missed.t())
        y_sum = y1 + y2
        # Single fused operations
        y_fused = torch.matmul(X, W_full.t())
        
    torch.cuda.synchronize(device)
    
    # 1. Benchmark Separate GEMM Launches (AAEC v2 Split Pipeline)
    iters = 1000
    start_time = time.time()
    for _ in range(iters):
        y1 = torch.matmul(X, W_cached.t())
        y2 = torch.matmul(X, W_missed.t())
        y_sum = y1 + y2
    torch.cuda.synchronize(device)
    split_lat = ((time.time() - start_time) / iters) * 1e6 # microseconds
    
    # 2. Benchmark Fused GEMM Launch (AAEC v3 Fused Pipeline)
    start_time = time.time()
    for _ in range(iters):
        y_fused = torch.matmul(X, W_full.t())
    torch.cuda.synchronize(device)
    fused_lat = ((time.time() - start_time) / iters) * 1e6 # microseconds
    
    print(f"Results for B=1, HiddenDim={hidden_dim}, ActiveColumns={cache_size+miss_size}:")
    print(f"  AAEC v2 Split Pipeline (Double Matmul + Add)  : {split_lat:.2f} us")
    print(f"  AAEC v3 Fused Pipeline (Single Matmul Launch) : {fused_lat:.2f} us")
    print(f"  Fused Execution Speedup                       : {split_lat / fused_lat:.2f}x")
    print(f"  Saved Microarchitectural Scheduler Latency    : {split_lat - fused_lat:.2f} us")
    
    # We will write the results to a file for documentation
    res_data = {
        "split_latency_us": split_lat,
        "fused_latency_us": fused_lat,
        "speedup": split_lat / fused_lat,
        "saved_latency_us": split_lat - fused_lat
    }
    
    import json
    with open("/home/palakm/MoEServingSim/qwen3_30b_plots/fused_execution_results.json", "w") as f:
        json.dump(res_data, f, indent=4)
        
    print("\nSaved benchmark results to qwen3_30b_plots/fused_execution_results.json")

def main():
    if not torch.cuda.is_available():
        print("CUDA not available. This script must run on GPU nodes.")
        return
        
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    
    print(f"Running Fused Pipeline overhead benchmark on GPU: {torch.cuda.get_device_name(device)}")
    benchmark_fused_pipeline(device)

if __name__ == "__main__":
    main()
