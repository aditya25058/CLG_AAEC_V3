# evaluation/scripts/e07_kernel_benchmarks.py
import os
import sys
import json
import torch
import torch.nn.functional as F

# Add vllm_integration directory to Python path
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
    },
    "mixtral_8x7b": {
        "hidden_size": 4096, "intermediate": 14336,
        "cache_size": 2048, "miss_size": 256,
    }
}

def timed(fn, device, iters=1000):
    """Measure GPU execution time using CUDA events (returns ms)."""
    torch.cuda.synchronize(device)
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    # Warmup
    for _ in range(50):
        fn()
    torch.cuda.synchronize(device)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize(device)
    return s.elapsed_time(e) / iters

def benchmark_sa_ffn_kernel(model_name: str, spec: dict, device: str = "cuda:0"):
    print(f"Benchmarking SA-FFN kernel for {model_name}...")
    H = spec["hidden_size"]
    I = spec["intermediate"]
    C = spec["cache_size"]
    M = spec["miss_size"]
    
    # Hidden states token input (Batch size M = 128, hidden size H)
    x = torch.randn(128, H, dtype=torch.bfloat16, device=device)
    
    # Warm Cached Weights
    W_gate_c = torch.randn(C, H, dtype=torch.bfloat16, device=device)
    W_up_c = torch.randn(C, H, dtype=torch.bfloat16, device=device)
    W_down_c = torch.randn(H, C, dtype=torch.bfloat16, device=device)
    
    # Streamed Weights
    W_gate_m = torch.randn(M, H, dtype=torch.bfloat16, device=device)
    W_up_m = torch.randn(M, H, dtype=torch.bfloat16, device=device)
    W_down_m = torch.randn(H, M, dtype=torch.bfloat16, device=device)
    
    # Standard complete baseline weights for comparison
    W_gate_full = torch.randn(C+M, H, dtype=torch.bfloat16, device=device)
    W_up_full = torch.randn(C+M, H, dtype=torch.bfloat16, device=device)
    W_down_full = torch.randn(H, C+M, dtype=torch.bfloat16, device=device)
    
    def run_standard_ffn():
        gate = torch.matmul(x, W_gate_full.t())
        up = torch.matmul(x, W_up_full.t())
        act = F.silu(gate) * up
        return torch.matmul(act, W_down_full.t())
        
    def run_sa_ffn():
        return sa_ffn_forward(
            x,
            W_gate_c, W_up_c, W_down_c,
            W_gate_m, W_up_m, W_down_m
        )
        
    t_std = timed(run_standard_ffn, device) * 1000.0  # Convert to microseconds
    t_sa = timed(run_sa_ffn, device) * 1000.0
    
    overhead_pct = (t_sa / t_std - 1) * 100.0
    
    results = {
        "model_name": model_name,
        "t_standard_us": t_std,
        "t_sa_us": t_sa,
        "overhead_pct": overhead_pct
    }
    
    print(f"  Standard FFN: {t_std:.2f} us | SA-FFN: {t_sa:.2f} us | Overhead: {overhead_pct:.1f}%")
    
    out_dir = f"/home/palakm/MoEServingSim/evaluation/results/e07_kernel/{model_name}"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "kernel_benchmarks.json"), "w") as f:
        json.dump(results, f, indent=4)

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    for name, spec in MODELS.items():
        benchmark_sa_ffn_kernel(name, spec, device)

if __name__ == "__main__":
    main()
