# vllm_integration/benchmark_vllm_moe.py
import torch
import os
import time

# Prepend the necessary library path for CUDA runtime 13 compatibility
os.environ["LD_LIBRARY_PATH"] = "/usr/local/lib/ollama/cuda_v13:" + os.environ.get("LD_LIBRARY_PATH", "")

from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

def run_benchmark(batch_size: int, hidden_size: int, intermediate_size: int, num_experts: int, top_k: int):
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    
    # Form token input tensors
    hidden_states = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=device)
    
    # Form expert weight parameters
    # w1 is packed: [num_experts, 2 * intermediate_size, hidden_size] (gate + up projection weights)
    w1 = torch.randn(num_experts, 2 * intermediate_size, hidden_size, dtype=torch.bfloat16, device=device)
    # w2 is [num_experts, hidden_size, intermediate_size] (down projection weights)
    w2 = torch.randn(num_experts, hidden_size, intermediate_size, dtype=torch.bfloat16, device=device)
    
    # Form routing scores with unique expert IDs per token
    topk_weights = torch.rand(batch_size, top_k, dtype=torch.bfloat16, device=device)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    topk_ids = torch.topk(torch.rand(batch_size, num_experts, device=device), top_k, dim=-1).indices.to(torch.int32)
    
    # ─── BENCHMARK 1: Standard vLLM Baseline ───
    os.environ["ENABLE_COLOSSUS"] = "0"
    
    # Warmup
    for _ in range(5):
        _ = fused_experts(hidden_states, w1, w2, topk_weights, topk_ids)
    torch.cuda.synchronize()
    
    # Execution
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(20):
        _ = fused_experts(hidden_states, w1, w2, topk_weights, topk_ids)
    end_event.record()
    torch.cuda.synchronize()
    
    baseline_time = start_event.elapsed_time(end_event) / 20.0
    
    # ─── BENCHMARK 2: COLOSSUS Pipelined Offloading ───
    os.environ["ENABLE_COLOSSUS"] = "1"
    os.environ["COLOSSUS_NUM_LAYERS"] = "1"
    os.environ["COLOSSUS_CACHE_SIZE"] = "512" # Cache 512 out of 768 columns
    
    # Warmup
    for _ in range(5):
        _ = fused_experts(hidden_states, w1, w2, topk_weights, topk_ids)
    torch.cuda.synchronize()
    
    # Execution
    start_event.record()
    for _ in range(20):
        _ = fused_experts(hidden_states, w1, w2, topk_weights, topk_ids)
    end_event.record()
    torch.cuda.synchronize()
    
    colossus_time = start_event.elapsed_time(end_event) / 20.0
    
    speedup = baseline_time / colossus_time
    
    print(f"| {batch_size:<10} | {baseline_time:12.3f} | {colossus_time:8.3f} | {speedup:7.2f}x |")

def main():
    print("==========================================================================")
    print("Benchmarking Physical vLLM MoE Layer vs. COLOSSUS Caching Policy")
    print("==========================================================================")
    print("Model Parameters: Qwen3-30B-A3B (128 Experts, Top-8 Routing)")
    print(f"  - Hidden Size: 2048, Intermediate Size per Expert: 768")
    print(f"  - COLOSSUS Cache Size: 512 columns (66.7% cached in VRAM)")
    print(f"  - COLOSSUS Miss Size: 16 columns dynamically fetched over PCIe")
    print("==========================================================================")
    print(f"| {'Tokens (M)':<10} | {'vLLM (ms)':<12} | {'COLOSSUS (ms)':<8} | {'Speedup':<8} |")
    print("--------------------------------------------------------------------------")
    
    # Sweep batch sizes (representing token batch sizes inside vLLM decode/prefill loops)
    for batch_size in [128, 512, 1024, 2048]:
        run_benchmark(
            batch_size=batch_size,
            hidden_size=2048,
            intermediate_size=768,
            num_experts=128,
            top_k=8
        )
    print("==========================================================================")

if __name__ == "__main__":
    main()
