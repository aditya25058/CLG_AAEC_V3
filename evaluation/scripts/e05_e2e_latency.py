# evaluation/scripts/e05_e2e_latency.py
# Rewritten: measure REAL latency breakdown via dual-stream timing
import os
import sys
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("/home/palakm/MoEServingSim/vllm_integration")
from sa_ffn_triton import sa_ffn_forward

MODELS = {
    "qwen3_30b": {
        "num_layers": 48, "num_experts": 128, "top_k": 8,
        "hidden_size": 2048, "intermediate": 768, "cache_size": 128, "miss_size": 16
    },
    "deepseek_v2_lite": {
        "num_layers": 26, "num_experts": 64, "top_k": 6,
        "hidden_size": 2048, "intermediate": 1408, "cache_size": 256, "miss_size": 32
    }
}

def timed(fn, device, warmup=50, iters=500):
    """Measure GPU execution time using CUDA events (returns ms)."""
    torch.cuda.synchronize(device)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize(device)
    return s.elapsed_time(e) / iters

def run_latency_benchmark(model_name: str, spec: dict, device: str = "cuda:0"):
    print(f"Benchmarking End-to-End Latency for {model_name}...")
    H = spec["hidden_size"]
    I = spec["intermediate"]
    C = spec["cache_size"]
    M = spec["miss_size"]

    # Weights on GPU (cached portion)
    W_gate_c = torch.randn(C, H, dtype=torch.bfloat16, device=device) / math.sqrt(H)
    W_up_c   = torch.randn(C, H, dtype=torch.bfloat16, device=device) / math.sqrt(H)
    W_down_c = torch.randn(H, C, dtype=torch.bfloat16, device=device) / math.sqrt(C)

    # Weights on CPU pinned memory (missed portion — simulates host DRAM)
    W_gate_m_cpu = torch.randn(M, H, dtype=torch.bfloat16).pin_memory()
    W_up_m_cpu   = torch.randn(M, H, dtype=torch.bfloat16).pin_memory()
    W_down_m_cpu = torch.randn(H, M, dtype=torch.bfloat16).pin_memory()

    # Pre-allocated GPU receive buffers for DMA
    W_gate_m_gpu = torch.empty(M, H, dtype=torch.bfloat16, device=device)
    W_up_m_gpu   = torch.empty(M, H, dtype=torch.bfloat16, device=device)
    W_down_m_gpu = torch.empty(H, M, dtype=torch.bfloat16, device=device)

    # DMA stream
    dma_stream = torch.cuda.Stream(device=device)

    # Token input (B=1 autoregressive)
    x = torch.randn(1, H, dtype=torch.bfloat16, device=device)

    # --- Measurement 1: HBM-only (cache-only, no PCIe transfer) ---
    def hbm_only_forward():
        g = torch.matmul(x, W_gate_c.t())
        u = torch.matmul(x, W_up_c.t())
        return torch.matmul(F.silu(g) * u, W_down_c.t())

    t_hbm_only = timed(hbm_only_forward, device)
    print(f"  HBM-only (Phase 1 only): {t_hbm_only:.4f} ms")

    # --- Measurement 2: Full SA-FFN with real PCIe transfer ---
    def full_sa_ffn_with_pcie():
        # Phase 1: compute cached on main stream
        g_c = torch.matmul(x, W_gate_c.t())
        u_c = torch.matmul(x, W_up_c.t())
        y_c = torch.matmul(F.silu(g_c) * u_c, W_down_c.t())

        # Concurrently: DMA missed columns from CPU to GPU
        compute_done = torch.cuda.Event()
        compute_done.record()
        with torch.cuda.stream(dma_stream):
            dma_stream.wait_event(compute_done)
            W_gate_m_gpu.copy_(W_gate_m_cpu, non_blocking=True)
            W_up_m_gpu.copy_(W_up_m_cpu, non_blocking=True)
            W_down_m_gpu.copy_(W_down_m_cpu, non_blocking=True)
            dma_done = torch.cuda.Event()
            dma_done.record()

        # Sync: wait for DMA completion before Phase 2
        torch.cuda.current_stream().wait_event(dma_done)

        # Phase 2: compute missed columns
        g_m = torch.matmul(x, W_gate_m_gpu.t())
        u_m = torch.matmul(x, W_up_m_gpu.t())
        y_m = torch.matmul(F.silu(g_m) * u_m, W_down_m_gpu.t())

        return y_c + y_m

    t_full = timed(full_sa_ffn_with_pcie, device)
    print(f"  Full SA-FFN + PCIe DMA: {t_full:.4f} ms")

    # --- Measurement 3: SA-FFN without PCIe (missed columns already on GPU) ---
    W_gate_m_local = W_gate_m_cpu.to(device)
    W_up_m_local = W_up_m_cpu.to(device)
    W_down_m_local = W_down_m_cpu.to(device)

    def full_sa_ffn_no_pcie():
        return sa_ffn_forward(
            x,
            W_gate_c, W_up_c, W_down_c,
            W_gate_m_local, W_up_m_local, W_down_m_local
        )

    t_no_pcie = timed(full_sa_ffn_no_pcie, device)
    print(f"  SA-FFN (all on GPU, no DMA): {t_no_pcie:.4f} ms")

    # --- Measurement 4: Raw PCIe transfer time only ---
    def pcie_only():
        with torch.cuda.stream(dma_stream):
            W_gate_m_gpu.copy_(W_gate_m_cpu, non_blocking=True)
            W_up_m_gpu.copy_(W_up_m_cpu, non_blocking=True)
            W_down_m_gpu.copy_(W_down_m_cpu, non_blocking=True)
        torch.cuda.current_stream().wait_stream(dma_stream)

    t_pcie_raw = timed(pcie_only, device)
    print(f"  Raw PCIe DMA transfer: {t_pcie_raw:.4f} ms")

    # Compute real exposed stall
    pcie_stall = max(0.0, t_full - t_no_pcie)
    overlap_ratio = 1.0 - (pcie_stall / max(t_pcie_raw, 1e-6))

    results = {
        "model_name": model_name,
        "method": "dual_stream_cuda_event_timing",
        "iterations": 500,
        "hbm_only_latency_ms": round(t_hbm_only, 4),
        "full_sa_ffn_with_pcie_ms": round(t_full, 4),
        "sa_ffn_no_pcie_ms": round(t_no_pcie, 4),
        "raw_pcie_transfer_ms": round(t_pcie_raw, 4),
        "exposed_pcie_stall_ms": round(pcie_stall, 4),
        "pcie_overlap_ratio": round(overlap_ratio, 4),
        "miss_columns": M,
        "transfer_size_bytes": M * 3 * spec["hidden_size"] * 2
    }

    print(f"  Exposed PCIe Stall: {pcie_stall:.4f} ms")
    print(f"  Overlap Ratio: {overlap_ratio*100:.1f}%")

    out_dir = f"/home/palakm/MoEServingSim/evaluation/results/e05_latency/{model_name}"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "latency_report.json"), "w") as f:
        json.dump(results, f, indent=4)

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    for name, spec in MODELS.items():
        run_latency_benchmark(name, spec, device)

if __name__ == "__main__":
    main()
