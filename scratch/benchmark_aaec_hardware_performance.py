import os
import json
import torch
import torch.nn as nn
import time
import numpy as np

def run_pcie_offload_benchmark(device):
    print("\n=== Scenario 1: PCIe Offloading Overlap Benchmark ===")
    
    # Dimensions for Qwen3-30B-A3B:
    # Hidden dimension = 4096, FFN dimension = 768.
    # Weight size per neuron = (3 * 4096) parameters = 12,288 parameters.
    # In BF16, this is 24,576 bytes = 24.576 KB per neuron column packet.
    # Baseline full expert weight size = 768 * 24.576 KB = 18.874 MB.
    
    hidden_dim = 4096
    ffn_dim = 768
    precision = torch.bfloat16
    
    # 1. Allocate full expert weights in PINNED CPU memory (crucial for fast DMA)
    print("Allocating full expert weights in pinned CPU host memory...")
    cpu_expert_gate_up = torch.randn(ffn_dim, hidden_dim * 2, dtype=precision).pin_memory()
    cpu_expert_down = torch.randn(hidden_dim, ffn_dim, dtype=precision).pin_memory()
    
    # 2. Allocate sub-expert cache on the GPU
    # Cache size = 128 neurons. Memory footprint = 128 * 24.576 KB = 3.14 MB.
    cache_size = 128
    print(f"Allocating sub-expert cache (size={cache_size}) on GPU {device}...")
    gpu_cache_gate_up = torch.randn(cache_size, hidden_dim * 2, dtype=precision, device=device)
    gpu_cache_down = torch.randn(hidden_dim, cache_size, dtype=precision, device=device)
    
    # 3. Allocations for attention layers (for computation overlap)
    # A standard self-attention layer GEMM on a batch of 8 tokens:
    seq_len = 8
    gpu_input = torch.randn(seq_len, hidden_dim, dtype=precision, device=device)
    gpu_attn_proj = torch.randn(hidden_dim, hidden_dim, dtype=precision, device=device)
    
    # Create independent streams
    stream_compute = torch.cuda.Stream(device=device)
    stream_dma = torch.cuda.Stream(device=device)
    
    # Synchronize CUDA device
    torch.cuda.synchronize(device)
    
    # Warmup runs
    for _ in range(5):
        # Attention compute representation
        res = torch.matmul(gpu_input, gpu_attn_proj)
        # Asynchronous Host-to-Device copy
        gpu_cache_gate_up[:10].copy_(cpu_expert_gate_up[:10], non_blocking=True)
    torch.cuda.synchronize(device)
    
    # --- Measure Baseline: Synchronous Full Expert Transfer ---
    # In traditional offloading, the entire expert weights (18.87 MB) are copied before execution.
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    # Synchronous full expert transfer simulation
    gpu_full_gate_up = cpu_expert_gate_up.to(device, non_blocking=False)
    gpu_full_down = cpu_expert_down.to(device, non_blocking=False)
    # Execute FFN on the full expert
    gate = torch.matmul(gpu_input, gpu_full_gate_up[:, :hidden_dim].t()) # (8, 768)
    up = torch.matmul(gpu_input, gpu_full_gate_up[:, hidden_dim:].t()) # (8, 768)
    act = torch.nn.functional.silu(gate) * up # (8, 768)
    out = torch.matmul(act, gpu_full_down.t()) # (8, 4096)
    end_event.record()
    torch.cuda.synchronize(device)
    baseline_lat_ms = start_event.elapsed_time(end_event)
    
    # --- Measure AAEC: Overlapped Column Fetching ---
    # We fetch only the missed columns (e.g. 10 columns = 245.8 KB) asynchronously
    # while the attention compute is running on stream_compute.
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    with torch.cuda.stream(stream_compute):
        # 1. Run Attention Layer compute (masks the copy time)
        gpu_attn_output = torch.matmul(gpu_input, gpu_attn_proj)
        
    with torch.cuda.stream(stream_dma):
        # 2. Asynchronously fetch only the missed 10 columns (245.8 KB) from CPU memory
        gpu_cache_gate_up[:10].copy_(cpu_expert_gate_up[:10], non_blocking=True)
        gpu_cache_down[:, :10].copy_(cpu_expert_down[:, :10], non_blocking=True)
        
    # Wait for both stream_compute and stream_dma to synchronize
    torch.cuda.Stream.wait_stream(stream_compute, stream_dma)
    
    with torch.cuda.stream(stream_compute):
        # 3. Execute the FFN layer on the warm cache
        gate = torch.matmul(gpu_attn_output, gpu_cache_gate_up[:, :hidden_dim].t()) # (8, 128)
        up = torch.matmul(gpu_attn_output, gpu_cache_gate_up[:, hidden_dim:].t()) # (8, 128)
        act = torch.nn.functional.silu(gate) * up # (8, 128)
        gpu_ffn_output = torch.matmul(act, gpu_cache_down.t()) # (8, 4096)
        
    end_event.record()
    torch.cuda.synchronize(device)
    aaec_lat_ms = start_event.elapsed_time(end_event)
    
    print(f"Results:")
    print(f"  Baseline (Sync Full Expert Transfer): {baseline_lat_ms*1000.0:8.2f} us")
    print(f"  AAEC (Overlapped Column Fetch):       {aaec_lat_ms*1000.0:8.2f} us")
    print(f"  Physical Hardware Latency Speedup:    {baseline_lat_ms / aaec_lat_ms:.2f}x")
    
    return baseline_lat_ms * 1000.0, aaec_lat_ms * 1000.0

def run_nvlink_p2p_benchmark():
    print("\n=== Scenario 2: NVLink Peer-to-Peer Copy Benchmark ===")
    
    if torch.cuda.device_count() < 2:
        print("P2P Benchmark requires at least 2 GPUs. Skipping.")
        return None
        
    device0 = torch.device('cuda:0')
    device1 = torch.device('cuda:1')
    
    # Check Peer-to-Peer access
    can_access = torch.cuda.can_device_access_peer(0, 1)
    print(f"Checking peer-to-peer access between GPU 0 and GPU 1: {can_access}")
    if not can_access:
        print("P2P access not supported between these GPUs. Skipping.")
        return None
        
    # Enable Peer access
    torch.cuda.set_device(0)
    # P2P enable helper (requires calling peer access in driver)
    # PyTorch enables this automatically when doing peer copies.
    
    hidden_dim = 4096
    ffn_dim = 768
    precision = torch.bfloat16
    
    # 1. Allocate "hot" columns on GPU 0
    print("Allocating hot columns on GPU 0...")
    gpu0_hot_gate_up = torch.randn(128, hidden_dim * 2, dtype=precision, device=device0)
    
    # 2. Allocate full/cold columns on GPU 1
    print("Allocating full columns on GPU 1...")
    gpu1_full_gate_up = torch.randn(ffn_dim, hidden_dim * 2, dtype=precision, device=device1)
    
    torch.cuda.synchronize()
    
    # Warmup P2P transfer
    for _ in range(5):
        # Copy 10 columns (245.8 KB) from GPU 1 to GPU 0
        gpu0_hot_gate_up[:10].copy_(gpu1_full_gate_up[:10], non_blocking=True)
    torch.cuda.synchronize()
    
    # --- Measure Baseline NVLink: Transfer Entire Expert (9.44 MB) ---
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Allocate destination on GPU 0
    gpu0_full_gate_up = torch.empty(ffn_dim, hidden_dim * 2, dtype=precision, device=device0)
    
    start_event.record()
    gpu0_full_gate_up.copy_(gpu1_full_gate_up, non_blocking=False)
    end_event.record()
    torch.cuda.synchronize()
    p2p_base_ms = start_event.elapsed_time(end_event)
    
    # --- Measure AAEC NVLink: Transfer Missed Columns Only (e.g. 10 columns = 245.8 KB) ---
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    gpu0_hot_gate_up[:10].copy_(gpu1_full_gate_up[:10], non_blocking=False)
    end_event.record()
    torch.cuda.synchronize()
    p2p_aaec_ms = start_event.elapsed_time(end_event)
    
    print(f"Results:")
    print(f"  NVLink Baseline (Transfer Full Expert): {p2p_base_ms*1000.0:8.2f} us")
    print(f"  NVLink AAEC (Transfer 10 Columns):     {p2p_aaec_ms*1000.0:8.2f} us")
    print(f"  NVLink Peer-to-Peer Transfer Speedup:  {p2p_base_ms / p2p_aaec_ms:.2f}x")
    
    return p2p_base_ms * 1000.0, p2p_aaec_ms * 1000.0

def main():
    if not torch.cuda.is_available():
        print("CUDA not available. This script must run on GPU nodes.")
        return
        
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    
    print(f"Running benchmarks on GPU: {torch.cuda.get_device_name(device)}")
    
    pcie_results = run_pcie_offload_benchmark(device)
    nvlink_results = run_nvlink_p2p_benchmark()
    
    # Save results to json for paper figures
    out_dir = "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019"
    os.makedirs(out_dir, exist_ok=True)
    
    results = {
        "pcie_baseline_us": pcie_results[0] if pcie_results else 0.0,
        "pcie_aaec_us": pcie_results[1] if pcie_results else 0.0,
        "nvlink_baseline_us": nvlink_results[0] if nvlink_results else 0.0,
        "nvlink_aaec_us": nvlink_results[1] if nvlink_results else 0.0
    }
    
    with open(os.path.join(out_dir, "real_hardware_benchmarks.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    # Generate Plot
    import matplotlib.pyplot as plt
    
    # Configure Matplotlib styles
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 200,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })
    
    categories = ['PCIe Offloading', 'NVLink P2P']
    baselines = [results["pcie_baseline_us"], results["nvlink_baseline_us"]]
    aaec_lats = [results["pcie_aaec_us"], results["nvlink_aaec_us"]]
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(6, 4.5))
    rects1 = ax.bar(x - width/2, baselines, width, label='Baseline (Monolithic Expert)', color='#9b2226')
    rects2 = ax.bar(x + width/2, aaec_lats, width, label='AAEC (Sub-Expert Caching)', color='#005f73')
    
    ax.set_ylabel('Execution / Transfer Latency (us)', fontweight='bold')
    ax.set_title('Real Hardware: H100 Physical Latency Comparison', fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontweight='bold')
    ax.legend()
    ax.set_yscale('log') # Log scale for contrast
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}us',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
                        
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "real_hardware_latencies.png"), dpi=200)
    plt.close()
    
    print("\nReal hardware benchmarks and latency comparison plot completed!")

if __name__ == "__main__":
    main()
