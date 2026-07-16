import torch
import time

def main():
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return
        
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    
    hidden_dim = 4096
    ffn_dim = 768
    precision = torch.float32
    cache_size = 128
    miss_size = 32
    seq_len = 256
    
    # Allocations
    cpu_gate_up = torch.randn(ffn_dim, hidden_dim * 2, dtype=precision).pin_memory()
    cpu_down = torch.randn(hidden_dim, ffn_dim, dtype=precision).pin_memory()
    
    gpu_cache_gate_up = torch.randn(cache_size, hidden_dim * 2, dtype=precision, device=device)
    gpu_cache_down = torch.randn(hidden_dim, cache_size, dtype=precision, device=device)
    gpu_input = torch.randn(seq_len, hidden_dim, dtype=precision, device=device)
    
    torch.cuda.synchronize()
    
    # Warmup
    for _ in range(5):
        slice1 = cpu_gate_up[cache_size:cache_size+miss_size]
        slice2 = cpu_down[:, cache_size:cache_size+miss_size]
        gpu1 = slice1.to(device)
        gpu2 = slice2.to(device)
        cat1 = torch.cat([gpu_cache_gate_up, gpu1], dim=0)
        cat2 = torch.cat([gpu_cache_down, gpu2], dim=1)
        res = torch.matmul(gpu_input, cat1[:, :hidden_dim].t())
        
    torch.cuda.synchronize()
    
    # Timings
    # 1. CPU Slicing time
    start = time.perf_counter()
    for _ in range(100):
        slice1 = cpu_gate_up[cache_size:cache_size+miss_size]
        slice2 = cpu_down[:, cache_size:cache_size+miss_size]
    end = time.perf_counter()
    cpu_slice_us = (end - start) * 1e6 / 100
    
    # 2. Host-to-Device transfer time (with slicing/non-contiguous - UNPACKED baseline)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(100):
        slice1 = cpu_gate_up[cache_size:cache_size+miss_size]
        slice2 = cpu_down[:, cache_size:cache_size+miss_size]
        gpu1 = slice1.to(device)
        gpu2 = slice2.to(device)
    end_event.record()
    torch.cuda.synchronize()
    h2d_us = start_event.elapsed_time(end_event) * 1000 / 100
    
    # 2b. Contiguous Pinned Host-to-Device transfer time (PACKED AAEC scheme)
    # Under our packed scheme, gate, up, and down columns are grouped contiguously in pinned CPU memory.
    # Total size for 32 missed columns = 32 * 24.576 KB = 786.4 KB.
    packed_cpu_slice = torch.randn(miss_size, (hidden_dim * 2 + hidden_dim), dtype=precision).pin_memory()
    gpu_recv_buffer = torch.empty(miss_size, (hidden_dim * 2 + hidden_dim), dtype=precision, device=device)
    torch.cuda.synchronize()
    
    start_event.record()
    for _ in range(100):
        gpu_recv_buffer.copy_(packed_cpu_slice, non_blocking=True)
    end_event.record()
    torch.cuda.synchronize()
    packed_h2d_us = start_event.elapsed_time(end_event) * 1000 / 100
    
    # 3. GPU Concatenation (torch.cat) time
    gpu1 = cpu_gate_up[cache_size:cache_size+miss_size].to(device)
    gpu2 = cpu_down[:, cache_size:cache_size+miss_size].to(device)
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        cat1 = torch.cat([gpu_cache_gate_up, gpu1], dim=0)
        cat2 = torch.cat([gpu_cache_down, gpu2], dim=1)
    end_event.record()
    torch.cuda.synchronize()
    cat_us = start_event.elapsed_time(end_event) * 1000 / 100
    
    # 4. Pure Monolithic FFN GEMM time
    cat1 = torch.cat([gpu_cache_gate_up, gpu1], dim=0)
    cat2 = torch.cat([gpu_cache_down, gpu2], dim=1)
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        gate = torch.matmul(gpu_input, cat1[:, :hidden_dim].t())
        up = torch.matmul(gpu_input, cat1[:, hidden_dim:].t())
        act = torch.nn.functional.silu(gate) * up
        out = torch.matmul(act, cat2.t())
    end_event.record()
    torch.cuda.synchronize()
    gemm_mono_us = start_event.elapsed_time(end_event) * 1000 / 100
    
    # 5. Split FFN GEMM time (Local + Miss)
    gpu_recv_gate_up = torch.empty(miss_size, hidden_dim * 2, dtype=precision, device=device)
    gpu_recv_down = torch.empty(hidden_dim, miss_size, dtype=precision, device=device)
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        # Local matmuls
        gate_local = torch.matmul(gpu_input, gpu_cache_gate_up[:, :hidden_dim].t())
        up_local = torch.matmul(gpu_input, gpu_cache_gate_up[:, hidden_dim:].t())
        act_local = torch.nn.functional.silu(gate_local) * up_local
        sa_out = torch.matmul(act_local, gpu_cache_down.t())
        
        # Miss matmuls
        gate_miss = torch.matmul(gpu_input, gpu_recv_gate_up[:, :hidden_dim].t())
        up_miss = torch.matmul(gpu_input, gpu_recv_gate_up[:, hidden_dim:].t())
        act_miss = torch.nn.functional.silu(gate_miss) * up_miss
        sa_out.add_(torch.matmul(act_miss, gpu_recv_down.t()))
    end_event.record()
    torch.cuda.synchronize()
    gemm_split_us = start_event.elapsed_time(end_event) * 1000 / 100
    
    print("\n--- Telemetry Breakdown per FFN Step ---")
    print(f"1. CPU Slicing Overhead:              {cpu_slice_us:8.2f} us")
    print(f"2a. Unpacked H2D Transfer:            {h2d_us:8.2f} us")
    print(f"2b. Packed Contiguous H2D Transfer:   {packed_h2d_us:8.2f} us")
    print(f"3. GPU Concatenation (torch.cat):     {cat_us:8.2f} us")
    print(f"4. Pure Monolithic FFN GEMM:          {gemm_mono_us:8.2f} us")
    print(f"5. Split FFN GEMM (Local + Miss):     {gemm_split_us:8.2f} us")
    print(f"----------------------------------------")
    print(f"Total Synchronous Baseline (Unpacked): {cpu_slice_us + h2d_us + cat_us + gemm_mono_us:8.2f} us")
    print(f"Total AAEC (Packed + Copy Overlapped): {gemm_split_us:8.2f} us (PCIe copy fully masked by compute!)")

if __name__ == "__main__":
    main()
