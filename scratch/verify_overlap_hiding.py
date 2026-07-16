import torch
import time

def main():
    if not torch.cuda.is_available():
        print("CUDA not available. This script must run on physical GPU hardware.")
        return
        
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    
    # -------------------------------------------------------------
    # 1. PARAMETERS FOR QWEN3 OVERLAP PROFILER
    # -------------------------------------------------------------
    hidden_size = 5120
    # Payload sizes:
    # 5.9 MB  (50% energy prefetch payload, ~192 columns)
    # 13.2 MB (70% energy prefetch payload under 53.39% cache hit, ~222 columns)
    # 28.4 MB (90% energy prefetch payload under 53.39% cache hit, ~414 columns)
    payloads_bytes = {
        "50% Energy (5.9 MB)": int(5.9 * 1024 * 1024),
        "70% Energy (13.2 MB)": int(13.2 * 1024 * 1024),
        "90% Energy (28.4 MB)": int(28.4 * 1024 * 1024)
    }
    
    # Create streams
    comm_stream = torch.cuda.Stream()
    default_stream = torch.cuda.current_stream()
    
    # Allocate dummy GPU memory for Attention variables
    seq_len = 1
    qkv_weight = torch.randn(3 * hidden_size, hidden_size, dtype=torch.bfloat16, device=device)
    out_weight = torch.randn(hidden_size, hidden_size, dtype=torch.bfloat16, device=device)
    x = torch.randn(seq_len, hidden_size, dtype=torch.bfloat16, device=device)
    
    # Target Attention compute window is 100 microseconds (0.10 ms)
    target_attn_us = 100.0
    
    # Calibrate loop count to hit exactly ~100 µs
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    for _ in range(100):
        qkv = torch.matmul(x, qkv_weight.t())
        out = torch.matmul(qkv[:, :hidden_size], out_weight.t())
    torch.cuda.synchronize()
    t_end = time.perf_counter()
    single_iter_us = ((t_end - t_start) * 1e6) / 100.0
    
    iters_needed = max(1, int(round(target_attn_us / single_iter_us)))
    
    def run_calibrated_attention():
        for _ in range(iters_needed):
            qkv = torch.matmul(x, qkv_weight.t())
            out = torch.matmul(qkv[:, :hidden_size], out_weight.t())
            
    # Warmup GPU streams and kernels
    cpu_dummy = torch.randn(1024, dtype=torch.bfloat16).pin_memory()
    gpu_dummy = torch.empty(1024, dtype=torch.bfloat16, device=device)
    torch.cuda.synchronize()
    for _ in range(10):
        run_calibrated_attention()
        gpu_dummy.copy_(cpu_dummy, non_blocking=True)
    torch.cuda.synchronize()
    
    # -------------------------------------------------------------
    # 2. MEASURE CALIBRATED ATTENTION COMPUTE ALONE (T_attn)
    # -------------------------------------------------------------
    t_attn_start = torch.cuda.Event(enable_timing=True)
    t_attn_end = torch.cuda.Event(enable_timing=True)
    
    t_attn_start.record()
    for _ in range(1000):
        run_calibrated_attention()
    t_attn_end.record()
    torch.cuda.synchronize()
    
    T_attn = t_attn_start.elapsed_time(t_attn_end) / 1000.0  # ms
    T_attn_us = T_attn * 1000.0
    
    print("=" * 80)
    print(" AAEC V3: REAL GPU HARDWARE OVERLAP & TIMELINE VERIFICATION")
    print(f" Physical GPU: {torch.cuda.get_device_name(device)}")
    print(f" Calibrated Attention Compute Window (T_attn): {T_attn_us:.2f} µs (loops = {iters_needed})")
    print("=" * 80)
    
    print(f"{'Payload':<25} | {'T_comm (µs)':<12} | {'T_overlap (µs)':<14} | {'Exposed Stall (µs)':<18} | {'% Hidden':<10}")
    print("-" * 85)
    
    # -------------------------------------------------------------
    # 3. SWEEP PAYLOADS AND MEASURE OVERLAP TIMELINES WITH EVENT SYNC
    # -------------------------------------------------------------
    for name, num_bytes in payloads_bytes.items():
        # Allocate pinned CPU source memory and GPU destination memory
        num_elements = num_bytes // 2  # BFloat16 (2 bytes per element)
        cpu_weights = torch.randn(num_elements, dtype=torch.bfloat16).pin_memory()
        gpu_weights = torch.empty(num_elements, dtype=torch.bfloat16, device=device)
        
        torch.cuda.synchronize()
        
        # Measure T_transfer alone on comm_stream
        t_comm_start = torch.cuda.Event(enable_timing=True)
        t_comm_end = torch.cuda.Event(enable_timing=True)
        
        t_comm_start.record(stream=comm_stream)
        for _ in range(500):
            with torch.cuda.stream(comm_stream):
                gpu_weights.copy_(cpu_weights, non_blocking=True)
        t_comm_end.record(stream=comm_stream)
        torch.cuda.synchronize()
        T_comm = t_comm_start.elapsed_time(t_comm_end) / 500.0  # ms
        
        # Measure T_overlap (Concurrent Execution with Stream Synchronization Dependency)
        t_overlap_start = torch.cuda.Event(enable_timing=True)
        t_overlap_end = torch.cuda.Event(enable_timing=True)
        
        # Create event to track when the copy finishes on comm_stream
        copy_finished_event = torch.cuda.Event()
        
        t_overlap_start.record(stream=default_stream)
        for _ in range(500):
            with torch.cuda.stream(comm_stream):
                # Queue transfer on background stream
                gpu_weights.copy_(cpu_weights, non_blocking=True)
                # Record event on comm_stream indicating transfer completion
                copy_finished_event.record(stream=comm_stream)
            
            # Queue compute concurrently on the default stream
            run_calibrated_attention()
            
            # Sync the default stream with the transfer completion event
            # (default stream must wait for copy_finished_event before it can continue)
            default_stream.wait_event(copy_finished_event)
            
        t_overlap_end.record(stream=default_stream)
        torch.cuda.synchronize()
        T_overlap = t_overlap_start.elapsed_time(t_overlap_end) / 500.0  # ms
        
        # Compute exposed stall and hiding percentage
        exposed_stall = max(0.0, T_overlap - T_attn)
        hidden_pct = max(0.0, min(100.0, (T_attn + T_comm - T_overlap) / T_comm * 100.0))
        
        # Convert to microseconds for output display
        t_comm_us = T_comm * 1000
        t_overlap_us = T_overlap * 1000
        exposed_stall_us = exposed_stall * 1000
        
        print(f"{name:<25} | {t_comm_us:<12.1f} | {t_overlap_us:<14.1f} | {exposed_stall_us:<18.1f} | {hidden_pct:<9.1f}%")
        
    print("=" * 80)
    print(" Verification complete.")

if __name__ == '__main__':
    main()
