import torch
import torch.nn.functional as F
import time

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

def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
        
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    
    # Qwen3-30B-A3B dimensions
    H = 4096      # hidden size
    I = 768       # full FFN dimension (columns)
    C = 128       # cached columns
    M = 16        # missed columns
    
    # Token inputs (batch size B=1, sequence length = 1)
    # Note: during autoregressive decoding, batch size is 1
    x = torch.randn(1, H, dtype=torch.bfloat16, device=device)
    
    # Full Weights (768 columns)
    W_gate_full = torch.randn(I, H, dtype=torch.bfloat16, device=device)
    W_up_full = torch.randn(I, H, dtype=torch.bfloat16, device=device)
    W_down_full = torch.randn(H, I, dtype=torch.bfloat16, device=device)
    
    # Cached Weights (128 columns)
    W_gate_c = torch.randn(C, H, dtype=torch.bfloat16, device=device)
    W_up_c = torch.randn(C, H, dtype=torch.bfloat16, device=device)
    W_down_c = torch.randn(H, C, dtype=torch.bfloat16, device=device)
    
    # Missed Weights (16 columns)
    W_gate_m = torch.randn(M, H, dtype=torch.bfloat16, device=device)
    W_up_m = torch.randn(M, H, dtype=torch.bfloat16, device=device)
    W_down_m = torch.randn(H, M, dtype=torch.bfloat16, device=device)
    
    # Computations
    def run_full_dense_ffn():
        gate = torch.matmul(x, W_gate_full.t())
        up = torch.matmul(x, W_up_full.t())
        act = F.silu(gate) * up
        return torch.matmul(act, W_down_full.t())
        
    def run_sa_ffn():
        # Cached sub-FFN
        gate_c = torch.matmul(x, W_gate_c.t())
        up_c = torch.matmul(x, W_up_c.t())
        act_c = F.silu(gate_c) * up_c
        y_c = torch.matmul(act_c, W_down_c.t())
        
        # Missed sub-FFN
        gate_m = torch.matmul(x, W_gate_m.t())
        up_m = torch.matmul(x, W_up_m.t())
        act_m = F.silu(gate_m) * up_m
        y_m = torch.matmul(act_m, W_down_m.t())
        
        return y_c + y_m
        
    t_full = timed(run_full_dense_ffn, device) * 1000.0  # us
    t_sa = timed(run_sa_ffn, device) * 1000.0          # us
    
    print(f"Physical H100 NVL GEMV Benchmarks (Batch Size 1, BF16):")
    print(f"  Full Dense Expert (768 columns) : {t_full:.2f} µs")
    print(f"  Sparse-Aware FFN (144 columns)   : {t_sa:.2f} µs")
    print(f"  Net Speedup of SA-FFN vs Dense   : {t_full / t_sa:.2f}x")

if __name__ == '__main__':
    main()
