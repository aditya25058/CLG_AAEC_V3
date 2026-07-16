# vllm_integration/sa_ffn_triton.py
import torch
import torch.nn.functional as F

def sa_ffn_forward(
    x: torch.Tensor,          # Shape [M, K] (Token inputs)
    W_gate_c: torch.Tensor,   # Shape [DC, K] (Cached Gate weights)
    W_up_c: torch.Tensor,     # Shape [DC, K] (Cached Up weights)
    W_down_c: torch.Tensor,   # Shape [K, DC] (Cached Down weights)
    W_gate_m: torch.Tensor,   # Shape [MS, K] (Streamed Gate weights)
    W_up_m: torch.Tensor,     # Shape [MS, K] (Streamed Up weights)
    W_down_m: torch.Tensor    # Shape [K, MS] (Streamed Down weights)
) -> torch.Tensor:
    """
    Streaming Accumulation FFN (SA-FFN) forward pass.
    Computes:
      Y = [silu(X @ W_gate_c.T) * (X @ W_up_c.T)] @ W_down_c.T
        + [silu(X @ W_gate_m.T) * (X @ W_up_m.T)] @ W_down_m.T
    Using PyTorch's cuBLAS/CUTLASS-backed GEMM operations for optimal H100 Tensor Core performance.
    """
    # 1. ──── PHASE 1: Compute on Warm Cached Columns ────
    # Compute gate and up projections in parallel
    gate_c = torch.matmul(x, W_gate_c.t())
    up_c = torch.matmul(x, W_up_c.t())
    
    # Apply SiLU activation and down project
    act_c = F.silu(gate_c) * up_c
    y_cached = torch.matmul(act_c, W_down_c.t())

    # 2. ──── PHASE 2: Compute on Missed/Streamed Columns ────
    # Compute gate and up projections on the fetched misses
    gate_m = torch.matmul(x, W_gate_m.t())
    up_m = torch.matmul(x, W_up_m.t())
    
    # Apply SiLU activation and down project
    act_m = F.silu(gate_m) * up_m
    y_missed = torch.matmul(act_m, W_down_m.t())

    # 3. ──── Accumulate Outputs In-Place ────
    y_cached.add_(y_missed)
    
    return y_cached
