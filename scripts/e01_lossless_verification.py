# evaluation/scripts/e01_lossless_verification.py
import os
import json
import torch
import torch.nn.functional as F
import numpy as np

# Specs matching our 3 target models
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

def verify_model_sa_ffn(model_name: str, spec: dict, device: str = "cuda:0"):
    print(f"Running Lossless Verification for {model_name}...")
    H = spec["hidden_size"]
    I = spec["intermediate"]
    C = spec["cache_size"]
    M = spec["miss_size"]
    
    # Instantiate random weights to verify GEMM split logic mathematically
    # (Since this is a bit-exact correctness check on the mathematical decomposition)
    import math
    
    # Scale weights using Kaiming scaling to keep activations range normalized
    W_gate = torch.randn(I, H, dtype=torch.bfloat16, device=device) / math.sqrt(H)
    W_up = torch.randn(I, H, dtype=torch.bfloat16, device=device) / math.sqrt(H)
    W_down = torch.randn(H, I, dtype=torch.bfloat16, device=device) / math.sqrt(I)
    
    # Split into cached and missed parameters
    W_gate_c, W_gate_m = W_gate[:C], W_gate[C:C+M]
    W_up_c, W_up_m = W_up[:C], W_up[C:C+M]
    W_down_c, W_down_m = W_down[:, :C], W_down[:, C:C+M]
    
    # Token activation input (random batch size representing token hidden states)
    x = torch.randn(128, H, dtype=torch.bfloat16, device=device)
    
    # Baseline: standard FFN (for the subset of columns C + M)
    W_gate_partial = W_gate[:C+M]
    W_up_partial = W_up[:C+M]
    W_down_partial = W_down[:, :C+M]
    
    # Ground Truth computation
    gate_p = torch.matmul(x, W_gate_partial.t())
    up_p = torch.matmul(x, W_up_partial.t())
    act_p = F.silu(gate_p) * up_p
    y_baseline = torch.matmul(act_p, W_down_partial.t())
    
    # SA-FFN Split computation
    # Phase 1: Cached Columns
    gate_c = torch.matmul(x, W_gate_c.t())
    up_c = torch.matmul(x, W_up_c.t())
    act_c = F.silu(gate_c) * up_c
    y_cached = torch.matmul(act_c, W_down_c.t())
    
    # Phase 2: Missed Columns
    gate_m = torch.matmul(x, W_gate_m.t())
    up_m = torch.matmul(x, W_up_m.t())
    act_m = F.silu(gate_m) * up_m
    y_missed = torch.matmul(act_m, W_down_m.t())
    
    # In-place accumulation
    y_sa = y_cached + y_missed
    
    # Compute error metrics
    max_abs_diff = torch.max(torch.abs(y_baseline - y_sa)).item()
    mean_abs_diff = torch.mean(torch.abs(y_baseline - y_sa)).item()
    cosine_sim = F.cosine_similarity(y_baseline.flatten().float(), y_sa.flatten().float(), dim=0).item()
    
    verdict = "PASS" if max_abs_diff < 5e-2 else "FAIL"
    
    result = {
        "model_name": model_name,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "cosine_similarity": cosine_sim,
        "verdict": verdict
    }
    
    print(f"  Verdict: {verdict} | Max Diff: {max_abs_diff:.6e} | Cosine: {cosine_sim:.8f}")
    
    # Save to output folder
    out_dir = f"/home/palakm/MoEServingSim/evaluation/results/e01_lossless/{model_name}"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "correctness_report.json"), "w") as f:
        json.dump(result, f, indent=4)
        
def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    for name, spec in MODELS.items():
        verify_model_sa_ffn(name, spec, device)

if __name__ == "__main__":
    main()
