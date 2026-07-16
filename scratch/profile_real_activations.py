import os
import json
import torch
import torch.nn as nn
import numpy as np

# =====================================================================
# 1. MOE INSTRUMENTATION HOOK (SwiGLU Post-Multiplication)
# =====================================================================
class SwiGLUPostMultHook:
    """
    Hook to capture actual neuron activations after element-wise SwiGLU multiplication.
    This registers the exact intermediate FFN channels contributing to downstream down_proj.
    """
    def __init__(self, layer_idx, expert_idx):
        self.layer_idx = layer_idx
        self.expert_idx = expert_idx
        self.records = []

    def hook_fn(self, module, input, output):
        # In SwiGLU: output is shape [batch, seq, intermediate_dim]
        # representing: SiLU(gate_proj(x)) * up_proj(x)
        y = output.detach().float()
        
        # Flatten sequence length to iterate per token
        flat_acts = y.view(-1, y.size(-1))  # [num_tokens, intermediate_dim]
        num_tokens = flat_acts.size(0)
        
        for t_pos in range(num_tokens):
            tok_act = flat_acts[t_pos]  # [intermediate_dim]
            abs_act = tok_act.abs()
            total_energy = abs_act.sum().item()
            
            if total_energy == 0:
                continue
                
            # A. Scientific Absolute Thresholds
            active_counts = {}
            for eps in [1e-5, 1e-4, 1e-3, 1e-2]:
                active_counts[f"active_count_eps_{eps}"] = torch.sum(abs_act > eps).item()
                
            # B. Energy Concentration (Top-k sorted energy)
            sorted_mags, sorted_indices = torch.sort(abs_act, descending=True)
            cumulative_energy = torch.cumsum(sorted_mags, dim=0) / total_energy
            
            energy_k = {}
            for eta in [0.95, 0.99, 0.999]:
                k_val = torch.where(cumulative_energy >= eta)[0]
                k_idx = k_val[0].item() + 1 if len(k_val) > 0 else len(cumulative_energy)
                energy_k[f"energy_k_{eta}"] = k_idx
                
            # Save token record
            self.records.append({
                "layer": self.layer_idx,
                "token_pos": t_pos,
                "expert_id": self.expert_idx,
                "active_counts": active_counts,
                "energy_k": energy_k,
                "active_indices_99": sorted_indices[:energy_k["energy_k_0.99"]].cpu().tolist()
            })

# =====================================================================
# 2. RUNNABLE VERIFICATION PIPELINE (Fallback for restricted environments)
# =====================================================================
class MockSwiGLUExpert(nn.Module):
    """
    Mock FFN layer representing an expert SwiGLU MLP projection.
    """
    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim)
        self.act_fn = nn.Identity() # Placeholder hook point
        
    def forward(self, x):
        g = torch.nn.functional.silu(self.gate_proj(x))
        u = self.up_proj(x)
        y = g * u
        # Run Identity act_fn to trigger the forward hook
        return self.act_fn(y)

def run_verification_pass():
    print("=====================================================================")
    print("RUNNING INSTRUMENTATION PASS (CPU Mock Mode for PyTorch Verification)")
    print("=====================================================================")
    
    hidden_dim = 1024
    intermediate_dim = 2048
    seq_len = 100
    
    # 1. Initialize Mock Expert
    expert = MockSwiGLUExpert(hidden_dim, intermediate_dim)
    
    # 2. Register Hook after SwiGLU multiplication
    hook = SwiGLUPostMultHook(layer_idx=10, expert_idx=0)
    expert.act_fn.register_forward_hook(hook.hook_fn)
    
    # 3. Generate representative semantically correlated hidden states
    # This simulates real transformer hidden states sequence drift
    inputs = torch.randn(1, seq_len, hidden_dim)
    for i in range(1, seq_len):
        inputs[0, i] = 0.95 * inputs[0, i-1] + 0.05 * inputs[0, i]
        
    # Run forward pass to trigger hooks
    with torch.no_grad():
        _ = expert(inputs)
        
    print(f"Captured {len(hook.records)} token activation records.")
    
    # 4. Extract Powers-of-Two Jaccard Overlaps
    distances = [1, 2, 4, 8, 16, 32, 64]
    jaccards_vs_dist = {}
    
    active_sets = [set(r["active_indices_99"]) for r in hook.records]
    
    for dist in distances:
        overlaps = []
        for i in range(len(active_sets) - dist):
            set_a = active_sets[i]
            set_b = active_sets[i + dist]
            union = len(set_a.union(set_b))
            inter = len(set_a.intersection(set_b))
            overlaps.append(inter / union if union > 0 else 0)
        jaccards_vs_dist[dist] = np.mean(overlaps) if overlaps else 0.0
        
    print("\n--- Jaccard Overlaps vs Distance (Powers of 2) ---")
    for d in distances:
        print(f"  Distance {d:2d}: {jaccards_vs_dist[d]:.4f}")
        
    # 5. Fit Eviction Time Constant tau
    # Fit: ln(J(d) - J_inf) = ln(J0 - J_inf) - d/tau
    j0 = jaccards_vs_dist[1]
    j_inf = 0.031 # Random overlap for pool size 2048 (64/2048)
    
    x_fit = []
    y_fit = []
    for d in [1, 2, 4, 8, 16]:
        j_d = jaccards_vs_dist[d]
        if j_d > j_inf:
            x_fit.append(d)
            y_fit.append(np.log(j_d - j_inf))
            
    fitted_tau = 0.0
    if len(x_fit) > 1:
        slope, intercept = np.polyfit(x_fit, y_fit, 1)
        fitted_tau = -1.0 / slope if slope != 0 else 0.0
        print(f"\nFitted Jaccard Eviction Decay Half-Life Time Constant (tau): {fitted_tau:.2f} tokens")
        
    # 6. Save output JSON
    output_path = "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/scratch/verification_pass_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results = {
        "jaccard_vs_distance": jaccards_vs_dist,
        "fitted_tau": float(fitted_tau),
        "mean_active_count_eps_1e-4": float(np.mean([r["active_counts"]["active_count_eps_0.0001"] for r in hook.records])),
        "mean_energy_k_99": float(np.mean([r["energy_k"]["energy_k_0.99"] for r in hook.records]))
    }
    with open(output_path, "w") as f:
        json.dump(results, f)
        
    print(f"\nResults successfully saved to: {output_path}")

if __name__ == "__main__":
    run_verification_pass()
