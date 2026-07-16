import os
import json
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "Write a Python function that implements a binary search tree with insert, delete, and search operations.",
    "Solve the differential equation dy/dx = xy + x using the integrating factor method.",
    "Explain how a CPU cache hierarchy works, including L1, L2, and L3 caches. What is cache coherence?"
]

def main():
    print("==================================================================")
    print("Running Experiment 5: Activation-Weighted FFN Energy Concentration")
    print("Loading tokenizer and Qwen3 model on GPU...")
    print("==================================================================")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B", trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-30B-A3B",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    print("Model loaded successfully on GPU!\n")
    
    # Store activation curves globally
    all_activation_curves = []
    
    # Monkey-patch hooked forward to capture intermediate activations
    # We patch layers 0, 12, 24, 36, 47 to get representative samples
    target_layers = [0, 12, 24, 36, 47]
    
    # Define a storage hook class
    class ActivationCapture:
        def __init__(self):
            self.curves = []
            
        def hook(self, module, input, output):
            # Qwen3MoeExperts output has shape [seq_len, hidden_size] or similar
            # Since experts are dispatched, let's hook gate_up_proj execution
            pass

    # A simpler approach: we can monkey patch the experts' forward pass
    # to inspect the gate * up intermediate activations dynamically
    captured_curves = []

    def patch_experts(layer_idx):
        experts_module = model.model.layers[layer_idx].mlp.experts
        original_forward = experts_module.forward
        
        def hooked_forward(hidden_states, top_k_index, top_k_weights):
            # hidden_states: [seq_len, hidden_size]
            # We compute intermediate values directly to calculate energy concentration
            num_experts = experts_module.gate_up_proj.size(0)
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            
            for expert_idx_tensor in expert_hit:
                expert_idx = int(expert_idx_tensor[0].item())
                if expert_idx == num_experts:
                    continue
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[token_idx]
                
                with torch.no_grad():
                    gate, up = nn.functional.linear(
                        current_state, experts_module.gate_up_proj[expert_idx]
                    ).chunk(2, dim=-1)
                    gate = torch.nn.functional.silu(gate)
                    intermediate = gate * up # [num_tokens_sent_to_expert, 768]
                    
                    abs_vals = torch.abs(intermediate).float().cpu().numpy()
                    
                    for tok_idx in range(abs_vals.shape[0]):
                        sorted_vals = np.sort(abs_vals[tok_idx])[::-1]
                        total_energy = np.sum(sorted_vals)
                        if total_energy > 0:
                            cum_energy = np.cumsum(sorted_vals) / total_energy
                            captured_curves.append(cum_energy)
                            
            return original_forward(hidden_states, top_k_index, top_k_weights)
            
        experts_module.forward = hooked_forward

    print("Registering activation capture patches on target layers...")
    for idx in target_layers:
        patch_experts(idx)
        
    print("Running inference on sample prompts...")
    for idx, prompt in enumerate(PROMPTS):
        print(f" - Prompt {idx+1}/{len(PROMPTS)}: '{prompt[:45]}...'")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)
            
    print(f"\nCaptured a total of {len(captured_curves)} token-expert intermediate activation traces.")
    
    if len(captured_curves) == 0:
        print("Error: No activations captured! Falling back to database trace profile.")
        return
        
    # Average curves
    mean_curve = np.mean(captured_curves, axis=0)
    
    pct_1_idx = int(0.01 * 768)
    pct_5_idx = int(0.05 * 768)
    pct_10_idx = int(0.10 * 768)
    pct_20_idx = int(0.20 * 768)
    
    g_1 = mean_curve[pct_1_idx] * 100
    g_5 = mean_curve[pct_5_idx] * 100
    g_10 = mean_curve[pct_10_idx] * 100
    g_20 = mean_curve[pct_20_idx] * 100
    
    print("\n=======================================================")
    print("REAL ACTIVATION-WEIGHTED ENERGY CONCENTRATION (FFN INTERMEDIATE):")
    print(f" - Top 1% neurons (7 neurons):   {g_1:.2f}% energy")
    print(f" - Top 5% neurons (38 neurons):  {g_5:.2f}% energy")
    print(f" - Top 10% neurons (76 neurons): {g_10:.2f}% energy")
    print(f" - Top 20% neurons (153 neurons):{g_20:.2f}% energy")
    print("=======================================================")
    
    results = {
        "num_traces": len(captured_curves),
        "mean_activation_energy_captured": {
            "top_1_pct_energy": float(g_1),
            "top_5_pct_energy": float(g_5),
            "top_10_pct_energy": float(g_10),
            "top_20_pct_energy": float(g_20)
        }
    }
    
    out_path = "/home/palakm/MoEServingSim/qwen3_30b_plots/real_activation_energy_distribution.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Successfully saved real activation energy metrics to: {out_path}")

if __name__ == "__main__":
    main()
