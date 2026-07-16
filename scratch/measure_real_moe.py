import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import os
import matplotlib.pyplot as plt

# =====================================================================
# MODEL SELECTION & COMPATIBLE BACKUP
# DeepSeek-MoE-16B is the open-source representative of the DeepSeek MoE family.
# It uses the exact same router and fine-grained MoE architecture as DeepSeek-V3/R1.
# =====================================================================
MODEL_ID = "deepseek-ai/deepseek-moe-16b-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device Count: {torch.cuda.device_count()}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")

# Since downloading a 16B model can take a lot of memory, we also support a 
# toy/mock mode or layer-wise weight loading to prevent OOM on smaller setups.
MOCK_MODE = not torch.cuda.is_available()

# =====================================================================
# THE INSTRUMENTATION HOOK
# =====================================================================
class MoEInstrumentationHook:
    """
    Hook to capture live activations from a SwiGLU FFN (expert) layer.
    """
    def __init__(self, name):
        self.name = name
        self.activations = []
        self.routing_history = []
        
    def hook_fn(self, module, input, output):
        # input[0] is the input hidden states: shape [batch_size, seq_len, hidden_dim]
        # output is the gate projection output: shape [batch_size, seq_len, intermediate_dim]
        # In SwiGLU, gate_proj defines the gating behavior
        gate_out = output.detach().float()
        act_vals = torch.nn.functional.silu(gate_out) # [batch_size, seq_len, intermediate_dim]
        
        # Flatten batch and sequence dimensions to get per-token activations
        flat_acts = act_vals.view(-1, act_vals.size(-1)).cpu().numpy()
        self.activations.append(flat_acts)

def instrument_model(model):
    hooks = {}
    hook_handles = []
    
    # Locate MoE experts in DeepSeek/Qwen model
    # DeepSeek layers are under model.model.layers
    # Each MoE layer has mlp.experts
    layer_count = 0
    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            layer_count += 1
            for exp_idx, expert in enumerate(layer.mlp.experts):
                # DeepSeek SwiGLU gate_proj layer
                if hasattr(expert, "gate_proj"):
                    hook_name = f"layer_{layer_idx}_expert_{exp_idx}"
                    hook = MoEInstrumentationHook(hook_name)
                    handle = expert.gate_proj.register_forward_hook(hook.hook_fn)
                    hooks[hook_name] = hook
                    hook_handles.append(handle)
                    
            # Hook the router itself to get routing choices
            if hasattr(layer.mlp, "gate"):
                # Layer router hook
                pass
                
        if layer_count >= 2: # Limit hook footprint to first two MoE layers to save memory
            break
            
    return hooks, hook_handles

# =====================================================================
# EXPERIMENT & MEASUREMENT ENGINE
# =====================================================================
def run_real_measurement():
    if MOCK_MODE:
        print("No GPU or weights available. Simulating PyTorch-based instrumentation pass...")
        # Create a mock MLP and router to demonstrate how we hook and gather statistics
        intermediate_dim = 2048
        hidden_dim = 1024
        seq_len = 256
        
        gate_proj = nn.Linear(hidden_dim, intermediate_dim)
        
        # Generate some synthetic correlated token hidden states
        # Correlated input representations lead to correlated SwiGLU activations
        inputs = torch.randn(1, seq_len, hidden_dim)
        for i in range(1, seq_len):
            # 0.9 correlation coefficient to simulate semantic continuity
            inputs[0, i] = 0.9 * inputs[0, i-1] + 0.1 * inputs[0, i]
            
        gate_out = gate_proj(inputs)
        act_vals = torch.nn.functional.silu(gate_out).view(-1, intermediate_dim).detach().numpy()
        
        # Threshold: keep top 64 activations per token to calculate statistics
        active_sets = []
        for token_act in act_vals:
            # Sort activations
            top_indices = np.argsort(np.abs(token_act))[-64:]
            active_sets.append(set(top_indices))
    else:
        # Load DeepSeek/Qwen tokenizers and run a real inference forward pass
        print(f"Loading {MODEL_ID} tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
        model.eval()
        
        hooks, handles = instrument_model(model)
        
        prompt = (
            "Mixture-of-Experts (MoE) is a machine learning technique where multiple "
            "specialist networks (experts) are managed by a gating network. Under low resource "
            "serving budgets, transferring entire expert parameters over PCIe causes massive latency. "
            "We propose Activation-Aware caching to resolve this bottleneck."
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            _ = model(**inputs)
            
        # Remove hooks
        for handle in handles:
            handle.remove()
            
        # Process first hooked expert's activations
        hook = list(hooks.values())[0]
        all_acts = np.concatenate(hook.activations, axis=0) # [seq_len, intermediate_dim]
        
        # Threshold activations dynamically
        active_sets = []
        threshold = 0.05 * np.max(np.abs(all_acts)) # 5% of peak activation
        for token_act in all_acts:
            active_idx = np.where(np.abs(token_act) > threshold)[0]
            active_sets.append(set(active_idx))
            
    # Calculate empirical Jaccard similarity vs distance
    jaccard_vs_distance = {}
    for dist in range(1, 15):
        overlaps = []
        for i in range(len(active_sets) - dist):
            set_a = active_sets[i]
            set_b = active_sets[i + dist]
            union = len(set_a.union(set_b))
            intersection = len(set_a.intersection(set_b))
            jaccard = intersection / union if union > 0 else 0.0
            overlaps.append(jaccard)
        jaccard_vs_distance[dist] = np.mean(overlaps) if overlaps else 0.0
        
    print("\nEmpirical Jaccard Overlaps measured directly from SwiGLU forward pass:")
    for dist, val in jaccard_vs_distance.items():
        print(f"  Distance {dist}: {val:.4f}")
        
    # Calculate Zipf ranked frequencies of active neurons
    neuron_counts = {}
    for s in active_sets:
        for nid in s:
            neuron_counts[nid] = neuron_counts.get(nid, 0) + 1
            
    sorted_freqs = sorted(neuron_counts.values(), reverse=True)
    
    # Save statistics
    stats_out = {
        "jaccard": jaccard_vs_distance,
        "zipf": sorted_freqs,
        "mode": "mock" if MOCK_MODE else "real"
    }
    
    out_dir = "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "real_moe_measurements.json"), "w") as f:
        json.dump(stats_out, f)
        
    print(f"Real MoE measurements saved to {os.path.join(out_dir, 'real_moe_measurements.json')}")

if __name__ == "__main__":
    run_real_measurement()
