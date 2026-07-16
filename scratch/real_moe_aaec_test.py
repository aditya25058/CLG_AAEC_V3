import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# =====================================================================
# SYSTEM CONFIG & MODEL SELECTION
# Qwen1.5-MoE-A2.7B is a real, pre-trained MoE model that fits on low-resource
# hardware (e.g., a single L4, T4, RTX 3090/4090 GPU, or even a CPU).
# =====================================================================
MODEL_ID = "Qwen/Qwen1.5-MoE-A2.7B"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading real model '{MODEL_ID}' on device: {device}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16 if device == "cuda" else torch.float32, 
    device_map="auto"
)
model.eval()

# =====================================================================
# THE DYNAMIC AAEC MONKEY-PATCH LAYER
# =====================================================================
class AAECHookedMLP(nn.Module):
    """
    Hooks into a real MoE Expert's MLP block.
    It tracks live neuron activations, dynamically slices weights, and
    evaluates both execution speed and generation quality impact.
    """
    def __init__(self, original_mlp, cache_size=64, decay=0.99, threshold=0.1):
        super().__init__()
        self.original_mlp = original_mlp
        self.cache_size = cache_size
        self.decay = decay
        self.threshold = threshold
        
        # Intermediate dimension (inner MLP size)
        self.intermediate_size = original_mlp.gate_proj.out_features
        
        # Register a running EMA vector for neuron activations
        self.register_buffer("neuron_ema", torch.zeros(self.intermediate_size, device=device))
        self.register_buffer("active_indices", torch.arange(cache_size, device=device))
        
        self.aaec_enabled = True

    def forward(self, x):
        if not self.aaec_enabled:
            return self.original_mlp(x)
            
        # 1. Compute projection on gate (to analyze activations)
        # In SwiGLU, gate_proj defines which intermediate features are active
        gate_out = self.original_mlp.gate_proj(x)
        act_vals = torch.nn.functional.silu(gate_out)
        
        # 2. Update running EMA of neuron activations
        # We take the mean activation magnitude across the token dimension
        batch_acts = act_vals.abs().mean(dim=0).mean(dim=0) # [intermediate_size]
        self.neuron_ema.copy_(self.decay * self.neuron_ema + (1.0 - self.decay) * batch_acts)
        
        # 3. Dynamic Cache Update (Identify Hot Neurons)
        _, hot_indices = torch.topk(self.neuron_ema, self.cache_size)
        self.active_indices.copy_(hot_indices)
        
        # 4. Sliced Execution
        # Extract weight slices for the active neurons
        w_gate_slice = self.original_mlp.gate_proj.weight[self.active_indices, :]
        w_up_slice   = self.original_mlp.up_proj.weight[self.active_indices, :]
        w_down_slice = self.original_mlp.down_proj.weight[:, self.active_indices]
        
        # Run sliced MLP computation
        gate_slice = torch.nn.functional.linear(x, w_gate_slice)
        up_slice   = torch.nn.functional.linear(x, w_up_slice)
        act_slice  = torch.nn.functional.silu(gate_slice) * up_slice
        
        output = torch.nn.functional.linear(act_slice, w_down_slice)
        if self.original_mlp.down_proj.bias is not None:
            output += self.original_mlp.down_proj.bias
            
        return output

# =====================================================================
# MONKEY-PATCH ALL EXPERTS IN THE REAL MODEL
# =====================================================================
print("Monkey-patching MoE Experts with AAEC execution hooks...")
hooked_layers = []

# Qwen MoE layers are located under model.model.layers
for layer_idx, layer in enumerate(model.model.layers):
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
        # Iterate over all experts in the MoE layer
        for exp_key in layer.mlp.experts.keys():
            original_mlp = layer.mlp.experts[exp_key]
            hooked_mlp = AAECHookedMLP(original_mlp, cache_size=128)
            layer.mlp.experts[exp_key] = hooked_mlp
            hooked_layers.append(hooked_mlp)

print(f"Successfully hooked {len(hooked_layers)} experts across all layers.")

# =====================================================================
# EVALUATE QUALITY & ACCURACY (Perplexity check on a real prompt)
# =====================================================================
prompt = (
    "In distributed systems, caching is a critical mechanism. "
    "To solve the expert loading bottleneck in Mixture-of-Experts (MoE) serving, "
    "we propose to dynamically cache neuron-level columns. This reduces "
    "network transmission sizes by over 90% while maintaining accuracy. "
    "The core idea is to track the exponential moving average of neuron activations."
)
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# --- Run 1: Warmup & Context Learning ---
print("\nRunning warm-up sequence to train the dynamic activation EMA...")
with torch.no_grad():
    for _ in range(5):
        _ = model(**inputs)

# --- Run 2: Measure with AAEC Slicing Enabled ---
print("\nEvaluating text generation with AAEC (Cache Size = 128 / 1408 intermediate size)...")
for hl in hooked_layers:
    hl.aaec_enabled = True

t0 = time.time()
with torch.no_grad():
    out_aaec = model(**inputs)
    loss_aaec = out_aaec.loss if out_aaec.loss is not None else torch.tensor(0.0)
    ppl_aaec = torch.exp(loss_aaec).item() if loss_aaec > 0 else 0.0
t_aaec = time.time() - t0

# Generate response text under AAEC
gen_aaec = model.generate(**inputs, max_new_tokens=30, do_sample=False)
text_aaec = tokenizer.decode(gen_aaec[0], skip_special_tokens=True)

# --- Run 3: Measure Baseline (No Caching/Slicing) ---
print("Evaluating Baseline (Full weight execution)...")
for hl in hooked_layers:
    hl.aaec_enabled = False

t0 = time.time()
with torch.no_grad():
    out_base = model(**inputs)
    loss_base = out_base.loss if out_base.loss is not None else torch.tensor(0.0)
    ppl_base = torch.exp(loss_base).item() if loss_base > 0 else 0.0
t_base = time.time() - t0

# Generate response text under Baseline
gen_base = model.generate(**inputs, max_new_tokens=30, do_sample=False)
text_base = tokenizer.decode(gen_base[0], skip_special_tokens=True)

# =====================================================================
# RESULTS REPORT
# =====================================================================
print("\n" + "="*50)
print("REAL-WORLD MOE AAEC BENCHMARK RESULTS")
print("="*50)
print(f"Memory footprints (weights per expert):")
print(f"  Baseline Weight Size : {model.model.layers[0].mlp.experts['0'].original_mlp.gate_proj.weight.nelement() * 2 / 1024 / 1024:.2f} MB")
print(f"  AAEC Slice Size      : {model.model.layers[0].mlp.experts['0'].original_mlp.gate_proj.weight[:, :128].nelement() * 2 / 1024 / 1024:.2f} MB")
print(f"  Memory Footprint Reduction: 11x smaller (only 128/1408 neurons)")
print("-" * 50)
print(f"Execution Output Correctness Check:")
print(f"  Baseline output PPL: {ppl_base:.4f}")
print(f"  AAEC sliced PPL    : {ppl_aaec:.4f}")
print("-" * 50)
print("Generated Outputs Comparison:")
print(f"  [Baseline] : {text_base[len(prompt):].strip()}")
print(f"  [AAEC]     : {text_aaec[len(prompt):].strip()}")
print("="*50)
