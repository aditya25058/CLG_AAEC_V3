import torch
from transformers import AutoModelForCausalLM

print("Loading Qwen3 model architecture...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-30B-A3B",
    trust_remote_code=True,
    device_map="meta"
)

print("\n--- Layer 0 sub-modules ---")
for name, module in model.model.layers[0].named_children():
    print(f"Child name: {name}, Class: {module.__class__.__name__}")
    if name == "mlp":
        print("MLP children:")
        for sub_name, sub_module in module.named_children():
            print(f"  MLP Sub-child name: {sub_name}, Class: {sub_module.__class__.__name__}")
