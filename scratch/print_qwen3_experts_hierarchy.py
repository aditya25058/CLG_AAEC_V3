import torch
from transformers import AutoModelForCausalLM

print("Loading Qwen3 model architecture...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-30B-A3B",
    trust_remote_code=True,
    device_map="meta"
)

experts = model.model.layers[0].mlp.experts
print("\n--- experts children ---")
for name, module in experts.named_children():
    print(f"Child name: {name}, Class: {module.__class__.__name__}")

print("\n--- experts parameter names ---")
for name, param in experts.named_parameters():
    print(f"Param name: {name}, Shape: {param.shape}")
