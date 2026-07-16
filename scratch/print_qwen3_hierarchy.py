import torch
from transformers import AutoModelForCausalLM

print("Loading Qwen3 model architecture...")
# Load config and initialize empty/meta model first to inspect keys
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-30B-A3B",
    trust_remote_code=True,
    device_map="meta",
    torch_dtype=torch.bfloat16
)

print("First layer FFN modules:")
for name, module in model.model.layers[0].mlp.named_modules():
    if "down_proj" in name or "down" in name or "w2" in name:
        print(f"Name: {name}, Module: {module}")
