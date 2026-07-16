import torch
from transformers import AutoConfig

try:
    config = AutoConfig.from_pretrained("Qwen/Qwen3-30B-A3B", trust_remote_code=True)
    print("Found config:", config)
except Exception as e:
    print("Failed to load config:", e)
