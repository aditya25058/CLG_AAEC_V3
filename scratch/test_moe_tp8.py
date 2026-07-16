import torch
# Dynamically register kimi_moe model class and configs
try:
    from transformers import AutoConfig
    from transformers.models.qwen2_moe.configuration_qwen2_moe import Qwen2MoeConfig
    class KimiMoeConfig(Qwen2MoeConfig):
        model_type = "kimi_moe"
    AutoConfig.register("kimi_moe", KimiMoeConfig)

    from vllm.model_executor.models import ModelRegistry
    ModelRegistry.models["KimiMoeForCausalLM"] = ModelRegistry.models["Qwen2MoeForCausalLM"]
except Exception as e:
    print("Registration error:", e)

from vllm import LLM, SamplingParams

print("Booting vLLM engine with TP8 emulation...")
import tempfile
import json
import os

with tempfile.TemporaryDirectory() as tmpdir:
    config = {
        "architectures": ["KimiMoeForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 100000,
        "eos_token_id": 100001,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 7168,
        "initializer_range": 0.02,
        "intermediate_size": 2304, # TP8 sharded
        "max_position_embeddings": 163840,
        "max_window_layers": 60,
        "mlp_only_layers": [],
        "model_type": "kimi_moe",
        "moe_intermediate_size": 2048,
        "num_attention_heads": 16, # TP8 sharded
        "num_experts": 512,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 1, # Shrunk to 1 layer for speed
        "num_key_value_heads": 16, # TP8 sharded
        "rms_norm_eps": 1e-06,
        "torch_dtype": "bfloat16",
        "vocab_size": 12800
    }
    with open(os.path.join(tmpdir, "config.json"), "w") as f:
        json.dump(config, f)

    llm = LLM(
        model=tmpdir,
        skip_tokenizer_init=True,
        load_format="dummy",
        block_size=16,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.8, # More memory for test
        max_num_batched_tokens=2304,
        max_num_seqs=256,
        enforce_eager=True,
    )

    print("Engine booted successfully!")
    
    # Test different token lengths
    sampling_params = SamplingParams(max_tokens=1)
    
    for n in [1, 2, 4, 8, 16, 32]:
        print(f"\n--- Testing prompt length {n} ---")
        try:
            prompts = [[1] * n]
            outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
            print(f"Success for prompt length {n}!")
        except Exception as e:
            print(f"FAILED for prompt length {n}: {e}")
            break
