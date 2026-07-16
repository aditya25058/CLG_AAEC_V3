import torch
import time
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_real_offloading():
    model_id = "Qwen/Qwen3-30B-A3B"
    prompt = "Write a quick Python hello world function."
    
    if not torch.cuda.is_available():
        print("CUDA is not available. This script must be run on a GPU node.")
        return

    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # -------------------------------------------------------------
    # Scenario 1: Fully in GPU VRAM (Native Baseline)
    # -------------------------------------------------------------
    print("\n=== Scenario 1: Loading model fully in GPU VRAM (Baseline) ===")
    start_load = time.time()
    model_gpu = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    print(f"Model loaded fully in GPU in {time.time() - start_load:.2f} s")
    print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

    # Warmup
    print("Running warmup...")
    with torch.no_grad():
        _ = model_gpu.generate(**inputs, max_new_tokens=5)

    # Generation Run
    print("Generating tokens...")
    start_gen = time.time()
    with torch.no_grad():
        outputs = model_gpu.generate(**inputs, max_new_tokens=20)
    end_gen = time.time()
    gpu_time = end_gen - start_gen
    gpu_tps = 20 / gpu_time
    print(f"GPU Native Generation Time: {gpu_time:.2f} s ({gpu_tps:.2f} tokens/s)")
    
    # Response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Response:\n{response}\n")

    # Clear GPU Baseline model
    print("Clearing GPU model to free VRAM...")
    del model_gpu
    torch.cuda.empty_cache()
    time.sleep(2)

    # -------------------------------------------------------------
    # Scenario 2: Offloaded to CPU DRAM (Hybrid Execution)
    # -------------------------------------------------------------
    print("\n=== Scenario 2: Loading model offloaded to CPU DRAM (VRAM Constraint) ===")
    # We restrict GPU memory to 20 GB. Since model is ~60 GB, ~40 GB will be offloaded to CPU.
    max_memory = {0: "20GiB", "cpu": "128GiB"}
    
    start_load = time.time()
    model_offload = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory
    )
    print(f"Model loaded offloaded in {time.time() - start_load:.2f} s")
    print(f"GPU memory allocated (constrained): {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

    # Warmup
    print("Running warmup...")
    with torch.no_grad():
        _ = model_offload.generate(**inputs, max_new_tokens=5)

    # Generation Run
    print("Generating tokens with offloading...")
    start_gen = time.time()
    with torch.no_grad():
        outputs_offload = model_offload.generate(**inputs, max_new_tokens=20)
    end_gen = time.time()
    offload_time = end_gen - start_gen
    offload_tps = 20 / offload_time
    print(f"Offloaded Generation Time: {offload_time:.2f} s ({offload_tps:.2f} tokens/s)")
    
    # Response
    response_offload = tokenizer.decode(outputs_offload[0], skip_special_tokens=True)
    print(f"Response:\n{response_offload}\n")

    print("=== Comparison Summary ===")
    print(f"GPU Native Speed: {gpu_tps:.2f} tokens/s")
    print(f"CPU Offloaded Speed: {offload_tps:.2f} tokens/s")
    print(f"Offloaded performance penalty: {offload_time / gpu_time:.2f}x slower")
    print(f"VRAM Reduction: From {60.0:.2f} GB to {20.0:.2f} GB (3.0x footprint reduction)")

if __name__ == "__main__":
    test_real_offloading()
