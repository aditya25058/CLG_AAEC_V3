#!/usr/bin/env python3
import os

def main():
    base_dir = "/home/palakm/MoEServingSim/evaluation"
    subdirs = [
        "configs",
        "utils",
        "scripts",
        "results",
        "plots",
        "results/traces/qwen3_30b",
        "results/traces/deepseek_v2_lite",
        "results/traces/mixtral_8x7b",
        "results/e01_lossless/qwen3_30b",
        "results/e01_lossless/deepseek_v2_lite",
        "results/e01_lossless/mixtral_8x7b",
        "results/e02_router/qwen3_30b",
        "results/e02_router/deepseek_v2_lite",
        "results/e02_router/mixtral_8x7b",
        "results/e03_energy/qwen3_30b",
        "results/e03_energy/deepseek_v2_lite",
        "results/e03_energy/mixtral_8x7b",
        "results/e04_cache/qwen3_30b",
        "results/e04_cache/deepseek_v2_lite",
        "results/e04_cache/mixtral_8x7b",
        "results/e05_latency/qwen3_30b",
        "results/e05_latency/deepseek_v2_lite",
        "results/e05_latency/mixtral_8x7b",
        "results/e06_bandwidth/qwen3_30b",
        "results/e06_bandwidth/deepseek_v2_lite",
        "results/e06_bandwidth/mixtral_8x7b",
        "results/e07_kernel/qwen3_30b",
        "results/e07_kernel/deepseek_v2_lite",
        "results/e07_kernel/mixtral_8x7b",
        "results/e08_stress/qwen3_30b",
        "results/e08_stress/deepseek_v2_lite",
        "results/e08_stress/mixtral_8x7b",
        "results/e09_quality/qwen3_30b",
        "results/e09_quality/deepseek_v2_lite",
        "results/e09_quality/mixtral_8x7b",
        "results/e10_ablation/qwen3_30b",
        "results/e10_ablation/deepseek_v2_lite",
        "results/e10_ablation/mixtral_8x7b",
        "results/e11_baselines/qwen3_30b",
        "results/e11_baselines/deepseek_v2_lite",
        "results/e11_baselines/mixtral_8x7b",
        "results/e12_scalability/qwen3_30b",
        "results/e12_scalability/deepseek_v2_lite",
        "results/e12_scalability/mixtral_8x7b",
        "plots/e01_lossless",
        "plots/e02_router",
        "plots/e03_energy",
        "plots/e04_cache",
        "plots/e05_latency",
        "plots/e06_bandwidth",
        "plots/e07_kernel",
        "plots/e08_stress",
        "plots/e09_quality",
        "plots/e10_ablation",
        "plots/e11_baselines",
        "plots/e12_scalability",
        "plots/paper_figures"
    ]
    
    print(f"Creating evaluation directory structure under {base_dir}...")
    for d in subdirs:
        path = os.path.join(base_dir, d)
        os.makedirs(path, exist_ok=True)
        print(f"Created: {path}")

    # Write configs/models.yaml
    models_yaml = """models:
  qwen3_30b:
    name: "Qwen/Qwen3-30B-A3B"
    path: "Qwen/Qwen3-30B-A3B"
    type: "qwen3_moe"
    num_experts: 128
    top_k: 8
    hidden_size: 4096
    intermediate_size: 768
    num_layers: 48
  deepseek_v2_lite:
    name: "deepseek-ai/DeepSeek-V2-Lite"
    path: "deepseek-ai/DeepSeek-V2-Lite"
    type: "deepseek_moe"
    num_experts: 64
    top_k: 6
    hidden_size: 2048
    intermediate_size: 1408
    num_layers: 27
  mixtral_8x7b:
    name: "mistralai/Mixtral-8x7B-v0.1"
    path: "mistralai/Mixtral-8x7B-v0.1"
    type: "mixtral"
    num_experts: 8
    top_k: 2
    hidden_size: 4096
    intermediate_size: 14336
    num_layers: 32
"""
    with open(os.path.join(base_dir, "configs/models.yaml"), "w") as f:
        f.write(models_yaml)
    print("Created config: configs/models.yaml")

    # Write configs/hardware.yaml
    hardware_yaml = """hardware:
  device: "cuda:0"
  num_gpus: 2
  interconnect:
    pcie_gen5_bw_gbps: 64.0
    nvlink_bw_gbps: 450.0
  memory:
    hbm_capacity_gb: 95.0
    dram_capacity_gb: 256.0
  cache:
    qwen3_30b:
      cache_size: 128  # columns per expert (128 / 768 = ~16.6% sparsity cached)
    deepseek_v2_lite:
      cache_size: 256  # columns per expert (256 / 1408 = ~18.1% sparsity cached)
    mixtral_8x7b:
      cache_size: 2048 # columns per expert (2048 / 14336 = ~14.2% sparsity cached)
"""
    with open(os.path.join(base_dir, "configs/hardware.yaml"), "w") as f:
        f.write(hardware_yaml)
    print("Created config: configs/hardware.yaml")

    # Write configs/experiments.yaml
    experiments_yaml = """experiments:
  seed: 42
  num_prompts: 100
  generation_tokens: 128
  quality_benchmarks:
    - mmlu
    - gsm8k
    - arc
"""
    with open(os.path.join(base_dir, "configs/experiments.yaml"), "w") as f:
        f.write(experiments_yaml)
    print("Created config: configs/experiments.yaml")

    print("\nInitialization complete!")

if __name__ == "__main__":
    main()
