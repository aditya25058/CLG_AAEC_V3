#!/usr/bin/env python3
import subprocess
import re
import os
import json

def parse_metrics(stdout):
    try:
        total_latency = float(re.search(r"Total latency \(s\):\s*([\d.]+)", stdout).group(1))
    except (AttributeError, ValueError):
        total_latency = 0.0

    try:
        prompt_thru = float(re.search(r"Average prompt throughput \(tok/s\):\s*([\d.]+)", stdout).group(1))
    except (AttributeError, ValueError):
        prompt_thru = 0.0

    try:
        gen_thru = float(re.search(r"Average generation throughput \(tok/s\):\s*([\d.]+)", stdout).group(1))
    except (AttributeError, ValueError):
        gen_thru = 0.0

    ttfts = [float(x) for x in re.findall(r"Mean TTFT \(ms\):\s*([\d.]+)", stdout)]
    tpots = [float(x) for x in re.findall(r"Mean TPOT \(ms\):\s*([\d.]+)", stdout)]

    avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0
    avg_tpot = sum(tpots) / len(tpots) if tpots else 0.0

    return {
        "total_latency_s": total_latency,
        "prompt_thru_tok_s": prompt_thru,
        "gen_thru_tok_s": gen_thru,
        "avg_ttft_ms": avg_ttft,
        "avg_tpot_ms": avg_tpot,
    }

def run_cmd(cmd, env=None):
    print(f"Running command: {' '.join(cmd)}")
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    res = subprocess.run(cmd, capture_output=True, text=True, env=run_env)
    if res.returncode != 0:
        print(f"Command failed with code {res.returncode}")
        print(res.stderr)
        return ""
    return res.stdout

def main():
    venv_python = "venv/bin/python3"
    os.makedirs("outputs/phase3", exist_ok=True)
    os.makedirs("configs/cluster", exist_ok=True)
    
    # Define models to sweep
    models_to_sweep = {
        "Qwen3-235B": {
            "config": "configs/cluster/single_node_moe_single_instance.json",
            "dataset": "datasets/qwen3_remote_10req_concurrent.jsonl",
            "num_npus": 2
        },
        "DeepSeek-R1": {
            "config": "configs/cluster/single_node_deepseek_r1_h100_tp2.json",
            "dataset": "datasets/deepseek_remote_10req_concurrent.jsonl",
            "num_npus": 2
        },
        "Llama4-Maverick": {
            "config": "configs/cluster/single_node_llama4_maverick_h100_tp2_pcie.json",
            "dataset": "datasets/llama4_remote_10req_concurrent.jsonl",
            "num_npus": 2
        }
    }
    
    cross_comparison_results = {}
    
    for model_name, model_info in models_to_sweep.items():
        print(f"\nEvaluating Model: {model_name}...")
        
        # Load baseline config and patch link_bw to 16.0 GB/s to represent an NVLink/PCIe intermediate bottleneck
        with open(model_info["config"], "r") as f:
            config_data = json.load(f)
        config_data["link_bw"] = 16.0
        
        temp_config_path = f"configs/cluster/temp_cross_{model_name.lower()}.json"
        with open(temp_config_path, "w") as f:
            json.dump(config_data, f, indent=4)
            
        cross_comparison_results[model_name] = {}
        
        for enabled in [False, True]:
            enabled_str = "enabled" if enabled else "disabled"
            csv_out = f"outputs/phase3/cross_{model_name.lower()}_epeg_{enabled_str}.csv"
            
            cmd = [
                venv_python, "-m", "serving",
                "--cluster-config", temp_config_path,
                "--dataset", model_info["dataset"],
                "--num-reqs", "10",
                "--expert-routing-policy", "DATASET",
                "--gpus-per-node", "2",
                "--output", csv_out
            ]
            if enabled:
                cmd.append("--enable-epeg")
                cmd.extend(["--epeg-tau-high", "0.40", "--epeg-tau-low", "0.05"])
                
            stdout = run_cmd(cmd)
            metrics = parse_metrics(stdout)
            
            cross_comparison_results[model_name][enabled_str] = metrics
            
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
            
        # Calculate speedup
        base_lat = cross_comparison_results[model_name]["disabled"]["total_latency_s"]
        epeg_lat = cross_comparison_results[model_name]["enabled"]["total_latency_s"]
        speedup = base_lat / epeg_lat if epeg_lat > 0 else 1.0
        cross_comparison_results[model_name]["speedup"] = speedup
        
        print(f"{model_name} Results:")
        print(f"  Baseline Latency: {base_lat:.3f}s | EPEG Latency: {epeg_lat:.3f}s")
        print(f"  Speedup Ratio: {speedup:.2f}x")
        
    # Write JSON results
    with open("outputs/phase3/epeg_cross_model_comparison.json", "w") as f:
        json.dump(cross_comparison_results, f, indent=4)
        
    print("\nSuccessfully finished cross-model evaluation sweeps!")
    print("Results saved in outputs/phase3/epeg_cross_model_comparison.json")

if __name__ == "__main__":
    main()
