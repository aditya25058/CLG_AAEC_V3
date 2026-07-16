#!/usr/bin/env python3
import subprocess
import re
import os
import json
import shutil
import matplotlib.pyplot as plt
import numpy as np

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
    
    config_path = "configs/cluster/dual_node_moe_dp_ep_instance_small.json"
    dataset_path = "datasets/qwen3_remote_10req_concurrent.jsonl"
    
    variants = {
        "Baseline": [
            venv_python, "-m", "serving",
            "--cluster-config", config_path,
            "--dataset", dataset_path,
            "--num-reqs", "10",
            "--expert-routing-policy", "DATASET",
            "--gpus-per-node", "1",
            "--output", "outputs/phase3/epeg_ablation_baseline.csv"
        ],
        "Comm Only": [
            venv_python, "-m", "serving",
            "--cluster-config", config_path,
            "--dataset", dataset_path,
            "--num-reqs", "10",
            "--expert-routing-policy", "DATASET",
            "--gpus-per-node", "1",
            "--enable-epeg",
            "--epeg-exclude-compute",
            "--output", "outputs/phase3/epeg_ablation_comm_only.csv"
        ],
        "Compute Only": [
            venv_python, "-m", "serving",
            "--cluster-config", config_path,
            "--dataset", dataset_path,
            "--num-reqs", "10",
            "--expert-routing-policy", "DATASET",
            "--gpus-per-node", "1",
            "--enable-epeg",
            "--epeg-exclude-comm",
            "--output", "outputs/phase3/epeg_ablation_compute_only.csv"
        ],
        "Full EPEG": [
            venv_python, "-m", "serving",
            "--cluster-config", config_path,
            "--dataset", dataset_path,
            "--num-reqs", "10",
            "--expert-routing-policy", "DATASET",
            "--gpus-per-node", "1",
            "--enable-epeg",
            "--output", "outputs/phase3/epeg_ablation_full_epeg.csv"
        ]
    }
    
    results = {}
    
    for var_name, cmd in variants.items():
        print(f"\nEvaluating Variant: {var_name}")
        stdout = run_cmd(cmd)
        metrics = parse_metrics(stdout)
        results[var_name] = metrics
        print(f"Metrics: Latency = {metrics['total_latency_s']:.3f}s | TTFT = {metrics['avg_ttft_ms']:.2f}ms | TPOT = {metrics['avg_tpot_ms']:.2f}ms")
        
    # Write JSON results
    json_path = "outputs/phase3/epeg_gain_ablation.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {json_path}")
    
    # ----------------------------------------------------
    # Generate Stunning Visualizations
    # ----------------------------------------------------
    names = list(results.keys())
    latencies = [results[n]["total_latency_s"] for n in names]
    tpots = [results[n]["avg_tpot_ms"] for n in names]
    
    # Speedup values
    base_lat = results["Baseline"]["total_latency_s"]
    speedups = [base_lat / results[n]["total_latency_s"] for n in names]
    
    # Plot configurations
    colors = ["#64748B", "#F59E0B", "#10B981", "#3B82F6"] # Slate, Amber, Emerald, Blue
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel 1: End-to-end Serving Latency (s)
    bars1 = ax1.bar(names, latencies, color=colors, edgecolor="#1E293B", width=0.5, alpha=0.85)
    ax1.set_title("End-to-End Serving Latency (s)", fontsize=13, fontweight='bold', pad=12)
    ax1.set_ylabel("Latency (seconds)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.4)
    
    # Add values on top of bars
    for bar, speedup in zip(bars1, speedups):
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"{yval:.3f}s\n({speedup:.2f}x)", 
                 ha='center', va='bottom', fontsize=10, fontweight='semibold')
                 
    # Panel 2: Mean Time-Per-Output-Token (ms)
    bars2 = ax2.bar(names, tpots, color=colors, edgecolor="#1E293B", width=0.5, alpha=0.85)
    ax2.set_title("Mean Time-Per-Output-Token (ms)", fontsize=13, fontweight='bold', pad=12)
    ax2.set_ylabel("TPOT (ms)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.4)
    
    # Add values on top of bars
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.2f}ms", 
                 ha='center', va='bottom', fontsize=10, fontweight='semibold')
                 
    plt.suptitle("EPEG Performance Gain Ablation (Qwen3-235B, 16 GB/s Link)", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    plot_path = "outputs/phase3/epeg_gain_ablation.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Successfully generated EPEG gain ablation plot at {plot_path}")
    
    # Copy files to artifact directory
    artifact_dir = "/home/palakm/.gemini/antigravity-ide/brain/ed3c2dc2-4e67-4959-b431-347772d4a219"
    try:
        shutil.copy(json_path, os.path.join(artifact_dir, "epeg_gain_ablation.json"))
        shutil.copy(plot_path, os.path.join(artifact_dir, "epeg_gain_ablation.png"))
        print(f"Successfully copied outputs to artifacts directory.")
    except Exception as e:
        print(f"Failed to copy to artifacts: {e}")

if __name__ == "__main__":
    main()
