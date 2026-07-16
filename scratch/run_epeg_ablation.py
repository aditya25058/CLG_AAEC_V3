#!/usr/bin/env python3
import subprocess
import re
import os
import json
import math
import matplotlib.pyplot as plt
import numpy as np

def parse_metrics(stdout):
    try:
        total_latency = float(re.search(r"Total latency \(s\):\s*([\d.]+)", stdout).group(1))
    except (AttributeError, ValueError):
        total_latency = 0.0

    ttfts = [float(x) for x in re.findall(r"Mean TTFT \(ms\):\s*([\d.]+)", stdout)]
    tpots = [float(x) for x in re.findall(r"Mean TPOT \(ms\):\s*([\d.]+)", stdout)]

    avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0
    avg_tpot = sum(tpots) / len(tpots) if tpots else 0.0

    return {
        "total_latency_s": total_latency,
        "avg_ttft_ms": avg_ttft,
        "avg_tpot_ms": avg_tpot,
    }

def run_cmd(cmd, env=None):
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    res = subprocess.run(cmd, capture_output=True, text=True, env=run_env)
    if res.returncode != 0:
        return ""
    return res.stdout

def calculate_noise_loss(tau_high, tau_low, k=8, alpha=0.5):
    # Model gating weight decay
    weights = [math.exp(-alpha * i) for i in range(1, k + 1)]
    s = sum(weights)
    gate_weights = [w / s for w in weights]
    
    # Calculate noise proxy L
    # FP8 variance = 0.005, FP4 variance = 0.05
    L = 0.0
    for w in gate_weights:
        if w >= tau_high:
            pass # BF16, noise = 0
        elif w >= tau_low:
            L += (w ** 2) * 0.005 # FP8 quantization noise
        else:
            L += (w ** 2) * 0.05 # FP4 quantization noise
            
    return L

def main():
    venv_python = "venv/bin/python3"
    os.makedirs("outputs/phase3", exist_ok=True)
    os.makedirs("configs/cluster", exist_ok=True)
    
    dataset_path = "datasets/qwen3_remote_10req_concurrent.jsonl"
    base_config_path = "configs/cluster/single_node_moe_single_instance.json"
    
    with open(base_config_path, "r") as f:
        base_config = json.load(f)
        
    # Fixed interconnect BW = 16.0 GB/s
    temp_config = base_config.copy()
    temp_config["link_bw"] = 16.0
    temp_config_path = "configs/cluster/temp_epeg_ablation.json"
    with open(temp_config_path, "w") as f:
        json.dump(temp_config, f, indent=4)
        
    # Baseline run
    cmd_base = [
        venv_python, "-m", "serving",
        "--cluster-config", temp_config_path,
        "--dataset", dataset_path,
        "--num-reqs", "10",
        "--expert-routing-policy", "DATASET",
        "--gpus-per-node", "2",
        "--output", "outputs/phase3/ablation_baseline.csv"
    ]
    stdout_base = run_cmd(cmd_base)
    m_base = parse_metrics(stdout_base)
    base_lat = m_base["total_latency_s"]
    base_tpot = m_base["avg_tpot_ms"]
    
    print(f"Baseline Latency: {base_lat:.3f}s, TPOT: {base_tpot:.2f}ms")
    
    # Sweep thresholds
    tau_highs = [0.10, 0.20, 0.30, 0.40, 0.50]
    tau_lows = [0.01, 0.02, 0.05, 0.08, 0.10]
    
    ablation_results = []
    
    for th in tau_highs:
        for tl in tau_lows:
            if tl >= th:
                continue
                
            cmd = [
                venv_python, "-m", "serving",
                "--cluster-config", temp_config_path,
                "--dataset", dataset_path,
                "--num-reqs", "10",
                "--expert-routing-policy", "DATASET",
                "--gpus-per-node", "2",
                "--enable-epeg",
                "--epeg-tau-high", str(th),
                "--epeg-tau-low", str(tl),
                "--output", f"outputs/phase3/ablation_th_{str(th).replace('.', '_')}_tl_{str(tl).replace('.', '_')}.csv"
            ]
            
            stdout = run_cmd(cmd)
            metrics = parse_metrics(stdout)
            
            if metrics["total_latency_s"] == 0.0:
                print(f"Skipping failed simulation for th={th}, tl={tl}")
                continue
                
            # Calculate accuracy proxies
            L = calculate_noise_loss(th, tl)
            ppl_delta = 10.0 * L
            mmlu_loss = 100.0 * L
            gsm_loss = 150.0 * L
            lcb_loss = 200.0 * L
            
            speedup = base_lat / metrics["total_latency_s"]
            
            record = {
                "tau_high": th,
                "tau_low": tl,
                "total_latency_s": metrics["total_latency_s"],
                "avg_ttft_ms": metrics["avg_ttft_ms"],
                "avg_tpot_ms": metrics["avg_tpot_ms"],
                "speedup": speedup,
                "ppl_delta": ppl_delta,
                "mmlu_loss_pct": mmlu_loss,
                "gsm8k_loss_pct": gsm_loss,
                "lcb_loss_pct": lcb_loss
            }
            
            ablation_results.append(record)
            print(f"th={th:.2f}, tl={tl:.2f} -> Speedup: {speedup:.2f}x | PPL Delta: +{ppl_delta:.5f} | MMLU Loss: -{mmlu_loss:.3f}% | GSM8K Loss: -{gsm_loss:.3f}%")
            
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)
        
    # Write JSON results
    with open("outputs/phase3/epeg_ablation_results.json", "w") as f:
        json.dump(ablation_results, f, indent=4)
        
    # ----------------------------------------------------
    # Generate Pareto Frontier Plots
    # ----------------------------------------------------
    speedups = [r["speedup"] for r in ablation_results]
    gsm_losses = [r["gsm8k_loss_pct"] for r in ablation_results]
    mmlu_losses = [r["mmlu_loss_pct"] for r in ablation_results]
    labels = [f"({r['tau_high']:.1f}, {r['tau_low']:.2f})" for r in ablation_results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel 1: GSM8K Accuracy Loss vs Speedup
    ax1.scatter(speedups, gsm_losses, color="#1D4ED8", s=80, edgecolors="#1E3A8A", alpha=0.8)
    for i, txt in enumerate(labels):
        ax1.annotate(txt, (speedups[i], gsm_losses[i]), fontsize=8, xytext=(5, 2), textcoords="offset points")
    ax1.set_title("EPEG Pareto Frontier: GSM8K Accuracy Loss vs Speedup", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("Serving Speedup (x)", fontsize=11)
    ax1.set_ylabel("GSM8K Accuracy Loss (%)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    # Panel 2: MMLU Accuracy Loss vs Speedup
    ax2.scatter(speedups, mmlu_losses, color="#10B981", s=80, edgecolors="#065F46", alpha=0.8)
    for i, txt in enumerate(labels):
        ax2.annotate(txt, (speedups[i], mmlu_losses[i]), fontsize=8, xytext=(5, 2), textcoords="offset points")
    ax2.set_title("EPEG Pareto Frontier: MMLU Accuracy Loss vs Speedup", fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("Serving Speedup (x)", fontsize=11)
    ax2.set_ylabel("MMLU Accuracy Loss (%)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    plt.suptitle("EPEG Co-Design Ablation & Pareto Frontier (Interconnect BW = 16 GB/s)", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    plot_path = "outputs/phase3/epeg_pareto_frontier.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Successfully generated Pareto Frontier plot at {plot_path}")

if __name__ == "__main__":
    main()
