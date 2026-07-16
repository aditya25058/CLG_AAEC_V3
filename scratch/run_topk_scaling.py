#!/usr/bin/env python3
import subprocess
import re
import os
import json
import shutil
import matplotlib.pyplot as plt

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
    print(f"Running command with env: {env} | Cmd: {' '.join(cmd)}")
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
    
    topk_vals = [2, 4, 8]
    
    results = {}
    
    for k in topk_vals:
        results[str(k)] = {}
        # Environment override for Top-k
        env_override = {"OVERRIDE_TOPK": str(k)}
        
        # 1. Baseline Run (EPEG Disabled)
        cmd_disabled = [
            venv_python, "-m", "serving",
            "--cluster-config", config_path,
            "--dataset", dataset_path,
            "--num-reqs", "10",
            "--expert-routing-policy", "DATASET",
            "--gpus-per-node", "1",
            "--output", f"outputs/phase3/epeg_topk_{k}_disabled.csv"
        ]
        print(f"\nEvaluating Top-k = {k} | EPEG = DISABLED")
        stdout_disabled = run_cmd(cmd_disabled, env=env_override)
        metrics_disabled = parse_metrics(stdout_disabled)
        results[str(k)]["disabled"] = metrics_disabled
        print(f"Disabled Latency: {metrics_disabled['total_latency_s']:.3f}s")
        
        # 2. EPEG Enabled Run
        cmd_enabled = [
            venv_python, "-m", "serving",
            "--cluster-config", config_path,
            "--dataset", dataset_path,
            "--num-reqs", "10",
            "--expert-routing-policy", "DATASET",
            "--gpus-per-node", "1",
            "--enable-epeg",
            "--output", f"outputs/phase3/epeg_topk_{k}_enabled.csv"
        ]
        print(f"Evaluating Top-k = {k} | EPEG = ENABLED")
        stdout_enabled = run_cmd(cmd_enabled, env=env_override)
        metrics_enabled = parse_metrics(stdout_enabled)
        results[str(k)]["enabled"] = metrics_enabled
        
        speedup = metrics_disabled["total_latency_s"] / metrics_enabled["total_latency_s"] if metrics_enabled["total_latency_s"] > 0 else 1.0
        results[str(k)]["speedup"] = speedup
        print(f"Enabled Latency: {metrics_enabled['total_latency_s']:.3f}s | Speedup: {speedup:.2f}x")
        
    # Save results to JSON
    json_path = "outputs/phase3/epeg_topk_scaling.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {json_path}")
    
    # ----------------------------------------------------
    # Generate Stunning Visualizations
    # ----------------------------------------------------
    speedups = [results[str(k)]["speedup"] for k in topk_vals]
    
    plt.figure(figsize=(8, 5))
    plt.plot(topk_vals, speedups, marker='o', markersize=8, color="#3B82F6", linewidth=2.5, label="EPEG Serving Speedup")
    
    plt.title("EPEG Scaling Benefit vs. Routing Top-k Experts (Qwen3-235B)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Top-k Active Experts (k)", fontsize=11)
    plt.ylabel("Serving Speedup Ratio (x)", fontsize=11)
    plt.xticks(topk_vals)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.ylim(1.0, max(speedups) + 0.1)
    
    # Add values next to markers
    for x, y in zip(topk_vals, speedups):
        plt.text(x, y + 0.02, f"{y:.2f}x", ha='center', va='bottom', fontsize=10, fontweight='semibold')
        
    plt.legend(loc="upper left")
    plt.tight_layout()
    
    plot_path = "outputs/phase3/epeg_topk_scaling.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Successfully generated EPEG Top-k scaling plot at {plot_path}")
    
    # Copy files to artifact directory
    artifact_dir = "/home/palakm/.gemini/antigravity-ide/brain/ed3c2dc2-4e67-4959-b431-347772d4a219"
    try:
        shutil.copy(json_path, os.path.join(artifact_dir, "epeg_topk_scaling.json"))
        shutil.copy(plot_path, os.path.join(artifact_dir, "epeg_topk_scaling.png"))
        print(f"Successfully copied outputs to artifacts directory.")
    except Exception as e:
        print(f"Failed to copy to artifacts: {e}")

if __name__ == "__main__":
    main()
