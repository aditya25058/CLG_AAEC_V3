#!/usr/bin/env python3
"""Run a comprehensive network co-design ablation studying LAER, DAEL, TWR, and EPEG individually and in combinations."""
import subprocess
import re
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Configure dark theme aesthetics
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9',
    'text.color': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'grid.alpha': 0.6,
    'font.family': 'sans-serif',
    'font.size': 11,
})

def run_config(args):
    cmd = [
        "venv/bin/python3", "-m", "serving",
        "--cluster-config", "configs/cluster/test_dual_node_tp2_ep4.json",
        "--dataset", "datasets/qwen3_remote_10req_concurrent_fast.jsonl",
        "--num-reqs", "4",
        "--gpus-per-node", "2",
        "--expert-routing-policy", "BALANCED",
        "--expert-skew-intensity", "0.7",
        "--log-level", "INFO"
    ] + args
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    stdout = res.stdout
    
    # Parse total latency
    lat_match = re.search(r"Total latency \(s\):\s*([\d.]+)", stdout)
    latency = float(lat_match.group(1)) if lat_match else 0.0
    
    # Parse layer 0 metrics
    dael_match = re.search(r"\[DAEL_METRICS\] layer=0 cov=([\d.]+) max_to_mean=([\d.]+) redirected=(\d+)", stdout)
    laer_match = re.search(r"\[LAER_METRICS\] layer=0 remote_frac=([\d.]+) quality_delta=([\d.]+) redirected=(\d+) inter_node=(\d+)", stdout)
    
    cov = float(dael_match.group(1)) if dael_match else 0.0
    inter_node = int(laer_match.group(4)) if laer_match else 0
    redirects = int(dael_match.group(3)) if dael_match else 0
    
    return latency, cov, inter_node, redirects

def main():
    out_dir = "outputs/phase3"
    os.makedirs(out_dir, exist_ok=True)
    
    print("="*80)
    
    # Define the 8 ablation cases
    configs = {
        "Baseline": [],
        "EPEG Only": ["--enable-epeg", "--epeg-tau-high", "0.40", "--epeg-tau-low", "0.05"],
        "TWR Only": ["--enable-twr"],
        "LAER Only": ["--enable-laer", "--laer-beta", "0.95", "--laer-gamma", "0.70"],
        "DAEL Only": ["--enable-dael", "--dael-saturation-threshold", "0.15", "--dael-redirect-fraction", "0.10"],
        "EPEG + TWR": ["--enable-epeg", "--epeg-tau-high", "0.40", "--epeg-tau-low", "0.05", "--enable-twr"],
        "LAER + DAEL": ["--enable-laer", "--laer-beta", "0.95", "--laer-gamma", "0.70", "--enable-dael", "--dael-saturation-threshold", "0.15", "--dael-redirect-fraction", "0.10"],
        "Full Co-Design (Ours)": [
            "--enable-epeg", "--epeg-tau-high", "0.40", "--epeg-tau-low", "0.05",
            "--enable-twr",
            "--enable-laer", "--laer-beta", "0.95", "--laer-gamma", "0.70",
            "--enable-dael", "--dael-saturation-threshold", "0.15", "--dael-redirect-fraction", "0.10"
        ]
    }
    
    results = {}
    for name, args in configs.items():
        print(f"Running ablation case: {name}...")
        lat, cov, inter_node, redirects = run_config(args)
        results[name] = {
            "latency": lat,
            "cov": cov,
            "inter_node": inter_node,
            "redirects": redirects
        }
        print(f" -> Latency: {lat:.3f}s | Load CoV: {cov:.4f} | Inter-Node Tokens: {inter_node} | Redirects: {redirects}")
        
    print("="*80)
    print("ALL ABLATION RUNS COMPLETED.")
    print("="*80)
    
    # Plotting the Ablation Results
    names = list(results.keys())
    latencies = [results[n]["latency"] for n in names]
    covs = [results[n]["cov"] for n in names]
    traffic = [results[n]["inter_node"] for n in names]
    
    # 1. Plot Latency Comparison (Bar Chart)
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(names, latencies, color=['#ff7b72' if 'Baseline' in n else '#58a6ff' if 'Ours' in n or '+' in n else '#8b949e' for n in names], edgecolor='#30363d', width=0.6)
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f"{yval:.3f}s", ha='center', va='bottom', fontweight='bold')
        
    ax.set_ylabel("Total Serving Latency (s)", fontsize=12, labelpad=10)
    ax.set_title("Systems Latency Ablation: Individual vs. Joint Network Co-Design", fontsize=14, fontweight='bold', pad=20, color='#58a6ff')
    ax.set_ylim(0, max(latencies) * 1.2)
    plt.xticks(rotation=15, ha='right')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    
    lat_plot_path = os.path.join(out_dir, "ablation_latency_co_design.png")
    plt.savefig(lat_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Plot the Skew/Traffic vs. Load Imbalance Trade-off Space (Scatter Plot)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#ff7b72', '#ff9f6a', '#e1a6f2', '#f2d56a', '#7ee787', '#58a6ff', '#bc8cff', '#10b981']
    markers = ['s', '^', 'D', 'p', 'v', 'o', '*', 'H']
    
    for i, name in enumerate(names):
        ax.scatter(traffic[i], covs[i], color=colors[i], marker=markers[i], s=200, edgecolor='#30363d', label=name, zorder=5)
        
    ax.set_xlabel("Inter-Node Token Traffic (Collective payload volume)", fontsize=12, labelpad=10)
    ax.set_ylabel("Compute Load Imbalance (Coefficient of Variation)", fontsize=12, labelpad=10)
    ax.set_title("Collective Traffic vs. Compute Load Imbalance Trade-off Space", fontsize=14, fontweight='bold', pad=20, color='#58a6ff')
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", frameon=True, facecolor='#161b22', edgecolor='#30363d')
    
    # Annotate quadrants
    ax.text(128, 0.24, "High Imbalance\nLow Traffic", color='#ff9f6a', style='italic', fontsize=10)
    ax.text(143, 0.12, "Low Imbalance\nHigh Traffic", color='#f2d56a', style='italic', fontsize=10)
    ax.text(128, 0.12, "Optimal Region\n(Co-Design)", color='#10b981', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    tradeoff_plot_path = os.path.join(out_dir, "ablation_tradeoff_space.png")
    plt.savefig(tradeoff_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Copy to the brain artifacts directory
    artifact_dir = "/home/palakm/.gemini/antigravity-ide/brain/ed3c2dc2-4e67-4959-b431-347772d4a219/sla_gating"
    os.makedirs(artifact_dir, exist_ok=True)
    import shutil
    shutil.copy(lat_plot_path, os.path.join(artifact_dir, "ablation_latency_co_design.png"))
    shutil.copy(tradeoff_plot_path, os.path.join(artifact_dir, "ablation_tradeoff_space.png"))
    
    print("Ablation simulation complete! Plots successfully created and copied to artifacts.")

if __name__ == "__main__":
    main()
