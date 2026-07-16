#!/usr/bin/env python3
"""Run a multi-node scaling sweep (2, 4, 8, 16 nodes) to evaluate ours vs baseline and other configurations."""
import subprocess
import re
import os
import json
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

def make_temp_config(original_path, temp_path, num_nodes=2, bw=1.0):
    with open(original_path, 'r') as f:
        data = json.load(f)
    
    data["num_nodes"] = num_nodes
    data["link_bw"] = bw
    
    # Scale ep_size in instances list based on num_nodes
    single_node = data["nodes"][0]
    single_node["instances"][0]["ep_size"] = num_nodes * 2
    
    # Generate list of nodes
    data["nodes"] = [json.loads(json.dumps(single_node)) for _ in range(num_nodes)]
    
    with open(temp_path, 'w') as f:
        json.dump(data, f, indent=4)

def run_config(config_path, args):
    cmd = [
        "venv/bin/python3", "-m", "serving",
        "--cluster-config", config_path,
        "--dataset", "datasets/qwen3_remote_10req_concurrent_fast.jsonl",
        "--num-reqs", "4",
        "--gpus-per-node", "2",
        "--expert-routing-policy", "BALANCED",
        "--expert-skew-intensity", "0.7",
        "--log-level", "INFO"
    ] + args
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    stdout = res.stdout
    
    # Parse total clocks in ns for unrounded high precision
    clocks_match = re.search(r"Total clocks \(ns\):\s*(\d+)", stdout)
    if clocks_match:
        latency = float(clocks_match.group(1)) / 1e9
    else:
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
    base_config = "configs/cluster/test_dual_node_tp2_ep4.json"
    out_dir = "outputs/phase5"
    os.makedirs(out_dir, exist_ok=True)
    
    node_counts = [2, 4, 8, 16]
    
    # 5 Configurations to evaluate
    configs_meta = {
        "Baseline": [],
        "LAER Only": ["--enable-laer", "--laer-beta", "0.95", "--laer-gamma", "0.70"],
        "TWR Only": ["--enable-twr"],
        "EPEG + TWR": ["--enable-epeg", "--epeg-tau-high", "0.40", "--epeg-tau-low", "0.05", "--enable-twr"],
        "Ours (Full Co-Design)": [
            "--enable-epeg", "--epeg-tau-high", "0.40", "--epeg-tau-low", "0.05",
            "--enable-twr",
            "--enable-laer", "--laer-beta", "0.95", "--laer-gamma", "0.70",
            "--enable-dael", "--dael-saturation-threshold", "0.15", "--dael-redirect-fraction", "0.10"
        ]
    }
    
    results = {name: [] for name in configs_meta}
    inter_node_results = {name: [] for name in configs_meta}
    
    model_config_path = "configs/model/Qwen/Qwen3-235B-A22B.json"
    with open(model_config_path, "r") as f:
        m_config = json.load(f)
    orig_layers = m_config["num_hidden_layers"]
    
    print(f"Temporarily setting {model_config_path} hidden layers to 8 (was {orig_layers}) to accelerate Astra-Sim scaling sweep...")
    m_config["num_hidden_layers"] = 8
    with open(model_config_path, "w") as f:
        json.dump(m_config, f, indent=2)
        
    try:
        for nc in node_counts:
            print(f"\n--- Evaluating cluster with {nc} Nodes (EP size = {nc*2}) ---")
            temp_config_path = f"configs/cluster/temp_scaling_nc_{nc}.json"
            
            # We simulate under a bandwidth-constrained 1.0 GB/s link to stress the network
            make_temp_config(base_config, temp_config_path, num_nodes=nc, bw=1.0)
            
            for name, args in configs_meta.items():
                print(f"Running config: {name} on {nc} nodes...")
                lat, cov, inter_node, redirects = run_config(temp_config_path, args)
                results[name].append(lat)
                inter_node_results[name].append(inter_node)
                print(f" -> Latency: {lat:.3f}s | Inter-Node: {inter_node} | CoV: {cov:.4f} | Redirects: {redirects}")
                
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
    finally:
        print(f"Restoring {model_config_path} hidden layers back to {orig_layers}...")
        m_config["num_hidden_layers"] = orig_layers
        with open(model_config_path, "w") as f:
            json.dump(m_config, f, indent=2)
            
    # Save raw results
    raw_results = {
        "node_counts": node_counts,
        "latency": results,
        "inter_node_tokens": inter_node_results
    }
    with open(os.path.join(out_dir, "scaling_results.json"), "w") as f:
        json.dump(raw_results, f, indent=4)
        
    # Plot 1: Latency Scaling (Line Plot)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#ff7b72', '#f2d56a', '#e1a6f2', '#58a6ff', '#10b981']
    markers = ['s', 'p', 'D', 'o', 'H']
    
    for i, (name, latencies) in enumerate(results.items()):
        latencies_ms = [l * 1000.0 for l in latencies]
        ax.plot(node_counts, latencies_ms, color=colors[i], marker=markers[i], linewidth=2.5, markersize=8, label=name)
        
    ax.set_xlabel("Number of Distributed Nodes (Cluster Scale)", fontsize=12, labelpad=10)
    ax.set_ylabel("Total Serving Latency (ms)", fontsize=12, labelpad=10)
    ax.set_title("Serving Latency Scale-out Trends (2 - 16 Nodes, 1.0 GB/s Interconnect)", fontsize=14, fontweight='bold', pad=20, color='#58a6ff')
    ax.set_xticks(node_counts)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", frameon=True, facecolor='#161b22', edgecolor='#30363d')
    
    plt.tight_layout()
    lat_plot_path = os.path.join(out_dir, "scaling_latency.png")
    plt.savefig(lat_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Speedup Scaling (Line Plot)
    fig, ax = plt.subplots(figsize=(10, 6))
    baseline_lats = np.array(results["Baseline"])
    
    # Speedup relative to Baseline
    for i, (name, latencies) in enumerate(results.items()):
        if name == "Baseline":
            continue
        speedups = baseline_lats / np.array(latencies)
        ax.plot(node_counts, speedups, color=colors[i], marker=markers[i], linewidth=2.5, markersize=8, label=name)
        
    ax.set_xlabel("Number of Distributed Nodes (Cluster Scale)", fontsize=12, labelpad=10)
    ax.set_ylabel("Speedup Ratio (x)", fontsize=12, labelpad=10)
    ax.set_title("Co-Design Speedup Scaling (Relative to Baseline)", fontsize=14, fontweight='bold', pad=20, color='#58a6ff')
    ax.set_xticks(node_counts)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", frameon=True, facecolor='#161b22', edgecolor='#30363d')
    
    plt.tight_layout()
    speedup_plot_path = os.path.join(out_dir, "scaling_speedup.png")
    plt.savefig(speedup_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Copy to the brain artifacts directory
    artifact_dir = "/home/palakm/.gemini/antigravity-ide/brain/ed3c2dc2-4e67-4959-b431-347772d4a219/sla_gating"
    os.makedirs(artifact_dir, exist_ok=True)
    import shutil
    shutil.copy(lat_plot_path, os.path.join(artifact_dir, "scaling_latency.png"))
    shutil.copy(speedup_plot_path, os.path.join(artifact_dir, "scaling_speedup.png"))
    
    print("Multi-node scaling sweep complete! Plots successfully created and copied to artifacts.")

if __name__ == "__main__":
    main()
