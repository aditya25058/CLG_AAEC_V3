# evaluation/scripts/generate_all_plots.py
import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Configure matplotlib styles for publication-ready outputs
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 200,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

RESULTS_DIR = "/home/palakm/MoEServingSim/evaluation/results"
PLOTS_DIR = "/home/palakm/MoEServingSim/evaluation/plots"

def plot_e03_energy():
    print("Plotting E03 Energy Concentration...")
    fig, ax = plt.subplots(figsize=(6, 4))
    
    thresholds = [50, 70, 80, 90, 95, 99]
    
    for model_name, label in [("qwen3_30b", "Qwen3-30B-A3B (FFN=768)"), ("deepseek_v2_lite", "DeepSeek-V2-Lite (FFN=1408)")]:
        path = os.path.join(RESULTS_DIR, f"e03_energy/{model_name}/energy_concentration.json")
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            data = json.load(f)
            
        fractions = [data[f"k_{t}"]["fraction_pct"] for t in thresholds]
        ax.plot(thresholds, fractions, marker="o", label=label, linewidth=1.5)
        
    ax.set_xlabel("Target Cumulative Activation Energy (%)")
    ax.set_ylabel("Columns Required (% of FFN Dim)")
    ax.set_title("Neuron Activation Energy Concentration")
    ax.set_xticks(thresholds)
    ax.set_ylim(0, 100)
    ax.legend()
    
    os.makedirs(os.path.join(PLOTS_DIR, "e03_energy"), exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "e03_energy/energy_cdf_curve.png"))
    plt.close()

def plot_e04_cache():
    print("Plotting E04 Cache Hit Rates...")
    for model_name in ["qwen3_30b", "deepseek_v2_lite"]:
        path = os.path.join(RESULTS_DIR, f"e04_cache/{model_name}/cache_comparison.json")
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            data = json.load(f)
            
        fig, ax = plt.subplots(figsize=(6, 4))
        sizes = sorted([int(k) for k in data["lru"].keys()])
        
        for policy in data.keys():
            hr = [data[policy][str(s)] * 100.0 for s in sizes]
            ax.plot(sizes, hr, marker="o", label=policy.upper(), linewidth=1.5)
            
        ax.set_xlabel("Cache Size (columns per expert)")
        ax.set_ylabel("Cache Hit Rate (%)")
        ax.set_title(f"Cache Hit Rate Sweep ({model_name})")
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_ylim(0, 100)
        ax.legend()
        
        os.makedirs(os.path.join(PLOTS_DIR, "e04_cache"), exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"e04_cache/{model_name}_hit_rates.png"))
        plt.close()

def plot_e10_ablation():
    print("Plotting E10 Ablation Bar Chart...")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    models = ["qwen3_30b", "deepseek_v2_lite"]
    ablations = ["full", "no_prefetcher", "no_column_level"]
    x = np.arange(len(models))
    width = 0.25
    
    for idx, ab in enumerate(ablations):
        rates = []
        for m in models:
            path = os.path.join(RESULTS_DIR, f"e10_ablation/{m}/ablation_results.json")
            if os.path.exists(path):
                with open(path, "r") as f:
                    rates.append(json.load(f)[ab]["hit_rate"] * 100.0)
            else:
                rates.append(0.0)
                
        rects = ax.bar(x + idx * width - width/2, rates, width, label=ab.upper().replace("_", " "))
        
    ax.set_ylabel("Cache Hit Rate (%)")
    ax.set_title("Ablation Study: Hit Rate under Core Component Removal")
    ax.set_xticks(x + width/2)
    ax.set_xticklabels([m.upper().replace("_", " ") for m in models])
    ax.set_ylim(0, 100)
    ax.legend()
    
    os.makedirs(os.path.join(PLOTS_DIR, "e10_ablation"), exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "e10_ablation/ablation_bar_chart.png"))
    plt.close()

def plot_e12_scalability():
    print("Plotting E12 Scalability Sweeps...")
    for model_name in ["qwen3_30b", "deepseek_v2_lite"]:
        path = os.path.join(RESULTS_DIR, f"e12_scalability/{model_name}/scaling_results.json")
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            data = json.load(f)
            
        fig, ax = plt.subplots(figsize=(6, 4))
        sizes = sorted([int(k) for k in data["pcie_gen5"].keys()])
        
        for conn in ["pcie_gen5", "nvlink"]:
            stalls = [data[conn][str(s)]["avg_stall_ms"] for s in sizes]
            ax.plot(sizes, stalls, marker="o", label=conn.upper().replace("_", " "), linewidth=1.5)
            
        ax.set_xlabel("Cache Size (columns per expert)")
        ax.set_ylabel("Average Exposed GPU Stall (ms/token)")
        ax.set_title(f"Stall Latency Scaling Curve ({model_name})")
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.legend()
        
        os.makedirs(os.path.join(PLOTS_DIR, "e12_scalability"), exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"e12_scalability/{model_name}_stalls.png"))
        plt.close()

def plot_e13_distributed():
    print("Plotting E13 Distributed Serving Results...")
    models = ["qwen3_30b", "deepseek_v2_lite"]
    systems = ["demand", "expert_cache", "aaec_column_cache"]
    labels = {
        "demand": "Demand-Only",
        "expert_cache": "Expert-Level Cache",
        "aaec_column_cache": "AAEC Column Cache"
    }
    colors = {
        "demand": "#D32F2F",          # red
        "expert_cache": "#FBC02D",    # yellow/gold
        "aaec_column_cache": "#1976D2" # blue
    }
    
    for m in models:
        path = os.path.join(RESULTS_DIR, f"e13_distributed/{m}/distributed_results.json")
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            data = json.load(f)
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        sizes = sorted([int(k) for k in data["demand"].keys()])
        
        for sys in systems:
            net_gb = [data[sys][str(s)]["network_gb"] for s in sizes]
            stall = [data[sys][str(s)]["avg_stall_ms"] for s in sizes]
            
            ax1.plot(sizes, net_gb, marker="o", label=labels[sys], color=colors[sys], linewidth=1.5)
            ax2.plot(sizes, stall, marker="o", label=labels[sys], color=colors[sys], linewidth=1.5)
            
        ax1.set_xlabel("Equivalent Cache Size (cols/expert)")
        ax1.set_ylabel("Network Traffic Transferred (GB)")
        ax1.set_title("Volume of Weight Data Transferred")
        ax1.set_xscale("log", base=2)
        ax1.set_xticks(sizes)
        ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax1.legend()
        
        ax2.set_xlabel("Equivalent Cache Size (cols/expert)")
        ax2.set_ylabel("Average Cross-Node Stall (ms/token)")
        ax2.set_title("Cross-Node Interconnect Stall Time")
        ax2.set_xscale("log", base=2)
        ax2.set_xticks(sizes)
        ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax2.legend()
        
        fig.suptitle(f"Distributed Serving Tradeoffs — {m.upper().replace('_', ' ')}", fontsize=13, fontweight='bold')
        os.makedirs(os.path.join(PLOTS_DIR, "e13_distributed"), exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"e13_distributed/{m}_network_stats.png"))
        plt.close()

def plot_e14_prefetcher():
    print("Plotting E14 Distributed Prefetcher Results...")
    models = ["qwen3_30b", "deepseek_v2_lite"]
    
    for m in models:
        path = os.path.join(RESULTS_DIR, f"e14_prefetcher/{m}/prefetcher_results.json")
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            data = json.load(f)
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        scales = sorted([int(k) for k in data["reactive"].keys()])
        
        # Colors: Reactive = Blue, Predictive = Green
        react_color = "#1976D2"
        pred_color = "#2E7D32"
        
        react_gb = [data["reactive"][str(s)]["network_gb"] for s in scales]
        pred_gb = [data["predictive"][str(s)]["network_gb"] for s in scales]
        
        react_tps = [data["reactive"][str(s)]["throughput"] for s in scales]
        pred_tps = [data["predictive"][str(s)]["throughput"] for s in scales]
        
        # Plot Network Volume
        ax1.plot(scales, react_gb, marker="o", label="Reactive AAEC", color=react_color, linewidth=1.5)
        ax1.plot(scales, pred_gb, marker="s", label="Predictive AAEC", color=pred_color, linewidth=1.5)
        ax1.set_xlabel("Cluster Scale (Nodes)")
        ax1.set_ylabel("Network Traffic Transferred (GB)")
        ax1.set_title("Volume of Weight Data Transferred")
        ax1.set_xticks(scales)
        ax1.legend()
        
        # Plot Throughput
        ax2.plot(scales, react_tps, marker="o", label="Reactive AAEC", color=react_color, linewidth=1.5)
        ax2.plot(scales, pred_tps, marker="s", label="Predictive AAEC", color=pred_color, linewidth=1.5)
        ax2.set_xlabel("Cluster Scale (Nodes)")
        ax2.set_ylabel("Serving Throughput (tokens/sec)")
        ax2.set_title("Token Generation Throughput")
        ax2.set_xticks(scales)
        ax2.legend()
        
        fig.suptitle(f"Distributed Prefetcher Tradeoffs — {m.upper().replace('_', ' ')}", fontsize=13, fontweight='bold')
        os.makedirs(os.path.join(PLOTS_DIR, "e14_prefetcher"), exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"e14_prefetcher/{m}_prefetcher_tradeoffs.png"))
        plt.close()

def main():
    plot_e03_energy()
    plot_e04_cache()
    plot_e10_ablation()
    plot_e12_scalability()
    plot_e13_distributed()
    plot_e14_prefetcher()
    print("All plots generated successfully!")

if __name__ == "__main__":
    main()
