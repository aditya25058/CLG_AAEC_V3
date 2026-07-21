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
    systems = ["demand", "expert_cache", "colossus_column_cache"]
    labels = {
        "demand": "Demand-Only",
        "expert_cache": "Expert-Level Cache",
        "colossus_column_cache": "COLOSSUS Column Cache"
    }
    colors = {
        "demand": "#D32F2F",          # red
        "expert_cache": "#FBC02D",    # yellow/gold
        "colossus_column_cache": "#1976D2" # blue
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
        
        policies = ["reactive", "markov", "temporal", "oracle"]
        colors = {
            "reactive": "#1976D2",
            "markov": "#FF9800",
            "temporal": "#4CAF50",
            "oracle": "#9C27B0"
        }
        markers = {
            "reactive": "o",
            "markov": "s",
            "temporal": "^",
            "oracle": "d"
        }
        labels = {
            "reactive": "Reactive",
            "markov": "Predictive (Markov)",
            "temporal": "Temporal Locality",
            "oracle": "Router Oracle"
        }
        
        for p in policies:
            net_gb = [data[p][str(s)]["network_gb"] for s in scales]
            tps = [data[p][str(s)]["throughput"] for s in scales]
            
            ax1.plot(scales, net_gb, marker=markers[p], label=labels[p], color=colors[p], linewidth=1.5)
            ax2.plot(scales, tps, marker=markers[p], label=labels[p], color=colors[p], linewidth=1.5)
            
        ax1.set_xlabel("Cluster Scale (Nodes)")
        ax1.set_ylabel("Network Traffic Transferred (GB)")
        ax1.set_title("Volume of Weight Data Transferred")
        ax1.set_xticks(scales)
        ax1.legend()
        
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

def plot_e17_amortized():
    print("Plotting E17 Amortized System Cost...")
    models = ["qwen3_30b", "deepseek_v2_lite"]
    
    for m in models:
        path = os.path.join(RESULTS_DIR, f"e17_amortized/{m}/amortized_kernel_results.json")
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            data = json.load(f)
            
        batches = sorted([int(k) for k in data.keys()])
        
        # We will plot the 16.0 GB/s link speed comparisons (PCIe Gen4)
        dense_tot = []
        sa_tot = []
        
        for B in batches:
            # find the 16.0 gbps entry
            for entry in data[str(B)]:
                if entry["link_speed_gbps"] == 16.0:
                    dense_tot.append(entry["total_dense_system_ms"])
                    sa_tot.append(entry["total_sa_system_ms"])
                    
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(batches))
        width = 0.35
        
        ax.bar(x - width/2, dense_tot, width, label="Dense FFN + Full Expert Load (Baseline)", color="#D32F2F")
        ax.bar(x + width/2, sa_tot, width, label="SA-FFN + Column Miss Load (COLOSSUS)", color="#1976D2")
        
        ax.set_ylabel("Total System Time per Layer (ms)")
        ax.set_xlabel("Batch Size")
        ax.set_title(f"Amortized Layer Overhead (PCIe Gen4) — {m.upper()}")
        ax.set_xticks(x)
        ax.set_xticklabels(batches)
        ax.legend()
        
        os.makedirs(os.path.join(PLOTS_DIR, "e17_amortized"), exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"e17_amortized/{m}_amortized_cost.png"))
        plt.close()

def plot_e18_sweep():
    print("Plotting E18 Predictor Sweep Sensitivity...")
    models = ["qwen3_30b", "deepseek_v2_lite"]
    
    for m in models:
        path = os.path.join(RESULTS_DIR, f"e18_sweep/{m}/predictor_sweep_results.json")
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            data = json.load(f)
            
        accuracies = [entry["target_accuracy"] * 100.0 for entry in data["column_level"]]
        col_stalls = [entry["avg_stall_ms"] for entry in data["column_level"]]
        exp_stalls = [entry["avg_stall_ms"] for entry in data["expert_level"]]
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(accuracies, col_stalls, marker="o", label="COLOSSUS Column-Level Cache", color="#1976D2", linewidth=1.5)
        ax.plot(accuracies, exp_stalls, marker="s", label="Expert-Level Cache", color="#D32F2F", linewidth=1.5)
        
        ax.set_xlabel("Predictor Accuracy (%)")
        ax.set_ylabel("Average Exposed GPU Stall (ms/token)")
        ax.set_title(f"Sensitivity to Prediction Quality ({m.upper()})")
        ax.set_xticks(accuracies)
        ax.legend()
        
        os.makedirs(os.path.join(PLOTS_DIR, "e18_sweep"), exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"e18_sweep/{m}_predictor_sensitivity.png"))
        plt.close()

def plot_e19_baselines():
    print("Plotting E19 External SOTA Baselines & Energy Efficiency...")
    models = ["qwen3_30b", "deepseek_v2_lite"]
    
    for m in models:
        path = os.path.join(RESULTS_DIR, f"e19_baselines/{m}/sota_comparison_results.json")
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            data = json.load(f)
            
        systems = list(data.keys())
        throughputs = [data[s]["throughput"] for s in systems]
        energies = [data[s]["joules_per_token"] for s in systems]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        
        colors = ["#757575", "#BDBDBD", "#9E9E9E", "#FF9800", "#4CAF50", "#9C27B0", "#1976D2"]
        x = np.arange(len(systems))
        
        bars1 = ax1.bar(x, throughputs, color=colors[:len(systems)])
        for bar in bars1:
            h = bar.get_height()
            ax1.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
                         
        ax1.set_ylabel("Serving Throughput (tokens/sec)")
        ax1.set_title(f"Throughput Comparison ({m.upper()})")
        ax1.set_xticks(x)
        ax1.set_xticklabels([s.upper().replace("_", "\n") for s in systems], fontsize=8)
        
        bars2 = ax2.bar(x, energies, color=colors[:len(systems)])
        for bar in bars2:
            h = bar.get_height()
            ax2.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
                         
        ax2.set_ylabel("Energy per Generated Token (Joules/Token)")
        ax2.set_title(f"Energy per Token Comparison ({m.upper()})")
        ax2.set_xticks(x)
        ax2.set_xticklabels([s.upper().replace("_", "\n") for s in systems], fontsize=8)
        
        fig.suptitle(f"SOTA Framework Serving & Energy Efficiency Comparison — {m.upper()}", fontsize=13, fontweight='bold')
        os.makedirs(os.path.join(PLOTS_DIR, "e19_baselines"), exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"e19_baselines/{m}_sota_comparison.png"))
        plt.close()

def plot_e20_honest_overlap():
    print("Plotting E20 Honest Overlap Waterfall...")
    path = os.path.join(RESULTS_DIR, "e20_overlap/honest_overlap_results.json")
    if not os.path.exists(path):
        return
    with open(path, "r") as f:
        data = json.load(f)
        
    # We will plot for Batch Size 1 across miss sizes and links
    links = list(data.keys())
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    for name in links:
        miss_cols = [entry["miss_cols"] for entry in data[name]["1"]]
        exposed_stalls = [entry["exposed_stall_ms"] for entry in data[name]["1"]]
        ax.plot(miss_cols, exposed_stalls, marker="o", label=name.replace("_", " "), linewidth=1.5)
        
    ax.set_xlabel("Active Miss Columns per Expert")
    ax.set_ylabel("Exposed Interconnect Stall (ms)")
    ax.set_title("Exposed Stall vs. Miss Payload & Link Speed (Batch=1)")
    ax.legend()
    
    os.makedirs(os.path.join(PLOTS_DIR, "e20_overlap"), exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "e20_overlap/honest_overlap_stall.png"))
    plt.close()

def main():
    plot_e03_energy()
    plot_e04_cache()
    plot_e10_ablation()
    plot_e12_scalability()
    plot_e13_distributed()
    plot_e14_prefetcher()
    plot_e17_amortized()
    plot_e18_sweep()
    plot_e19_baselines()
    plot_e20_honest_overlap()
    print("All plots generated successfully!")

if __name__ == "__main__":
    main()
