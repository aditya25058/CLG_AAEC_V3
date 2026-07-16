#!/usr/bin/env python3
import json
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np

def main():
    results_path = "outputs/phase3/epeg_tp_ep_sweep_results.json"
    if not os.path.exists(results_path):
        print(f"Results file {results_path} not found. Please run the sweep script first.")
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    os.makedirs("outputs/phase3", exist_ok=True)
    artifact_dir = "/home/palakm/.gemini/antigravity-ide/brain/ed3c2dc2-4e67-4959-b431-347772d4a219"

    # Set up matplotlib style for professional, clean aesthetics
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
    plt.rcParams['axes.edgecolor'] = '#CBD5E1'
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['xtick.color'] = '#475569'
    plt.rcParams['ytick.color'] = '#475569'

    colors = {
        "Baseline": "#64748B",       # Slate
        "Comm Only": "#F59E0B",      # Amber
        "Compute Only": "#10B981",   # Emerald
        "Full EPEG": "#3B82F6",      # Blue
        "Line1": "#2563EB",          # Royal Blue
        "Line2": "#DC2626",          # Crimson
        "Line3": "#059669"           # Forest Green
    }

    var_names = ["Baseline", "Comm Only", "Compute Only", "Full EPEG"]

    # ====================================================
    # 1. Ablation Breakdowns Plot
    # ====================================================
    print("Plotting Ablation Breakdowns...")
    ablation = data["ablation"]
    configs = list(ablation.keys()) # ['tp8_ep8', 'tp4_ep4', 'tp2_ep4_multi', 'tp4_ep8_multi']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, cfg in enumerate(configs):
        ax = axes[idx]
        metrics = ablation[cfg]
        latencies = [metrics[var]["total_latency_s"] for var in var_names]
        tpots = [metrics[var]["avg_tpot_ms"] for var in var_names]
        
        base_lat = metrics["Baseline"]["total_latency_s"]
        speedups = [base_lat / metrics[var]["total_latency_s"] for var in var_names]
        
        bars = ax.bar(var_names, latencies, color=[colors[var] for var in var_names], edgecolor="#1E293B", width=0.45, alpha=0.85)
        ax.set_title(f"Config: {cfg.upper().replace('_MULTI', ' (DP=2)')}", fontsize=12, fontweight='bold', pad=10)
        ax.set_ylabel("Serving Latency (s)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3)
        
        # Add labels on top of bars
        for bar, speedup in zip(bars, speedups):
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"{yval:.2f}s\n({speedup:.2f}x)", 
                    ha='center', va='bottom', fontsize=9, fontweight='semibold')
            
    plt.suptitle("EPEG Performance ablation Breakdown across Configurations", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig("outputs/phase3/epeg_ablation_breakdown.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ====================================================
    # 2. EP Scaling Efficiency Plot (Speedup vs. EP Size)
    # ====================================================
    print("Plotting EP Scaling Efficiency...")
    ep_tp = data["ep_tp_scaling"]
    
    # Extract fixed TP=8 EP scaling speedups
    ep_sizes_tp8 = [1, 2, 4, 8]
    speedups_tp8 = []
    for ep in ep_sizes_tp8:
        metrics = ep_tp[f"fixed_tp8_ep{ep}"]
        speedups_tp8.append(metrics["Baseline"]["total_latency_s"] / metrics["Full EPEG"]["total_latency_s"])
        
    # Extract fixed TP=4 EP scaling speedups
    ep_sizes_tp4 = [1, 2, 4]
    speedups_tp4 = []
    for ep in ep_sizes_tp4:
        key = f"fixed_tp4_ep{ep}" if ep > 1 else "fixed_ep1_tp4"
        metrics = ep_tp[key]
        speedups_tp4.append(metrics["Baseline"]["total_latency_s"] / metrics["Full EPEG"]["total_latency_s"])
        
    plt.figure(figsize=(8, 6))
    plt.plot(ep_sizes_tp8, speedups_tp8, marker='o', linewidth=2.5, color=colors["Line1"], label="Fixed TP=8")
    plt.plot(ep_sizes_tp4, speedups_tp4, marker='s', linewidth=2.5, linestyle="--", color=colors["Line2"], label="Fixed TP=4")
    plt.title("EPEG Speedup vs. Expert Parallelism (EP) Size", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Expert Parallelism Size (EP)", fontsize=11)
    plt.ylabel("Serving Speedup (x)", fontsize=11)
    plt.xticks([1, 2, 4, 8])
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(frameon=True, facecolor="#F8FAFC", edgecolor="#E2E8F0")
    
    # Annotate points
    for x, y in zip(ep_sizes_tp8, speedups_tp8):
        plt.annotate(f"{y:.2f}x", (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold', color=colors["Line1"])
    for x, y in zip(ep_sizes_tp4, speedups_tp4):
        plt.annotate(f"{y:.2f}x", (x, y), textcoords="offset points", xytext=(0,-15), ha='center', fontweight='bold', color=colors["Line2"])
        
    plt.tight_layout()
    plt.savefig("outputs/phase3/epeg_ep_scaling_efficiency.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ====================================================
    # 3. TP vs. EP Separation Plot
    # ====================================================
    print("Plotting TP vs. EP Scaling Separation...")
    # Fixed TP=8, scale EP vs Fixed EP=1, scale TP
    tp_sizes = [1, 2, 4, 8]
    ep_sizes = [1, 2, 4, 8]
    
    # Latencies
    lat_fixed_tp8_base = [ep_tp[f"fixed_tp8_ep{ep}"]["Baseline"]["total_latency_s"] for ep in ep_sizes]
    lat_fixed_tp8_epeg = [ep_tp[f"fixed_tp8_ep{ep}"]["Full EPEG"]["total_latency_s"] for ep in ep_sizes]
    
    lat_fixed_ep1_base = [ep_tp[f"fixed_ep1_tp{tp}"]["Baseline"]["total_latency_s"] for tp in tp_sizes]
    lat_fixed_ep1_epeg = [ep_tp[f"fixed_ep1_tp{tp}"]["Full EPEG"]["total_latency_s"] for tp in tp_sizes]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: EP scaling under fixed TP=8
    ax1.plot(ep_sizes, lat_fixed_tp8_base, marker='o', linewidth=2, color=colors["Baseline"], label="Baseline (BF16)")
    ax1.plot(ep_sizes, lat_fixed_tp8_epeg, marker='^', linewidth=2.5, color=colors["Full EPEG"], label="Full EPEG")
    ax1.set_title("Expert Parallelism Scaling (Fixed TP=8)", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("Expert Parallelism Size (EP)", fontsize=10)
    ax1.set_ylabel("Latency (s)", fontsize=10)
    ax1.set_xticks(ep_sizes)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend()
    
    # Right: TP scaling under fixed EP=1
    ax2.plot(tp_sizes, lat_fixed_ep1_base, marker='o', linewidth=2, color=colors["Baseline"], label="Baseline (BF16)")
    ax2.plot(tp_sizes, lat_fixed_ep1_epeg, marker='^', linewidth=2.5, color=colors["Full EPEG"], label="Full EPEG")
    ax2.set_title("Tensor Parallelism Scaling (Fixed EP=1)", fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("Tensor Parallelism Size (TP)", fontsize=10)
    ax2.set_ylabel("Latency (s)", fontsize=10)
    ax2.set_xticks(tp_sizes)
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend()
    
    plt.suptitle("Separated TP vs. EP Scaling Characteristics", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig("outputs/phase3/epeg_tp_vs_ep_scaling.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ====================================================
    # 4. Bandwidth Sensitivity Plot
    # ====================================================
    print("Plotting Bandwidth Sensitivity...")
    bw_sens = data["bw_sensitivity"]
    
    bw_vals = [0.25, 1.0, 4.0]
    bw_keys = ["bw_0_25", "bw_1_0", "bw_4_0"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel 1: TP=2, EP=4, DP=2 Sensitivity
    t2e4 = bw_sens["tp2_ep4_dp2"]
    for var, color, marker in zip(var_names[1:], [colors["Comm Only"], colors["Compute Only"], colors["Full EPEG"]], ['o', 's', '^']):
        speedups = [t2e4[bk]["Baseline"]["total_latency_s"] / t2e4[bk][var]["total_latency_s"] for bk in bw_keys]
        ax1.plot(bw_vals, speedups, marker=marker, linewidth=2.5, color=color, label=var)
    ax1.set_title("TP=2, EP=4 (DP=2) Sensitivity", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("Interconnect Bandwidth (GB/s)", fontsize=10)
    ax1.set_ylabel("EPEG Speedup Ratio (x)", fontsize=10)
    ax1.set_xscale('log')
    ax1.set_xticks(bw_vals)
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.grid(True, which="both", linestyle="--", alpha=0.3)
    ax1.legend()
    
    # Panel 2: TP=4, EP=8, DP=2 Sensitivity
    t4e8 = bw_sens["tp4_ep8_dp2"]
    for var, color, marker in zip(var_names[1:], [colors["Comm Only"], colors["Compute Only"], colors["Full EPEG"]], ['o', 's', '^']):
        speedups = [t4e8[bk]["Baseline"]["total_latency_s"] / t4e8[bk][var]["total_latency_s"] for bk in bw_keys]
        ax2.plot(bw_vals, speedups, marker=marker, linewidth=2.5, color=color, label=var)
    ax2.set_title("TP=4, EP=8 (DP=2) Sensitivity", fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("Interconnect Bandwidth (GB/s)", fontsize=10)
    ax2.set_ylabel("EPEG Speedup Ratio (x)", fontsize=10)
    ax2.set_xscale('log')
    ax2.set_xticks(bw_vals)
    ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax2.grid(True, which="both", linestyle="--", alpha=0.3)
    ax2.legend()
    
    plt.suptitle("EPEG Interconnect Bandwidth Sensitivity Analysis", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig("outputs/phase3/epeg_bandwidth_sensitivity.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ====================================================
    # 5. Top-k Scaling Study
    # ====================================================
    print("Plotting Top-k Scaling Study...")
    topk = data["topk"]
    k_vals = [2, 4, 8]
    speedups_k = []
    for k in k_vals:
        metrics = topk[f"k_{k}"]
        speedups_k.append(metrics["Baseline"]["total_latency_s"] / metrics["Full EPEG"]["total_latency_s"])
        
    plt.figure(figsize=(7, 5))
    plt.plot(k_vals, speedups_k, marker='o', linewidth=2.5, color=colors["Line3"])
    plt.title("EPEG Speedup vs. Routing Expert Target (k) (TP=8, EP=8)", fontsize=12, fontweight='bold', pad=10)
    plt.xlabel("Number of Top-k Experts (k)", fontsize=10)
    plt.ylabel("EPEG Speedup Ratio (x)", fontsize=10)
    plt.xticks(k_vals)
    plt.grid(True, linestyle="--", alpha=0.4)
    for x, y in zip(k_vals, speedups_k):
        plt.annotate(f"{y:.2f}x", (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold', color=colors["Line3"])
        
    plt.tight_layout()
    plt.savefig("outputs/phase3/epeg_topk_scaling_study.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ====================================================
    # 6. Concurrency Scaling Study
    # ====================================================
    print("Plotting Concurrency Scaling Study...")
    concur = data["concurrency"]
    reqs_vals = [10, 50, 200]
    latencies_base = [concur[f"reqs_{r}"]["Baseline"]["total_latency_s"] for r in reqs_vals]
    latencies_epeg = [concur[f"reqs_{r}"]["Full EPEG"]["total_latency_s"] for r in reqs_vals]
    
    speedups_c = [latencies_base[i] / latencies_epeg[i] for i in range(len(reqs_vals))]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: End-to-end Latency scaling
    x = np.arange(len(reqs_vals))
    width = 0.35
    ax1.bar(x - width/2, latencies_base, width, label='Baseline (BF16)', color=colors["Baseline"], edgecolor="#1E293B")
    ax1.bar(x + width/2, latencies_epeg, width, label='Full EPEG', color=colors["Full EPEG"], edgecolor="#1E293B")
    ax1.set_title("Serving Latency vs. Concurrency", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("Number of Concurrent Requests", fontsize=10)
    ax1.set_ylabel("Latency (s)", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(r) for r in reqs_vals])
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend()
    
    # Right: Speedup line
    ax2.plot(reqs_vals, speedups_c, marker='o', linewidth=2.5, color=colors["Line1"])
    ax2.set_title("EPEG Speedup vs. Request Concurrency Level", fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("Concurrent Requests", fontsize=10)
    ax2.set_ylabel("EPEG Speedup Ratio (x)", fontsize=10)
    ax2.set_xticks(reqs_vals)
    ax2.grid(True, linestyle="--", alpha=0.4)
    for rx, y in zip(reqs_vals, speedups_c):
        ax2.annotate(f"{y:.2f}x", (rx, y), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold', color=colors["Line1"])
        
    plt.suptitle("EPEG Scalability Study with Varying Concurrency (TP=8, EP=8)", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig("outputs/phase3/epeg_concurrency_scaling.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ====================================================
    # Copy all plots and JSON to artifact directory
    # ====================================================
    print("Copying plots and json to artifacts...")
    files_to_copy = [
        "epeg_tp_ep_sweep_results.json",
        "epeg_ablation_breakdown.png",
        "epeg_ep_scaling_efficiency.png",
        "epeg_tp_vs_ep_scaling.png",
        "epeg_bandwidth_sensitivity.png",
        "epeg_topk_scaling_study.png",
        "epeg_concurrency_scaling.png"
    ]
    for fn in files_to_copy:
        src = os.path.join("outputs/phase3", fn)
        dst = os.path.join(artifact_dir, fn)
        if os.path.exists(src):
            try:
                shutil.copy(src, dst)
                print(f"Copied {fn} to artifacts.")
            except Exception as e:
                print(f"Error copying {fn}: {e}")

    print("All plots generated and copied successfully!")

if __name__ == "__main__":
    main()
