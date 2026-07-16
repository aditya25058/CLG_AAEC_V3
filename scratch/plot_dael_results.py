#!/usr/bin/env python3
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import shutil

def main():
    results_path = "outputs/phase4/dael_results.json"
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found. Run scratch/run_dael_sweep.py first.")
        return
        
    with open(results_path, "r") as f:
        data = json.load(f)
        
    os.makedirs("outputs/phase4/dael_steering", exist_ok=True)
    
    # Artifact directory path from metadata
    artifact_dir = "/home/palakm/.gemini/antigravity-ide/brain/ed3c2dc2-4e67-4959-b431-347772d4a219/dael_steering"
    os.makedirs(artifact_dir, exist_ok=True)
    
    # Color palette
    colors = {
        "baseline": "#EF4444",   # Red
        "epeg": "#10B981",       # Emerald Green
        "dael": "#8B5CF6",       # Purple (DAEL)
        "epeg_dael": "#3B82F6"   # Blue (Hybrid)
    }
    
    variants = ["baseline", "epeg", "dael", "epeg_dael"]
    variant_labels = ["Baseline (Uniform BF16)", "EPEG Only", "DAEL Only", "EPEG + P-DAEL"]
    
    # Use standard DejaVu Sans
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['font.family'] = 'sans-serif'
    
    # ----------------------------------------------------
    # Plot 1: Serving Latency & Speedup (High Load, 2 GB/s vs 16 GB/s)
    # ----------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(2)  # 2 GB/s, 16 GB/s
    width = 0.18
    
    # Panel 1: Qwen3 Latency
    qwen_base = [data["qwen3"]["2GBs"]["high_load"]["baseline"]["total_latency_s"],
                 data["qwen3"]["16GBs"]["high_load"]["baseline"]["total_latency_s"]]
    qwen_epeg = [data["qwen3"]["2GBs"]["high_load"]["epeg"]["total_latency_s"],
                 data["qwen3"]["16GBs"]["high_load"]["epeg"]["total_latency_s"]]
    qwen_dael = [data["qwen3"]["2GBs"]["high_load"]["dael"]["total_latency_s"],
                 data["qwen3"]["16GBs"]["high_load"]["dael"]["total_latency_s"]]
    qwen_hybrid = [data["qwen3"]["2GBs"]["high_load"]["epeg_dael"]["total_latency_s"],
                   data["qwen3"]["16GBs"]["high_load"]["epeg_dael"]["total_latency_s"]]
    
    ax1.bar(x - 1.5*width, qwen_base, width, label=variant_labels[0], color=colors["baseline"], edgecolor="#B91C1C", zorder=3)
    ax1.bar(x - 0.5*width, qwen_epeg, width, label=variant_labels[1], color=colors["epeg"], edgecolor="#047857", zorder=3)
    ax1.bar(x + 0.5*width, qwen_dael, width, label=variant_labels[2], color=colors["dael"], edgecolor="#6D28D9", zorder=3)
    ax1.bar(x + 1.5*width, qwen_hybrid, width, label=variant_labels[3], color=colors["epeg_dael"], edgecolor="#1D4ED8", zorder=3)
    
    ax1.set_title("Qwen3-235B: Serving Latency (High Load)", fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["2 GB/s (Constrained)", "16 GB/s (Standard PCIe)"], fontsize=11)
    ax1.set_ylabel("Total Latency (Seconds)", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax1.legend(fontsize=10)
    
    # Panel 2: DeepSeek Latency
    ds_base = [data["deepseek"]["2GBs"]["high_load"]["baseline"]["total_latency_s"],
               data["deepseek"]["16GBs"]["high_load"]["baseline"]["total_latency_s"]]
    ds_epeg = [data["deepseek"]["2GBs"]["high_load"]["epeg"]["total_latency_s"],
               data["deepseek"]["16GBs"]["high_load"]["epeg"]["total_latency_s"]]
    ds_dael = [data["deepseek"]["2GBs"]["high_load"]["dael"]["total_latency_s"],
               data["deepseek"]["16GBs"]["high_load"]["dael"]["total_latency_s"]]
    ds_hybrid = [data["deepseek"]["2GBs"]["high_load"]["epeg_dael"]["total_latency_s"],
                 data["deepseek"]["16GBs"]["high_load"]["epeg_dael"]["total_latency_s"]]
    
    ax2.bar(x - 1.5*width, ds_base, width, label=variant_labels[0], color=colors["baseline"], edgecolor="#B91C1C", zorder=3)
    ax2.bar(x - 0.5*width, ds_epeg, width, label=variant_labels[1], color=colors["epeg"], edgecolor="#047857", zorder=3)
    ax2.bar(x + 0.5*width, ds_dael, width, label=variant_labels[2], color=colors["dael"], edgecolor="#6D28D9", zorder=3)
    ax2.bar(x + 1.5*width, ds_hybrid, width, label=variant_labels[3], color=colors["epeg_dael"], edgecolor="#1D4ED8", zorder=3)
    
    ax2.set_title("DeepSeek-R1: Serving Latency (High Load)", fontsize=13, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["2 GB/s (Constrained)", "16 GB/s (Standard PCIe)"], fontsize=11)
    ax2.set_ylabel("Total Latency (Seconds)", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax2.legend(fontsize=10)
    
    plt.suptitle("DAEL: Serving Latency Optimization Across Models", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    plot1_path = "outputs/phase4/dael_steering/dael_latency_comparison.png"
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    shutil.copy(plot1_path, os.path.join(artifact_dir, "dael_latency_comparison.png"))
    print(f"Saved latency comparison to {plot1_path}")
    
    # ----------------------------------------------------
    # Plot 2: Network Link Saturation & Expert Queue Load Balance (High Load)
    # ----------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    bws = [2.0, 16.0, 32.0]
    x_bws = np.arange(len(bws))
    
    # Panel 1: Switch Link Saturation CoV (Qwen3)
    qwen_base_cov = [data["qwen3"][f"{int(bw)}GBs"]["high_load"]["baseline"]["avg_link_saturation_cov"] for bw in bws]
    qwen_dael_cov = [data["qwen3"][f"{int(bw)}GBs"]["high_load"]["dael"]["avg_link_saturation_cov"] for bw in bws]
    qwen_hybrid_cov = [data["qwen3"][f"{int(bw)}GBs"]["high_load"]["epeg_dael"]["avg_link_saturation_cov"] for bw in bws]
    
    ax1.plot(x_bws, qwen_base_cov, marker='o', linewidth=2, label="Baseline", color=colors["baseline"])
    ax1.plot(x_bws, qwen_dael_cov, marker='s', linewidth=2, label="DAEL Only", color=colors["dael"])
    ax1.plot(x_bws, qwen_hybrid_cov, marker='^', linewidth=2, label="EPEG + P-DAEL", color=colors["epeg_dael"])
    
    ax1.set_title("Qwen3-235B: Switch Link Saturation CoV", fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(x_bws)
    ax1.set_xticklabels([f"{int(bw)} GB/s" for bw in bws], fontsize=11)
    ax1.set_ylabel("Coefficient of Variation (CoV - Lower is Better)", fontsize=12)
    ax1.set_xlabel("Interconnect Link Bandwidth", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=10)
    
    # Panel 2: Expert Queue Load Balance (DeepSeek Max/Mean Ratio)
    ds_base_mtm = [data["deepseek"][f"{int(bw)}GBs"]["high_load"]["baseline"]["avg_expert_queue_max_to_mean"] for bw in bws]
    ds_dael_mtm = [data["deepseek"][f"{int(bw)}GBs"]["high_load"]["dael"]["avg_expert_queue_max_to_mean"] for bw in bws]
    ds_hybrid_mtm = [data["deepseek"][f"{int(bw)}GBs"]["high_load"]["epeg_dael"]["avg_expert_queue_max_to_mean"] for bw in bws]
    
    ax2.plot(x_bws, ds_base_mtm, marker='o', linewidth=2, label="Baseline", color=colors["baseline"])
    ax2.plot(x_bws, ds_dael_mtm, marker='s', linewidth=2, label="DAEL Only", color=colors["dael"])
    ax2.plot(x_bws, ds_hybrid_mtm, marker='^', linewidth=2, label="EPEG + P-DAEL", color=colors["epeg_dael"])
    
    # Target threshold at 1.2
    ax2.axhline(y=1.20, color="#475569", linestyle="--", linewidth=1.5, label="Optimized Target (<1.2)", zorder=2)
    
    ax2.set_title("DeepSeek-R1: Expert Queue Load Balance (Max/Mean)", fontsize=13, fontweight='bold', pad=15)
    ax2.set_xticks(x_bws)
    ax2.set_xticklabels([f"{int(bw)} GB/s" for bw in bws], fontsize=11)
    ax2.set_ylabel("Max-to-Mean Ratio (Lower is Better)", fontsize=12)
    ax2.set_xlabel("Interconnect Link Bandwidth", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=10)
    
    plt.suptitle("DAEL: Link & Queue Load Balance Sensitivity Study", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    plot2_path = "outputs/phase4/dael_steering/dael_load_balance.png"
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    shutil.copy(plot2_path, os.path.join(artifact_dir, "dael_load_balance.png"))
    print(f"Saved load balance comparison to {plot2_path}")
    
    # ----------------------------------------------------
    # Plot 3: Routing Overhead vs Serving Latency Savings
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models_list = ["Qwen3-235B", "DeepSeek-R1"]
    x_pos = np.arange(len(models_list))
    width = 0.35
    
    # Gains in latency (Seconds)
    qwen_gain = data["qwen3"]["2GBs"]["high_load"]["baseline"]["total_latency_s"] - data["qwen3"]["2GBs"]["high_load"]["dael"]["total_latency_s"]
    ds_gain = data["deepseek"]["2GBs"]["high_load"]["baseline"]["total_latency_s"] - data["deepseek"]["2GBs"]["high_load"]["dael"]["total_latency_s"]
    
    # Overhead in latency (Converted from ms to seconds)
    qwen_overhead = data["qwen3"]["2GBs"]["high_load"]["dael"]["routing_overhead_ms"] / 1000.0
    ds_overhead = data["deepseek"]["2GBs"]["high_load"]["dael"]["routing_overhead_ms"] / 1000.0
    
    gains = [qwen_gain, ds_gain]
    overheads = [qwen_overhead, ds_overhead]
    
    # We plot both bars side by side on a log scale (or with labels) to highlight the massive difference
    # Since overhead is so small, we'll plot it on a logarithmic scale or just print the text labels.
    rects1 = ax.bar(x_pos - width/2, gains, width, label="Serving Latency Reduction", color="#10B981", edgecolor="#047857")
    rects2 = ax.bar(x_pos + width/2, overheads, width, label="Routing Decision Overhead", color="#EF4444", edgecolor="#B91C1C")
    
    ax.set_title("Routing Overhead vs. Serving Latency Savings (at 2 GB/s Link)", fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models_list, fontsize=12)
    ax.set_ylabel("Latency Delta (Seconds - log scale)", fontsize=12)
    ax.set_yscale('log')
    ax.grid(True, linestyle="--", alpha=0.5, which="both")
    ax.legend(fontsize=10)
    
    # Add values on top of bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.3f}s',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
                    
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h*1000.0:.2f}ms',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
                    
    plt.tight_layout()
    plot3_path = "outputs/phase4/dael_steering/dael_routing_overhead.png"
    plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    shutil.copy(plot3_path, os.path.join(artifact_dir, "dael_routing_overhead.png"))
    print(f"Saved overhead analysis to {plot3_path}")
    print(f"All DAEL plots generated successfully in {artifact_dir}!")

if __name__ == "__main__":
    main()
