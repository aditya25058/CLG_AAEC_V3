#!/usr/bin/env python3
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import shutil

def main():
    results_path = "outputs/phase3/advanced_epeg_results.json"
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found. Run scratch/run_advanced_epeg_sweep.py first.")
        return
        
    with open(results_path, "r") as f:
        data = json.load(f)
        
    os.makedirs("outputs/phase3/advanced_epeg", exist_ok=True)
    
    # Artifact directory path from metadata
    artifact_dir = "/home/palakm/.gemini/antigravity-ide/brain/ed3c2dc2-4e67-4959-b431-347772d4a219/advanced_epeg"
    os.makedirs(artifact_dir, exist_ok=True)
    
    # Ordered variants
    variants = [
        "baseline",
        "static_epeg",
        "epeg_sla",
        "epeg_sla_caps",
        "epeg_sla_slice",
        "full_epeg"
    ]
    
    labels = [
        "Baseline (BF16)",
        "Static EPEG",
        "EPEG-SLA",
        "EPEG-SLA + CAPS",
        "EPEG-SLA + Slice",
        "Full Co-Designed EPEG"
    ]
    
    colors = [
        "#EF4444",  # Red
        "#F59E0B",  # Amber
        "#10B981",  # Emerald
        "#8B5CF6",  # Purple
        "#EC4899",  # Pink
        "#3B82F6"   # Blue
    ]
    
    # Retrieve metrics
    latencies = [data[v]["total_latency_s"] for v in variants]
    base_latency = data["baseline"]["total_latency_s"]
    speedups = [base_latency / lat for lat in latencies]
    
    q_delays = [data[v]["mean_q_delay_ms"] for v in variants]
    ttfts = [data[v]["avg_ttft_ms"] for v in variants]
    tpots = [data[v]["avg_tpot_ms"] for v in variants]
    
    # Set plot styles
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['font.family'] = 'sans-serif'
    
    # ----------------------------------------------------
    # Plot 1: Serving Latency & Speedup
    # ----------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(variants))
    
    # Left: Total Latency
    bars1 = ax1.bar(x, latencies, color=colors, edgecolor="#1E293B", zorder=3, width=0.6)
    ax1.set_title("Total Serving Latency", fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax1.set_ylabel("Latency (Seconds)", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.5, zorder=0)
    
    # Add values on top of bars
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}s',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
                    
    # Right: Speedup Ratio
    bars2 = ax2.bar(x, speedups, color=colors, edgecolor="#1E293B", zorder=3, width=0.6)
    ax2.set_title("Serving Speedup Ratio", fontsize=13, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax2.set_ylabel("Speedup (x)", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.5, zorder=0)
    
    # Add values on top of bars
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}x',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
                    
    plt.suptitle("Advanced EPEG: Co-Design Gating, Scheduling (CAPS) & DRAM Slicing (EPEG-Slice)", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    plot1_path = "outputs/phase3/advanced_epeg/advanced_epeg_latency_speedup.png"
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Copy to artifacts
    shutil.copy(plot1_path, os.path.join(artifact_dir, "advanced_epeg_latency_speedup.png"))
    print(f"Saved serving latency comparison to {plot1_path}")
    
    # ----------------------------------------------------
    # Plot 2: Mean Queuing Delay & Mean TTFT
    # ----------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Mean Queuing Delay
    bars1 = ax1.bar(x, q_delays, color=colors, edgecolor="#1E293B", zorder=3, width=0.6)
    ax1.set_title("Mean Queuing Delay", fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax1.set_ylabel("Delay (Milliseconds)", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.5, zorder=0)
    
    # Draw SLA queue target line
    ax1.axhline(y=50.0, color="#475569", linestyle="--", linewidth=1.5, label="SLA Queue Limit (50ms)", zorder=4)
    ax1.legend(fontsize=10)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}ms',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
                    
    # Right: Mean TTFT
    bars2 = ax2.bar(x, ttfts, color=colors, edgecolor="#1E293B", zorder=3, width=0.6)
    ax2.set_title("Mean Time-to-First-Token (TTFT)", fontsize=13, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax2.set_ylabel("TTFT (Milliseconds)", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.5, zorder=0)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}ms',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
                    
    plt.suptitle("Advanced EPEG: Queue Congestion & TTFT Co-Design Benefits", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    plot2_path = "outputs/phase3/advanced_epeg/advanced_epeg_queue_ttft.png"
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Copy to artifacts
    shutil.copy(plot2_path, os.path.join(artifact_dir, "advanced_epeg_queue_ttft.png"))
    print(f"Saved queue and TTFT comparison to {plot2_path}")
    print(f"Successfully generated all advanced EPEG plots in outputs/phase3/advanced_epeg and copied to {artifact_dir}")

if __name__ == "__main__":
    main()
