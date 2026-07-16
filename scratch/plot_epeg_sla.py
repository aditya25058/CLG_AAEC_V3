#!/usr/bin/env python3
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import shutil

def main():
    results_path = "outputs/phase3/epeg_sla_results.json"
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found. Run scratch/run_epeg_sla_sweep.py first.")
        return
        
    with open(results_path, "r") as f:
        data = json.load(f)
        
    os.makedirs("outputs/phase3/sla_gating", exist_ok=True)
    
    # Artifact directory path from metadata
    artifact_dir = "/home/palakm/.gemini/antigravity-ide/brain/ed3c2dc2-4e67-4959-b431-347772d4a219/sla_gating"
    os.makedirs(artifact_dir, exist_ok=True)
    
    # Colors
    colors = {
        "baseline": "#EF4444",    # Coral/Red
        "static_epeg": "#10B981", # Emerald
        "epeg_sla": "#3B82F6"     # Blue (SLA)
    }
    
    loads = ["low_load", "high_load"]
    variants = ["baseline", "static_epeg", "epeg_sla"]
    variant_labels = ["Baseline (Uniform BF16)", "Static EPEG (0.40, 0.05)", "EPEG-SLA (Closed-Loop)"]
    
    # ----------------------------------------------------
    # Plot 1: Latency & Queuing Delay Comparison (Double Panel)
    # ----------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['font.family'] = 'sans-serif'
    
    x = np.arange(len(loads))
    width = 0.25
    
    # Panel 1: Total Serving Latency (s)
    base_lats = [data[load]["baseline"]["total_latency_s"] for load in loads]
    static_lats = [data[load]["static_epeg"]["total_latency_s"] for load in loads]
    sla_lats = [data[load]["epeg_sla"]["total_latency_s"] for load in loads]
    
    ax1.bar(x - width, base_lats, width, label=variant_labels[0], color=colors["baseline"], edgecolor="#B91C1C", zorder=3)
    ax1.bar(x, static_lats, width, label=variant_labels[1], color=colors["static_epeg"], edgecolor="#047857", zorder=3)
    ax1.bar(x + width, sla_lats, width, label=variant_labels[2], color=colors["epeg_sla"], edgecolor="#1D4ED8", zorder=3)
    
    ax1.set_title("Total Serving Latency (Seconds)", fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Low Load (10 reqs staggered)", "High Load (50 reqs concurrent)"], fontsize=11)
    ax1.set_ylabel("Latency (s)", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax1.legend(fontsize=10)
    
    # Panel 2: Mean Queuing Delay (ms)
    base_q = [data[load]["baseline"]["mean_q_delay_ms"] for load in loads]
    static_q = [data[load]["static_epeg"]["mean_q_delay_ms"] for load in loads]
    sla_q = [data[load]["epeg_sla"]["mean_q_delay_ms"] for load in loads]
    
    ax2.bar(x - width, base_q, width, label=variant_labels[0], color=colors["baseline"], edgecolor="#B91C1C", zorder=3)
    ax2.bar(x, static_q, width, label=variant_labels[1], color=colors["static_epeg"], edgecolor="#047857", zorder=3)
    ax2.bar(x + width, sla_q, width, label=variant_labels[2], color=colors["epeg_sla"], edgecolor="#1D4ED8", zorder=3)
    
    # Draw SLA target line at 50ms
    ax2.axhline(y=50.0, color="#475569", linestyle="--", linewidth=1.5, label="SLA Queue Limit (50ms)", zorder=4)
    
    ax2.set_title("Mean Queuing Delay (Milliseconds)", fontsize=13, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Low Load (10 reqs staggered)", "High Load (50 reqs concurrent)"], fontsize=11)
    ax2.set_ylabel("Queuing Delay (ms)", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax2.legend(fontsize=10)
    
    plt.suptitle("EPEG-SLA: Queuing Delay & Serving Latency Co-Design Evaluation", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    plot1_path = "outputs/phase3/sla_gating/epeg_sla_latency_comparison.png"
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Copy to artifacts
    shutil.copy(plot1_path, os.path.join(artifact_dir, "epeg_sla_latency_comparison.png"))
    print(f"Saved serving latency comparison to {plot1_path}")
    
    # ----------------------------------------------------
    # Plot 2: Projected Downstream Accuracy Penalty Breakdown
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Metrics
    metrics_list = ["GSM8K Loss", "MMLU Loss", "LCB Loss"]
    
    # Extract projected losses
    low_static = [data["low_load"]["static_epeg"]["projected_gsm8k_loss_pct"], 
                  data["low_load"]["static_epeg"]["projected_mmlu_loss_pct"], 
                  data["low_load"]["static_epeg"]["projected_lcb_loss_pct"]]
    low_sla = [data["low_load"]["epeg_sla"]["projected_gsm8k_loss_pct"], 
                data["low_load"]["epeg_sla"]["projected_mmlu_loss_pct"], 
                data["low_load"]["epeg_sla"]["projected_lcb_loss_pct"]]
                
    high_static = [data["high_load"]["static_epeg"]["projected_gsm8k_loss_pct"], 
                   data["high_load"]["static_epeg"]["projected_mmlu_loss_pct"], 
                   data["high_load"]["static_epeg"]["projected_lcb_loss_pct"]]
    high_sla = [data["high_load"]["epeg_sla"]["projected_gsm8k_loss_pct"], 
                 data["high_load"]["epeg_sla"]["projected_mmlu_loss_pct"], 
                 data["high_load"]["epeg_sla"]["projected_lcb_loss_pct"]]
    
    # Set up positions
    y = np.arange(len(metrics_list))
    height = 0.2
    
    # Horizontal bars for accuracy loss
    # Low Load
    ax.barh(y + 1.5*height, low_static, height, label="Static EPEG (Low Load)", color=colors["static_epeg"], alpha=0.4, edgecolor="#047857", hatch="//", zorder=3)
    ax.barh(y + 0.5*height, low_sla, height, label="EPEG-SLA (Low Load)", color=colors["epeg_sla"], alpha=0.4, edgecolor="#1D4ED8", zorder=3)
    
    # High Load
    ax.barh(y - 0.5*height, high_static, height, label="Static EPEG (High Load)", color=colors["static_epeg"], edgecolor="#047857", hatch="\\\\", zorder=3)
    ax.barh(y - 1.5*height, high_sla, height, label="EPEG-SLA (High Load)", color=colors["epeg_sla"], edgecolor="#1D4ED8", zorder=3)
    
    ax.set_title("Projected Downstream Task Accuracy Penalty Breakdown (%)", fontsize=13, fontweight='bold', pad=15)
    ax.set_yticks(y)
    ax.set_yticklabels(metrics_list, fontsize=11)
    ax.set_xlabel("Accuracy Penalty (% Loss - Lower is Better)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.legend(fontsize=10, loc="lower right")
    
    # Invert y-axis to have GSM8K at the top
    ax.invert_yaxis()
    
    plt.tight_layout()
    plot2_path = "outputs/phase3/sla_gating/epeg_sla_accuracy_penalty.png"
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Copy to artifacts
    shutil.copy(plot2_path, os.path.join(artifact_dir, "epeg_sla_accuracy_penalty.png"))
    print(f"Saved accuracy penalty breakdown to {plot2_path}")
    print(f"Successfully generated all SLA plots in outputs/phase3/sla_gating and copied to {artifact_dir}")

if __name__ == "__main__":
    main()
