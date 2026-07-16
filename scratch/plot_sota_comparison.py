#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    json_path = "outputs/phase3/epeg_sota_comparison.json"
    if not os.path.exists(json_path):
        print(f"Error: JSON file {json_path} not found!")
        return

    with open(json_path, "r") as f:
        data = json.load(f)
        
    models = ["Qwen3-235B", "DeepSeek-R1", "Llama4-Maverick"]
    baselines_to_plot = [
        "Uniform BF16 (Standard EP)",
        "Uniform FP8 (DeepEP / GEMQ-FP8)",
        "EPEG (Ours - Elastic Gating 0.40/0.05)",
        "Uniform FP4 (MoPEQ / GEMQ-FP4)"
    ]
    
    # Extract latencies
    latencies = {name: [] for name in baselines_to_plot}
    for model in models:
        records = data[model]
        for baseline in baselines_to_plot:
            # find corresponding latency
            lat = 0.0
            for r in records:
                if r["name"] == baseline:
                    lat = r["total_latency_s"]
                    break
            latencies[baseline].append(lat)
            
    # Set up styling
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models))
    width = 0.18
    
    # Elegant color palette
    colors = {
        "Uniform BF16 (Standard EP)": "#94A3B8",  # Sleek slate gray
        "Uniform FP8 (DeepEP / GEMQ-FP8)": "#38BDF8",  # Light blue
        "EPEG (Ours - Elastic Gating 0.40/0.05)": "#3B82F6",  # Dark royal blue
        "Uniform FP4 (MoPEQ / GEMQ-FP4)": "#F43F5E"  # Soft rose red
    }
    
    rects1 = ax.bar(x - 1.5*width, latencies["Uniform BF16 (Standard EP)"], width, label="Uniform BF16 (Baseline)", color=colors["Uniform BF16 (Standard EP)"], edgecolor="#475569", alpha=0.9)
    rects2 = ax.bar(x - 0.5*width, latencies["Uniform FP8 (DeepEP / GEMQ-FP8)"], width, label="Uniform FP8 (DeepEP)", color=colors["Uniform FP8 (DeepEP / GEMQ-FP8)"], edgecolor="#0284C7", alpha=0.9)
    rects3 = ax.bar(x + 0.5*width, latencies["EPEG (Ours - Elastic Gating 0.40/0.05)"], width, label="EPEG (Ours)", color=colors["EPEG (Ours - Elastic Gating 0.40/0.05)"], edgecolor="#1D4ED8", alpha=0.9)
    rects4 = ax.bar(x + 1.5*width, latencies["Uniform FP4 (MoPEQ / GEMQ-FP4)"], width, label="Uniform FP4 (MoPEQ)", color=colors["Uniform FP4 (MoPEQ / GEMQ-FP4)"], edgecolor="#BE123C", alpha=0.9)
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}s',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
            
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    
    ax.set_ylabel("End-to-End Serving Latency (Seconds)", fontsize=12, fontweight='bold')
    ax.set_title("EPEG vs. State-of-the-Art Serving Baselines: Latency Comparison (16.0 GB/s link)", fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight='bold')
    ax.legend(frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=10)
    
    # Adjust layout
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    plot_path = "outputs/phase3/epeg_sota_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Successfully generated comparison plot at {plot_path}")

if __name__ == "__main__":
    main()
