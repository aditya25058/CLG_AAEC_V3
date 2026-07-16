import json
import numpy as np
import matplotlib.pyplot as plt
import os

# Set style for premium aesthetic
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 18,
    'legend.fontsize': 12,
    'grid.alpha': 0.3
})

def main():
    stats_path = "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/scratch/activation_stats.json"
    if not os.path.exists(stats_path):
        print(f"Error: {stats_path} not found. Please run the analysis script first.")
        return

    with open(stats_path, "r") as f:
        data = json.load(f)

    qwen = data.get("qwen3")
    dseek = data.get("deepseek")

    os.makedirs("/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/", exist_ok=True)

    # -----------------------------------------------------------------
    # PLOT 1: CDF of unique neurons activated per request per expert
    # -----------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    for model, name, color in [(qwen, "Qwen3-235B (128 Experts)", "#3498db"), (dseek, "DeepSeek-R1 (256 Experts)", "#e74c3c")]:
        if model is None:
            continue
        vals = np.array(model["active_neurons_per_req_exp"])
        # Convert to percentage of intermediate candidates (1024)
        pct_vals = (vals / 1024.0) * 100.0
        sorted_vals = np.sort(pct_vals)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        
        plt.plot(sorted_vals, cdf, label=name, color=color, linewidth=2.5)
        
    plt.xlabel("Percentage of Expert's Neurons Activated (%)")
    plt.ylabel("Cumulative Probability (CDF)")
    plt.title("CDF of Active Neurons per Request-Expert Pair")
    plt.xlim(0, 100)
    plt.ylim(0, 1.05)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig("/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/active_neurons_cdf.png", dpi=300)
    plt.close()

    # -----------------------------------------------------------------
    # PLOT 2: Expert Activation Frequency Distribution (Normalized Load Skew)
    # -----------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    for model, name, color in [(qwen, "Qwen3-235B (128 Experts)", "#3498db"), (dseek, "DeepSeek-R1 (256 Experts)", "#e74c3c")]:
        if model is None:
            continue
        # expert_skew_counts is request_count x num_experts. We sum activations per expert across all requests.
        skew_counts = np.array(model["expert_skew_counts"])
        total_counts_per_expert = np.sum(skew_counts, axis=0)
        
        # Normalize to percentage of total routing decisions
        total_routing_decisions = np.sum(total_counts_per_expert)
        normalized_counts = (total_counts_per_expert / total_routing_decisions) * 100.0
        
        # Sort in descending order to show skewness
        sorted_counts = np.sort(normalized_counts)[::-1]
        
        # Plot relative to the uniform routing baseline
        uniform_baseline = 100.0 / len(total_counts_per_expert)
        
        plt.plot(range(len(sorted_counts)), sorted_counts, label=name, color=color, linewidth=2)
        plt.axhline(y=uniform_baseline, color=color, linestyle="--", alpha=0.5, label=f"{name} Uniform Baseline ({uniform_baseline:.3f}%)")
        
    plt.xlabel("Expert Index (Sorted by Activation Frequency)")
    plt.ylabel("Percentage of Total Activations (%)")
    plt.title("Expert Routing Load Distribution (Skew)")
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig("/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/expert_coverage_hist.png", dpi=300)
    plt.close()

    # -----------------------------------------------------------------
    # PLOT 3: Temporal Similarity (Jaccard Overlap) vs Token Distance
    # -----------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    for model, name, color in [(qwen, "Qwen3-235B", "#3498db"), (dseek, "DeepSeek-R1", "#e74c3c")]:
        if model is None:
            continue
        jaccard_vs_dist = model["jaccard_vs_distance"]
        dists = sorted([int(k) for k in jaccard_vs_dist.keys()])
        overlaps = [jaccard_vs_dist[str(d)] for d in dists]
        
        plt.plot(dists, overlaps, marker='o', label=name, color=color, linewidth=2)
        
    plt.xlabel("Token Distance (Steps between activations)")
    plt.ylabel("Jaccard Similarity of Active Neurons")
    plt.title("Temporal Reuse Locality of Active Neurons")
    plt.xlim(1, 15)
    plt.ylim(0, 1.05)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig("/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/temporal_reuse_jaccard.png", dpi=300)
    plt.close()

    # -----------------------------------------------------------------
    # PLOT 4: Zipf Ranking of Neuron Frequencies
    # -----------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    for model, name, color in [(qwen, "Qwen3-235B", "#3498db"), (dseek, "DeepSeek-R1", "#e74c3c")]:
        if model is None:
            continue
        freqs = list(model["neuron_global_freqs"].values())
        if not freqs:
            continue
        sorted_freqs = sorted(freqs, reverse=True)
        ranks = np.arange(1, len(sorted_freqs) + 1)
        
        plt.loglog(ranks, sorted_freqs, label=name, color=color, linewidth=2.5)
        
    plt.xlabel("Neuron Global Rank (Log Scale)")
    plt.ylabel("Activation Count (Log Scale)")
    plt.title("Neuron Activation Skewness (Power-Law / Zipf)")
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig("/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/neuron_zipf_rank.png", dpi=300)
    plt.close()

    print("Plots generated successfully and saved to artifact folder.")

if __name__ == "__main__":
    main()
