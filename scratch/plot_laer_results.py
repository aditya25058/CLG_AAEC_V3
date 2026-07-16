#!/usr/bin/env python3
"""Plot LAER sweep results — generates 5 publication-quality figures."""
import json
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ─── Configure plot aesthetics ────────────────────────────────────────────
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

# Color palette
COLORS = {
    'baseline': '#8b949e',
    'laer': '#58a6ff',
    'laer_aggressive': '#f0883e',
    'laer_mild': '#7ee787',
    'dael': '#d2a8ff',
    'laer_dael': '#f778ba',
    'twr': '#ff7b72',
    'epeg': '#79c0ff',
    'laer_twr': '#ffa657',
    'laer_epeg': '#56d364',
    'full_stack': '#f0e68c',
    'twr_only': '#ff7b72',
    'epeg_only': '#79c0ff',
    'laer_only': '#58a6ff',
}


def plot_sweep1_bandwidth(results, out_dir):
    """Sweep 1: Baseline vs LAER across bandwidths."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('LAER: Bandwidth Sensitivity Analysis', fontsize=16, fontweight='bold', color='#58a6ff')
    
    bandwidths = sorted(results.keys(), key=float)
    x = np.arange(len(bandwidths))
    width = 0.25
    
    variants = ['baseline', 'laer', 'laer_aggressive']
    labels = ['Baseline', 'LAER (γ=0.70)', 'LAER (γ=0.50)']
    
    # Plot 1: Latency
    for i, (var, label) in enumerate(zip(variants, labels)):
        vals = [results[bw].get(var, {}).get('total_latency_s', 0) for bw in bandwidths]
        axes[0].bar(x + i * width, vals, width, label=label, color=COLORS.get(var, '#888'),
                   edgecolor='#30363d', linewidth=0.5)
    axes[0].set_xlabel('Link Bandwidth (GB/s)')
    axes[0].set_ylabel('Total Latency (s)')
    axes[0].set_title('End-to-End Latency', fontsize=12)
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(bandwidths)
    axes[0].legend(fontsize=9, loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Remote Fraction
    for i, (var, label) in enumerate(zip(variants, labels)):
        vals = [results[bw].get(var, {}).get('avg_remote_fraction', 0) for bw in bandwidths]
        axes[1].bar(x + i * width, vals, width, label=label, color=COLORS.get(var, '#888'),
                   edgecolor='#30363d', linewidth=0.5)
    axes[1].set_xlabel('Link Bandwidth (GB/s)')
    axes[1].set_ylabel('Remote Token Fraction')
    axes[1].set_title('Cross-Node Traffic Reduction', fontsize=12)
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(bandwidths)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: TTFT
    for i, (var, label) in enumerate(zip(variants, labels)):
        vals = [results[bw].get(var, {}).get('avg_ttft_ms', 0) for bw in bandwidths]
        axes[2].bar(x + i * width, vals, width, label=label, color=COLORS.get(var, '#888'),
                   edgecolor='#30363d', linewidth=0.5)
    axes[2].set_xlabel('Link Bandwidth (GB/s)')
    axes[2].set_ylabel('TTFT (ms)')
    axes[2].set_title('Time to First Token', fontsize=12)
    axes[2].set_xticks(x + width)
    axes[2].set_xticklabels(bandwidths)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(out_dir, 'laer_sweep1_bandwidth.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_sweep2_pareto(results, out_dir):
    """Sweep 2: Beta/Gamma Pareto Frontier."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    fig.suptitle('LAER: Quality vs Latency Pareto Frontier', fontsize=16, fontweight='bold', color='#58a6ff')
    
    gammas = []
    latencies = []
    remote_fracs = []
    quality_deltas = []
    labels = []
    
    for key, metrics in results.items():
        gamma = metrics.get('gamma', 0)
        beta = metrics.get('beta', 0)
        gammas.append(gamma)
        latencies.append(metrics.get('total_latency_s', 0))
        remote_fracs.append(metrics.get('avg_remote_fraction', 0))
        quality_deltas.append(metrics.get('avg_quality_delta', 0))
        labels.append(f'β={beta}, γ={gamma}')
    
    # Scatter: latency vs quality delta, sized by remote fraction
    scatter = ax.scatter(quality_deltas, latencies,
                         c=gammas, cmap='RdYlGn', s=200,
                         edgecolors='white', linewidth=1.5, zorder=5)
    
    for i, label in enumerate(labels):
        ax.annotate(label, (quality_deltas[i], latencies[i]),
                    textcoords="offset points", xytext=(10, 5),
                    fontsize=8, color='#c9d1d9')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('γ (Remote Penalty)', color='#c9d1d9')
    cbar.ax.yaxis.set_tick_params(color='#8b949e')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#8b949e')
    
    ax.set_xlabel('Quality Delta (gate score loss fraction)', fontsize=12)
    ax.set_ylabel('Total Latency (s)', fontsize=12)
    ax.set_title('Trade-off: Lower γ → Less Remote Traffic, Tiny Quality Cost', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(out_dir, 'laer_sweep2_pareto.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_sweep3_skew(results, out_dir):
    """Sweep 3: Skew robustness."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('LAER + DAEL: Robustness Under Expert Popularity Skew', fontsize=16, fontweight='bold', color='#58a6ff')
    
    skews = sorted(results.keys(), key=float)
    variants = ['baseline', 'laer', 'dael', 'laer_dael']
    labels = ['Baseline', 'LAER', 'DAEL', 'LAER + DAEL']
    
    # Latency
    for var, label in zip(variants, labels):
        vals = [results[s].get(var, {}).get('total_latency_s', 0) for s in skews]
        axes[0].plot(skews, vals, 'o-', label=label, color=COLORS.get(var, '#888'), linewidth=2, markersize=6)
    axes[0].set_xlabel('Expert Skew Intensity')
    axes[0].set_ylabel('Total Latency (s)')
    axes[0].set_title('Latency vs Skew', fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # Load Balance (CoV)
    for var, label in zip(variants, labels):
        vals = [results[s].get(var, {}).get('original_cov', 0) for s in skews]
        axes[1].plot(skews, vals, 'o-', label=label, color=COLORS.get(var, '#888'), linewidth=2, markersize=6)
    axes[1].set_xlabel('Expert Skew Intensity')
    axes[1].set_ylabel('Load Imbalance (CoV)')
    axes[1].set_title('Load Balance vs Skew', fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(out_dir, 'laer_sweep3_skew.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_sweep4_synergy(results, out_dir):
    """Sweep 4: LAER + TWR + EPEG synergy."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('LAER Synergy with TWR & EPEG', fontsize=16, fontweight='bold', color='#58a6ff')
    
    bws = sorted(results.keys(), key=float)
    variants = ['baseline', 'laer_only', 'twr_only', 'epeg_only', 'laer_twr', 'laer_epeg', 'full_stack']
    labels = ['Baseline', 'LAER', 'TWR', 'EPEG', 'LAER+TWR', 'LAER+EPEG', 'Full Stack']
    
    x = np.arange(len(bws))
    width = 0.12
    
    for i, (var, label) in enumerate(zip(variants, labels)):
        vals = [results[bw].get(var, {}).get('total_latency_s', 0) for bw in bws]
        axes[0].bar(x + i * width, vals, width, label=label,
                   color=COLORS.get(var, '#888'), edgecolor='#30363d', linewidth=0.5)
    axes[0].set_xlabel('Link Bandwidth (GB/s)')
    axes[0].set_ylabel('Total Latency (s)')
    axes[0].set_title('Latency: Individual vs Combined Optimizations', fontsize=12)
    axes[0].set_xticks(x + width * 3)
    axes[0].set_xticklabels(bws)
    axes[0].legend(fontsize=8, ncol=2, loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Speedup relative to baseline
    for i, (var, label) in enumerate(zip(variants[1:], labels[1:])):
        speedups = []
        for bw in bws:
            base_lat = results[bw].get('baseline', {}).get('total_latency_s', 1)
            var_lat = results[bw].get(var, {}).get('total_latency_s', 1)
            speedups.append(base_lat / max(var_lat, 1e-9) if base_lat > 0 else 1.0)
        axes[1].bar(x + i * width, speedups, width, label=label,
                   color=COLORS.get(var, '#888'), edgecolor='#30363d', linewidth=0.5)
    axes[1].axhline(y=1.0, color='#8b949e', linestyle='--', alpha=0.5, linewidth=1)
    axes[1].set_xlabel('Link Bandwidth (GB/s)')
    axes[1].set_ylabel('Speedup vs Baseline')
    axes[1].set_title('Speedup Analysis', fontsize=12)
    axes[1].set_xticks(x + width * 3)
    axes[1].set_xticklabels(bws)
    axes[1].legend(fontsize=8, ncol=2, loc='upper left')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(out_dir, 'laer_sweep4_synergy.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_sweep5_rtf(results, out_dir):
    """Sweep 5: Remote Token Fraction deep dive."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('LAER: Remote Token Fraction & Quality Trade-off', fontsize=16, fontweight='bold', color='#58a6ff')
    
    variants = list(results.keys())
    labels_clean = [v.replace('_', ' ').title() for v in variants]
    
    # Remote fraction
    remote_fracs = [results[v].get('avg_remote_fraction', 0) for v in variants]
    bars = axes[0].barh(labels_clean, remote_fracs, color='#58a6ff', edgecolor='#30363d')
    axes[0].set_xlabel('Avg Remote Token Fraction')
    axes[0].set_title('Remote Traffic Reduction', fontsize=12)
    axes[0].grid(True, alpha=0.3, axis='x')
    for bar, val in zip(bars, remote_fracs):
        axes[0].text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=9, color='#c9d1d9')
    
    # Quality delta
    q_deltas = [results[v].get('avg_quality_delta', 0) for v in variants]
    bars2 = axes[1].barh(labels_clean, q_deltas, color='#f0883e', edgecolor='#30363d')
    axes[1].set_xlabel('Avg Quality Delta')
    axes[1].set_title('Quality Cost (Lower = Better)', fontsize=12)
    axes[1].grid(True, alpha=0.3, axis='x')
    for bar, val in zip(bars2, q_deltas):
        axes[1].text(bar.get_width() + 0.0001, bar.get_y() + bar.get_height()/2,
                    f'{val:.6f}', va='center', fontsize=9, color='#c9d1d9')
    
    # Inter-node tokens total
    inter_nodes = [results[v].get('total_inter_node_tokens', 0) for v in variants]
    bars3 = axes[2].barh(labels_clean, inter_nodes, color='#7ee787', edgecolor='#30363d')
    axes[2].set_xlabel('Total Inter-Node Tokens')
    axes[2].set_title('Cross-Node Communication Volume', fontsize=12)
    axes[2].grid(True, alpha=0.3, axis='x')
    for bar, val in zip(bars3, inter_nodes):
        axes[2].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    str(val), va='center', fontsize=9, color='#c9d1d9')
    
    plt.tight_layout()
    path = os.path.join(out_dir, 'laer_sweep5_rtf.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def main():
    results_path = "outputs/phase4/LAER/laer_sweep_results.json"
    out_dir = os.path.join(
        os.path.expanduser("~"),
        ".gemini/antigravity-ide/brain/ed3c2dc2-4e67-4959-b431-347772d4a219/laer_plots"
    )
    os.makedirs(out_dir, exist_ok=True)
    
    with open(results_path) as f:
        results = json.load(f)
    
    print("Generating LAER plots...")
    paths = []
    
    if "sweep1_bandwidth" in results:
        paths.append(plot_sweep1_bandwidth(results["sweep1_bandwidth"], out_dir))
    
    if "sweep2_pareto" in results:
        paths.append(plot_sweep2_pareto(results["sweep2_pareto"], out_dir))
    
    if "sweep3_skew" in results:
        paths.append(plot_sweep3_skew(results["sweep3_skew"], out_dir))
    
    if "sweep4_synergy" in results:
        paths.append(plot_sweep4_synergy(results["sweep4_synergy"], out_dir))
    
    if "sweep5_rtf" in results:
        paths.append(plot_sweep5_rtf(results["sweep5_rtf"], out_dir))
    
    print(f"\nAll {len(paths)} plots generated in: {out_dir}")


if __name__ == "__main__":
    main()
