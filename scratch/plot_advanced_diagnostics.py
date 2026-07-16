#!/usr/bin/env python3
"""Generate Latency Component Breakdown and Network Roofline plots for systems diagnostics."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Configure dark theme aesthetics matching the existing figures
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

def plot_latency_breakdown(out_dir):
    configs = [
        "Baseline",
        "LAER Only",
        "TWR Only",
        "EPEG + TWR",
        "Ours (Full Co-Design)"
    ]
    
    # Components in ms
    compute = np.array([6.40, 6.40, 6.40, 6.40, 6.40])
    communication = np.array([10.12, 7.24, 4.06, 4.35, 4.35])  # exposed communication latency
    queueing = np.array([15.20, 15.20, 1.30, 0.90, 0.90])
    sync_stall = np.array([178.12, 180.88, 0.00, 0.00, 0.00])  # RTO / Congestion timeouts
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bar_width = 0.5
    y_pos = np.arange(len(configs))
    
    # Plot stacked bars
    p1 = ax.barh(y_pos, compute, bar_width, label="NPU Compute", color='#f2d56a', edgecolor='#30363d')
    p2 = ax.barh(y_pos, communication, bar_width, left=compute, label="Exposed Network Tx (Serialization/Prop)", color='#58a6ff', edgecolor='#30363d')
    p3 = ax.barh(y_pos, queueing, bar_width, left=compute+communication, label="Switch Queueing Delay", color='#bc8cff', edgecolor='#30363d')
    p4 = ax.barh(y_pos, sync_stall, bar_width, left=compute+communication+queueing, label="Synchronization Stall (RTO timeouts)", color='#ff7b72', hatch='//', edgecolor='#30363d')
    
    # Value labels
    for idx in range(len(configs)):
        total = compute[idx] + communication[idx] + queueing[idx] + sync_stall[idx]
        ax.text(total + 2, idx, f"{total:.2f} ms", va='center', ha='left', fontweight='bold', color='#c9d1d9')
        
    ax.set_yticks(y_pos)
    ax.set_yticklabels(configs, fontweight='bold')
    ax.invert_yaxis()  # top-down order
    
    ax.set_xlabel("Time-to-Last-Token Latency (ms)", fontsize=12, labelpad=10)
    ax.set_title("Critical Path Latency Component Breakdown (16 Nodes, EP=32)", fontsize=14, fontweight='bold', pad=20, color='#58a6ff')
    ax.grid(True, axis='x', alpha=0.3)
    ax.legend(loc="lower right", frameon=True, facecolor='#161b22', edgecolor='#30363d')
    ax.set_xlim(0, 230)
    
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "latency_breakdown.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated Latency Breakdown plot at: {plot_path}")

def plot_network_roofline(out_dir):
    # Data points
    # (Remote Bytes in MB, Latency in ms)
    points = {
        "Baseline": (9.50, 209.84, '#ff7b72', 's'),
        "LAER Only": (6.62, 209.72, '#ff9f6a', '^'),
        "TWR Only": (9.50, 11.76, '#bc8cff', 'D'),
        "EPEG + TWR": (5.04, 11.65, '#f2d56a', 'p'),
        "Ours (Full Co-Design)": (3.51, 11.65, '#10b981', 'H')
    }
    
    fig, ax = plt.subplots(figsize=(10, 6.5))
    
    for name, (bytes_val, lat_val, color, marker) in points.items():
        ax.scatter(bytes_val, lat_val, color=color, marker=marker, s=250, edgecolor='#30363d', label=name, zorder=5)
        
    # Draw arrows representing design iterations
    # 1. Baseline -> LAER (Bytes reduction, same latency)
    ax.annotate("LAER: Traffic Locality", 
                xy=(6.75, 209.72), xytext=(9.3, 209.84),
                arrowprops=dict(facecolor='#ff9f6a', shrink=0.08, width=1.5, headwidth=6, headlength=6),
                color='#ff9f6a', fontsize=9, fontweight='bold', ha='right', va='bottom')
                
    # 2. Baseline -> TWR (Congestion removal, huge latency reduction)
    ax.annotate("TWR: Congestion Elimination", 
                xy=(9.50, 14.5), xytext=(9.50, 175.0),
                arrowprops=dict(facecolor='#bc8cff', shrink=0.08, width=1.5, headwidth=6, headlength=6),
                color='#bc8cff', fontsize=9, fontweight='bold', ha='left', va='center')
                
    # 3. TWR -> EPEG + TWR (Bytes compression)
    ax.annotate("EPEG: Compression", 
                xy=(5.2, 11.65), xytext=(9.2, 11.76),
                arrowprops=dict(facecolor='#f2d56a', shrink=0.08, width=1.5, headwidth=6, headlength=6),
                color='#f2d56a', fontsize=9, fontweight='bold', ha='right', va='top')
                
    # 4. EPEG+TWR -> Ours (Locality + Compression synergy)
    ax.annotate("Ours (Locality + Compression)", 
                xy=(3.65, 11.65), xytext=(4.9, 11.65),
                arrowprops=dict(facecolor='#10b981', shrink=0.08, width=1.5, headwidth=6, headlength=6),
                color='#10b981', fontsize=9, fontweight='bold', ha='left', va='bottom')
                
    ax.set_yscale('log')
    ax.set_ylim(5, 400)
    # Custom y-ticks for log scale readability
    ax.set_yticks([5, 10, 20, 50, 100, 200, 300])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    
    ax.set_xlim(3.0, 10.5)
    ax.set_xlabel("Remote Inter-Node Data Transmitted (MB)", fontsize=12, labelpad=10)
    ax.set_ylabel("Time-to-Last-Token Latency (ms, Log Scale)", fontsize=12, labelpad=10)
    ax.set_title("MoE Network Collective Roofline: Traffic Footprint vs. Latency", fontsize=14, fontweight='bold', pad=20, color='#58a6ff')
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="center left", frameon=True, facecolor='#161b22', edgecolor='#30363d')
    
    # Draw a line representing congestion boundary
    ax.axhline(y=32.0, color='#ea4a4a', linestyle=':', alpha=0.5)
    ax.text(3.2, 35.0, "Switch Incast Congestion Threshold (Packet Drops)", color='#ea4a4a', fontsize=9, alpha=0.8)
    
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "roofline_network.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated Network Roofline plot at: {plot_path}")

def main():
    out_dir = "outputs/phase5"
    os.makedirs(out_dir, exist_ok=True)
    
    plot_latency_breakdown(out_dir)
    plot_network_roofline(out_dir)
    
    # Copy to the brain artifacts directory
    artifact_dir = "/home/palakm/.gemini/antigravity-ide/brain/ed3c2dc2-4e67-4959-b431-347772d4a219/sla_gating"
    os.makedirs(artifact_dir, exist_ok=True)
    import shutil
    shutil.copy(os.path.join(out_dir, "latency_breakdown.png"), os.path.join(artifact_dir, "latency_breakdown.png"))
    shutil.copy(os.path.join(out_dir, "roofline_network.png"), os.path.join(artifact_dir, "roofline_network.png"))
    print("Successfully copied advanced diagnostics plots to brain artifacts.")

if __name__ == "__main__":
    main()
