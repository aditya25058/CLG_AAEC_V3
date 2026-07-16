#!/usr/bin/env python3
"""
Phase 5: Topological Wavefront Routing (TWR) - Result Visualization

Reads outputs/phase5/twr_sweep_results.json and produces four plots:
  1. Bandwidth Sensitivity – grouped bar chart of serving latency across variants.
  2. Multi-Node Scaling – speedup lines for EPEG+DAEL and EPEG+TWR vs baseline.
  3. Hotspot Stress – latency + CoV dual-axis plot across expert skew intensities.
  4. NVLink Alpha Sensitivity – latency line across twr-alpha values.

All plots saved to outputs/phase5/ and copied to the artifact directory.
"""
import os
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt

# ─── Paths ────────────────────────────────────────────────────────────────────
RESULTS_JSON = "outputs/phase5/twr_sweep_results.json"
OUTPUT_DIR = "outputs/phase5"
ARTIFACT_DIR = "/home/palakm/.gemini/antigravity-ide/brain/ed3c2dc2-4e67-4959-b431-347772d4a219/phase5_plots"

# ─── Colour Palette ──────────────────────────────────────────────────────────
COLORS = {
    "baseline":  "#EF4444",   # Red
    "dael":      "#8B5CF6",   # Purple
    "epeg":      "#10B981",   # Emerald
    "epeg_dael": "#3B82F6",   # Blue
    "epeg_twr":  "#F59E0B",   # Amber (new hero colour for TWR)
}
EDGE = {
    "baseline":  "#B91C1C",
    "dael":      "#6D28D9",
    "epeg":      "#047857",
    "epeg_dael": "#1D4ED8",
    "epeg_twr":  "#D97706",
}
LABELS = {
    "baseline":  "Baseline (Uniform BF16)",
    "dael":      "DAEL Only",
    "epeg":      "EPEG Only",
    "epeg_dael": "EPEG + P-DAEL",
    "epeg_twr":  "EPEG + TWR",
}

# ─── Setup ────────────────────────────────────────────────────────────────────
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'


def _bar_label(ax, rects, fmt="{:.2f}"):
    """Add text labels on top of bars."""
    for rect in rects:
        h = rect.get_height()
        if h > 0:
            ax.annotate(fmt.format(h),
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')


def plot_sweep1_bandwidth(data):
    """Sweep 1: Bandwidth Sensitivity – grouped bar chart."""
    bw_data = data["sweep1_bandwidth"]
    bandwidths = sorted(bw_data.keys(), key=float)
    variants = ["baseline", "dael", "epeg", "epeg_dael", "epeg_twr"]
    n_vars = len(variants)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(bandwidths))
    width = 0.15

    for i, var in enumerate(variants):
        latencies = [bw_data[bw].get(var, {}).get("total_latency_s", 0.0) for bw in bandwidths]
        offset = (i - (n_vars - 1) / 2) * width
        rects = ax.bar(x + offset, latencies, width,
                       label=LABELS[var], color=COLORS[var], edgecolor=EDGE[var], zorder=3)
        _bar_label(ax, rects, fmt="{:.1f}")

    ax.set_title("Sweep 1: Serving Latency vs. Link Bandwidth", fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{float(b):.0f} GB/s" for b in bandwidths], fontsize=11)
    ax.set_xlabel("Interconnect Link Bandwidth", fontsize=12)
    ax.set_ylabel("Total Serving Latency (s)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.legend(fontsize=9, ncol=3, loc="upper right")
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "twr_sweep1_bandwidth.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    return out


def plot_sweep2_scaling(data):
    """Sweep 2: Multi-Node Scaling – speedup line chart."""
    sc_data = data["sweep2_scaling"]
    node_counts = sorted(sc_data.keys(), key=int)
    scaling_variants = ["epeg_dael", "epeg_twr"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for var in scaling_variants:
        # Speedup = baseline_latency / variant_latency
        speedups = []
        for nc in node_counts:
            base_lat = sc_data[nc].get("baseline", {}).get("total_latency_s", 1.0)
            var_lat = sc_data[nc].get(var, {}).get("total_latency_s", base_lat)
            speedups.append(base_lat / var_lat if var_lat > 0 else 1.0)

        marker = 's' if var == "epeg_dael" else '^'
        ax.plot([int(n) for n in node_counts], speedups, marker=marker, linewidth=2.5,
                markersize=9, label=LABELS[var], color=COLORS[var])

    ax.axhline(y=1.0, color="#94A3B8", linestyle="--", linewidth=1.2, label="Baseline (1×)", zorder=2)

    ax.set_title("Sweep 2: Multi-Node Scaling – Speedup over Baseline", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Number of Nodes", fontsize=12)
    ax.set_ylabel("Speedup (×)", fontsize=12)
    ax.set_xticks([int(n) for n in node_counts])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=10)
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "twr_sweep2_scaling.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    return out


def plot_sweep3_skew(data):
    """Sweep 3: Hotspot Stress – dual-axis (latency + CoV) line chart."""
    sk_data = data["sweep3_skew"]
    skew_vals = sorted(sk_data.keys(), key=float)
    skew_variants = ["epeg", "epeg_dael", "epeg_twr"]

    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax2 = ax1.twinx()

    for var in skew_variants:
        latencies = [sk_data[s].get(var, {}).get("total_latency_s", 0.0) for s in skew_vals]
        covs      = [sk_data[s].get(var, {}).get("original_cov", 0.0) for s in skew_vals]

        x_pos = [float(s) for s in skew_vals]
        marker_lat = {'epeg': 'o', 'epeg_dael': 's', 'epeg_twr': '^'}[var]
        marker_cov = {'epeg': 'D', 'epeg_dael': 'P', 'epeg_twr': 'X'}[var]

        ax1.plot(x_pos, latencies, marker=marker_lat, linewidth=2.5, markersize=8,
                 label=f"{LABELS[var]} (latency)", color=COLORS[var])
        ax2.plot(x_pos, covs, marker=marker_cov, linewidth=1.8, markersize=7,
                 linestyle='--', label=f"{LABELS[var]} (CoV)", color=COLORS[var], alpha=0.7)

    ax1.set_title("Sweep 3: Hotspot Resilience – Latency & Load CoV vs. Expert Skew",
                   fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel("Expert Skew Intensity", fontsize=12)
    ax1.set_ylabel("Total Serving Latency (s)", fontsize=12, color="#1E293B")
    ax2.set_ylabel("Load Coefficient of Variation (CoV)", fontsize=12, color="#64748B")
    ax1.set_xticks([float(s) for s in skew_vals])
    ax1.set_xticklabels([f"{float(s):.0%}" for s in skew_vals], fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Merge legends
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, ncol=2, loc="upper left")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "twr_sweep3_skew.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    return out


def plot_sweep4_alpha(data):
    """Sweep 4: NVLink Alpha Sensitivity – latency bar + line chart."""
    al_data = data["sweep4_alpha"]
    alphas = sorted(al_data.keys(), key=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(alphas))

    latencies = [al_data[a].get("total_latency_s", 0.0) for a in alphas]

    bars = ax.bar(x_pos, latencies, 0.5, color=COLORS["epeg_twr"], edgecolor=EDGE["epeg_twr"], zorder=3)
    ax.plot(x_pos, latencies, marker='o', linewidth=2, markersize=8, color="#D97706", zorder=4)
    _bar_label(ax, bars, fmt="{:.2f}")

    ax.set_title("Sweep 4: NVLink Speed Scale (α) Sensitivity – EPEG+TWR",
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"α = {float(a):.0f}" for a in alphas], fontsize=12)
    ax.set_xlabel("NVLink-to-RDMA Bandwidth Ratio (α)", fontsize=12)
    ax.set_ylabel("Total Serving Latency (s)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4, zorder=0)
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "twr_sweep4_alpha.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    return out


def main():
    if not os.path.exists(RESULTS_JSON):
        print(f"Error: {RESULTS_JSON} not found. Run scratch/run_twr_sweeps.py first.")
        return

    with open(RESULTS_JSON, "r") as f:
        data = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    plots = []
    print("=== Generating TWR Phase 5 Plots ===")

    out = plot_sweep1_bandwidth(data)
    shutil.copy(out, os.path.join(ARTIFACT_DIR, os.path.basename(out)))
    plots.append(out)
    print(f"  ✓ Bandwidth sensitivity:  {out}")

    out = plot_sweep2_scaling(data)
    shutil.copy(out, os.path.join(ARTIFACT_DIR, os.path.basename(out)))
    plots.append(out)
    print(f"  ✓ Multi-node scaling:     {out}")

    out = plot_sweep3_skew(data)
    shutil.copy(out, os.path.join(ARTIFACT_DIR, os.path.basename(out)))
    plots.append(out)
    print(f"  ✓ Hotspot stress:         {out}")

    out = plot_sweep4_alpha(data)
    shutil.copy(out, os.path.join(ARTIFACT_DIR, os.path.basename(out)))
    plots.append(out)
    print(f"  ✓ Alpha sensitivity:      {out}")

    print(f"\n=== All 4 plots saved to {OUTPUT_DIR}/ and {ARTIFACT_DIR}/ ===")


if __name__ == "__main__":
    main()
