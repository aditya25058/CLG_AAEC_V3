#!/usr/bin/env python3
"""
plot_real_aaec_results.py — Publication-Quality Plots for AAEC Paper

Reads the SQLite database produced by instrument_aaec_real.py and generates
all plots needed for the AAEC paper.

Usage:
  python3 plot_real_aaec_results.py --db aaec_activations.db
  python3 plot_real_aaec_results.py --db scratch/mock_activations.db  # test with mock
"""

import argparse
import json
import os
import sqlite3
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# Style configuration for publication quality
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# Color palette
COLORS = {
    "primary": "#2563EB",
    "secondary": "#DC2626",
    "accent": "#059669",
    "warning": "#D97706",
    "purple": "#7C3AED",
    "gray": "#6B7280",
}


def load_db(db_path: str):
    """Load the SQLite database."""
    conn = sqlite3.connect(db_path)
    meta = {}
    try:
        for k, v in conn.execute("SELECT key, value FROM metadata"):
            meta[k] = v
    except sqlite3.OperationalError:
        pass
    print(f"Database: {db_path}")
    print(f"Metadata: {meta}")
    total = conn.execute("SELECT COUNT(*) FROM activations").fetchone()[0]
    print(f"Total activation records: {total}")
    return conn, meta


# =====================================================================
# PLOT 1: Working-Set Growth W(n) — The Core AAEC Justification
# =====================================================================
def plot_working_set_growth(conn, output_dir, model_name="Model"):
    """
    W(n) = |union_{i=1}^{n} A_i| for each expert.
    Plots curves for various energy concentration thresholds.
    """
    # Check what columns are available in the activations table
    cursor = conn.execute("PRAGMA table_info(activations)")
    columns = [row[1] for row in cursor.fetchall()]

    has_new_columns = "energy_k_50" in columns

    if has_new_columns:
        query = (
            "SELECT layer, expert_id, token_pos, active_indices, "
            "energy_k_50, energy_k_70, energy_k_80, energy_k_90, energy_k_95, intermediate_dim "
            "FROM activations ORDER BY layer, expert_id, prompt_id, token_pos"
        )
    else:
        query = (
            "SELECT layer, expert_id, token_pos, active_indices, "
            "NULL, NULL, NULL, NULL, energy_k_95, intermediate_dim "
            "FROM activations ORDER BY layer, expert_id, prompt_id, token_pos"
        )

    cursor = conn.execute(query)

    expert_sequences = {
        "99": defaultdict(list),
        "95": defaultdict(list),
    }
    if has_new_columns:
        expert_sequences["90"] = defaultdict(list)
        expert_sequences["80"] = defaultdict(list)
        expert_sequences["70"] = defaultdict(list)
        expert_sequences["50"] = defaultdict(list)

    intermediate_dim = 512
    for layer, exp_id, t_pos, idx_json, k50, k70, k80, k90, k95, idim in cursor:
        indices = json.loads(idx_json)
        expert_sequences["99"][(layer, exp_id)].append(indices)
        expert_sequences["95"][(layer, exp_id)].append(indices[:k95])
        if has_new_columns:
            expert_sequences["90"][(layer, exp_id)].append(indices[:k90])
            expert_sequences["80"][(layer, exp_id)].append(indices[:k80])
            expert_sequences["70"][(layer, exp_id)].append(indices[:k70])
            expert_sequences["50"][(layer, exp_id)].append(indices[:k50])
        intermediate_dim = idim

    max_tokens = 200
    
    def compute_curves(seq_dict):
        curves = []
        for (layer, exp_id), sequences in seq_dict.items():
            if len(sequences) < 10:
                continue
            cumulative_set = set()
            curve = []
            for i, indices in enumerate(sequences[:max_tokens]):
                cumulative_set.update(indices)
                curve.append(len(cumulative_set))
            curves.append(curve)
        return curves

    computed_curves = {}
    for key, seq_dict in expert_sequences.items():
        computed_curves[key] = compute_curves(seq_dict)

    if not computed_curves["99"]:
        print("WARNING: No growth curves computed (insufficient data).")
        return

    def get_mean_pct(curves):
        max_len = max(len(c) for c in curves)
        padded = []
        for c in curves:
            padded.append(c + [c[-1]] * (max_len - len(c)))
        arr = np.array(padded)
        return arr.mean(axis=0) / intermediate_dim * 100, max_len

    means = {}
    max_len = 0
    for key, curves in computed_curves.items():
        means[key], max_len = get_mean_pct(curves)

    x = np.arange(1, max_len + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Custom color map for curves
    curve_colors = {
        "99": COLORS["primary"],
        "95": COLORS["secondary"],
        "90": COLORS["accent"],
        "80": COLORS["warning"],
        "70": COLORS["purple"],
        "50": COLORS["gray"],
    }

    for key in sorted(means.keys(), key=lambda x: int(x), reverse=True):
        ax.plot(x, means[key], color=curve_colors[key], linewidth=2.5, label=f"{key}% Energy W(n)")
    
    ax.axhline(y=100, color="#CCCCCC", linestyle="--", alpha=0.5,
               label=f"Full dim ({intermediate_dim} neurons)")

    ax.set_xlabel("Tokens Processed (n)")
    ax.set_ylabel("Unique Neurons Seen (%)")
    ax.set_title(f"Working-Set Growth W(n) — {model_name}\n"
                 f"(Averaged over {len(computed_curves['99'])} expert invocations)")
    ax.legend(loc="lower right")
    ax.set_xlim(1, max_len)
    ax.set_ylim(0, 105)

    out_path = os.path.join(output_dir, "working_set_growth.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


# =====================================================================
# PLOT 2: Routing Entropy per Layer
# =====================================================================
def plot_routing_entropy(conn, output_dir, model_name="Model"):
    cursor = conn.execute(
        "SELECT layer, expert_id, COUNT(*) as cnt FROM activations GROUP BY layer, expert_id"
    )
    layer_expert_counts = defaultdict(lambda: defaultdict(int))
    for layer, exp_id, cnt in cursor:
        layer_expert_counts[layer][exp_id] = cnt

    layers = sorted(layer_expert_counts.keys())
    entropies = []
    max_entropies = []
    for layer in layers:
        counts = np.array(list(layer_expert_counts[layer].values()), dtype=float)
        probs = counts / counts.sum()
        H = -np.sum(probs * np.log2(probs + 1e-12))
        H_max = np.log2(len(counts))
        entropies.append(H)
        max_entropies.append(H_max)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(layers, entropies, color=COLORS["primary"], alpha=0.8, label="Routing Entropy H")
    ax.plot(layers, max_entropies, color=COLORS["secondary"], linewidth=2,
            linestyle="--", marker="o", markersize=4, label="Max Entropy (uniform)")

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Entropy (bits)")
    ax.set_title(f"Routing Entropy per Layer — {model_name}")
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    out_path = os.path.join(output_dir, "routing_entropy.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")

# =====================================================================
# PLOT 3: Jaccard Decay with Fitted Exponential
# =====================================================================
def plot_jaccard_decay(conn, output_dir, model_name="Model"):
    # 1. Get 200 random active experts
    cursor = conn.execute("SELECT DISTINCT layer, expert_id FROM activations")
    all_experts = cursor.fetchall()
    
    import random
    random.seed(42)
    if len(all_experts) > 200:
        sampled_experts = random.sample(all_experts, 200)
    else:
        sampled_experts = all_experts

    # Check what columns are available
    cursor = conn.execute("PRAGMA table_info(activations)")
    columns = [row[1] for row in cursor.fetchall()]
    has_energy_k50 = "energy_k_50" in columns

    # 2. Query activations only for the sampled experts
    expert_sequences = defaultdict(list)
    for layer, exp_id in sampled_experts:
        if has_energy_k50:
            c = conn.execute(
                "SELECT active_indices, energy_k_50 FROM activations "
                "WHERE layer = ? AND expert_id = ? "
                "ORDER BY prompt_id, token_pos",
                (layer, exp_id)
            )
            for idx_json, k50 in c.fetchall():
                expert_sequences[(layer, exp_id)].append(set(json.loads(idx_json)[:k50]))
        else:
            c = conn.execute(
                "SELECT active_indices FROM activations "
                "WHERE layer = ? AND expert_id = ? "
                "ORDER BY prompt_id, token_pos",
                (layer, exp_id)
            )
            for (idx_json,) in c.fetchall():
                expert_sequences[(layer, exp_id)].append(set(json.loads(idx_json)))

    distances = [1, 2, 4, 8, 16, 32, 64]
    jaccards = {d: [] for d in distances}

    for (layer, exp_id) in sampled_experts:
        sets = expert_sequences[(layer, exp_id)]
        n = len(sets)
        for dist in distances:
            for i in range(min(n - dist, 100)):  # cap to 100 per expert for performance
                union = len(sets[i].union(sets[i + dist]))
                inter = len(sets[i].intersection(sets[i + dist]))
                if union > 0:
                    jaccards[dist].append(inter / union)

    means = {}
    stds = {}
    for d in distances:
        if jaccards[d]:
            means[d] = np.mean(jaccards[d])
            stds[d] = np.std(jaccards[d])

    if not means:
        print("WARNING: No Jaccard data.")
        return

    x = sorted(means.keys())
    y = [means[d] for d in x]
    yerr = [stds[d] for d in x]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(x, y, yerr=yerr, fmt="o-", color=COLORS["primary"],
                linewidth=2.5, markersize=8, capsize=5, capthick=1.5,
                label="Measured Jaccard J(d)")

    # Fit exponential decay
    if len(means) >= 3:
        j_inf = min(means.values())
        x_fit, y_fit = [], []
        for d, j in means.items():
            if j > j_inf + 0.001:
                x_fit.append(d)
                y_fit.append(np.log(j - j_inf))
        if len(x_fit) >= 2:
            slope, intercept = np.polyfit(x_fit, y_fit, 1)
            tau = -1.0 / slope if slope != 0 else 100
            j0_shift = np.exp(intercept)
            x_smooth = np.linspace(1, max(x), 200)
            y_smooth = j0_shift * np.exp(-x_smooth / tau) + j_inf
            ax.plot(x_smooth, y_smooth, "--", color=COLORS["secondary"],
                    linewidth=1.5, alpha=0.8,
                    label=f"Fit: J(d) = {j0_shift:.3f}·exp(-d/{tau:.1f}) + {j_inf:.3f}")

    ax.set_xlabel("Token Distance (d)")
    ax.set_ylabel("Jaccard Similarity J(d)")
    energy_label = "50% Energy Set" if has_energy_k50 else "99% Energy Set"
    ax.set_title(f"Temporal Neuron Reuse Decay ({energy_label}) — {model_name}")
    ax.legend(loc="upper right")
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1.05)

    out_path = os.path.join(output_dir, "jaccard_decay.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


# =====================================================================
# PLOT 4: Neuron Activation Sparsity (Energy CDF)
# =====================================================================
def plot_energy_sparsity(conn, output_dir, model_name="Model"):
    cursor = conn.execute(
        "SELECT energy_k_95, energy_k_99, energy_k_999, intermediate_dim "
        "FROM activations"
    )
    k95_list, k99_list, k999_list = [], [], []
    intermediate_dim = 512
    for k95, k99, k999, idim in cursor:
        k95_list.append(k95 / idim * 100)
        k99_list.append(k99 / idim * 100)
        k999_list.append(k999 / idim * 100)
        intermediate_dim = idim

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 100, 50)

    ax.hist(k95_list, bins=bins, alpha=0.5, color=COLORS["primary"],
            label=f"95% energy (mean {np.mean(k95_list):.1f}%)", density=True)
    ax.hist(k99_list, bins=bins, alpha=0.5, color=COLORS["secondary"],
            label=f"99% energy (mean {np.mean(k99_list):.1f}%)", density=True)
    ax.hist(k999_list, bins=bins, alpha=0.3, color=COLORS["accent"],
            label=f"99.9% energy (mean {np.mean(k999_list):.1f}%)", density=True)

    ax.set_xlabel(f"% of Intermediate Dimension ({intermediate_dim} neurons)")
    ax.set_ylabel("Density")
    ax.set_title(f"Neuron Activation Energy Concentration — {model_name}")
    ax.legend()

    out_path = os.path.join(output_dir, "energy_sparsity.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(f"  95% energy captured by {np.mean(k95_list):.1f}% of neurons")
    print(f"  99% energy captured by {np.mean(k99_list):.1f}% of neurons")


# =====================================================================
# PLOT 5: Zipf Rank-Frequency Distribution
# =====================================================================
def plot_zipf_distribution(conn, output_dir, model_name="Model"):
    cursor = conn.execute("SELECT active_indices FROM activations")

    neuron_counts = defaultdict(int)
    total_invocations = 0
    for i, (idx_json,) in enumerate(cursor):
        if i % 20 != 0:
            continue
        indices = json.loads(idx_json)
        for idx in indices:
            neuron_counts[idx] += 1
        total_invocations += 1

    if not neuron_counts:
        print("WARNING: No neuron data for Zipf plot.")
        return

    # Sort by frequency (descending)
    sorted_counts = sorted(neuron_counts.values(), reverse=True)
    ranks = np.arange(1, len(sorted_counts) + 1)
    freqs = np.array(sorted_counts) / total_invocations

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(ranks, freqs, "o", color=COLORS["primary"], markersize=3, alpha=0.5,
              label="Measured")

    # Fit Zipf: f(r) = C * r^(-alpha)
    log_r = np.log(ranks)
    log_f = np.log(freqs + 1e-12)
    alpha, log_C = np.polyfit(log_r, log_f, 1)
    C = np.exp(log_C)
    ax.loglog(ranks, C * ranks ** alpha, "--", color=COLORS["secondary"],
              linewidth=2, label=f"Zipf fit: α = {-alpha:.2f}")

    ax.set_xlabel("Neuron Rank")
    ax.set_ylabel("Activation Frequency (per invocation)")
    ax.set_title(f"Neuron Activation Rank-Frequency — {model_name}")
    ax.legend()

    out_path = os.path.join(output_dir, "zipf_distribution.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(f"  Zipf exponent α = {-alpha:.2f}")


# =====================================================================
# PLOT 6: Expert Load Distribution (Routing Skew)
# =====================================================================
def plot_expert_load(conn, output_dir, model_name="Model"):
    cursor = conn.execute(
        "SELECT expert_id, COUNT(*) as cnt FROM activations GROUP BY expert_id"
    )
    expert_counts = {}
    for exp_id, cnt in cursor:
        expert_counts[exp_id] = cnt

    total = sum(expert_counts.values())
    n_experts = len(expert_counts)
    uniform_pct = 100.0 / n_experts

    sorted_experts = sorted(expert_counts.items(), key=lambda x: x[1], reverse=True)
    ids = [e[0] for e in sorted_experts]
    pcts = [e[1] / total * 100 for e in sorted_experts]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(ids)), pcts, color=COLORS["primary"], alpha=0.8)
    ax.axhline(y=uniform_pct, color=COLORS["secondary"], linestyle="--",
               linewidth=2, label=f"Uniform ({uniform_pct:.2f}%)")

    ax.set_xlabel("Expert (sorted by load)")
    ax.set_ylabel("% of Total Activations")
    ax.set_title(f"Expert Load Distribution — {model_name}")
    ax.legend()

    out_path = os.path.join(output_dir, "expert_load.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


# =====================================================================
# MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate AAEC publication plots")
    parser.add_argument("--db", type=str, required=True, help="SQLite database path")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for plots (default: same as db)")
    args = parser.parse_args()

    conn, meta = load_db(args.db)
    model_name = meta.get("model", "Mock Model")

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.db))
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nGenerating plots in: {output_dir}\n")
    plot_working_set_growth(conn, output_dir, model_name)
    plot_routing_entropy(conn, output_dir, model_name)
    plot_jaccard_decay(conn, output_dir, model_name)
    plot_energy_sparsity(conn, output_dir, model_name)
    plot_zipf_distribution(conn, output_dir, model_name)
    plot_expert_load(conn, output_dir, model_name)

    conn.close()
    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
