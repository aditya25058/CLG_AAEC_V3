#!/usr/bin/env python3
import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Load data dynamically from sweep_results.json
results_json_path = "outputs/sweep_results.json"
if not os.path.exists(results_json_path):
    print(f"Error: {results_json_path} not found. Please run scratch/run_sweeps.py first.")
    exit(1)

with open(results_json_path, "r") as f:
    results = json.load(f)

sweep2_data = results["sweep2"]

lambdas = [0.0, 0.2, 0.5, 0.8]
lambdas_str = ["0.0", "0.2", "0.5", "0.8"]

ttft_slow = [sweep2_data["1.0"][l]["avg_ttft_ms"] for l in lambdas_str]
tpot_slow = [sweep2_data["1.0"][l]["avg_tpot_ms"] for l in lambdas_str]

ttft_fast = [sweep2_data["32.0"][l]["avg_ttft_ms"] for l in lambdas_str]
tpot_fast = [sweep2_data["32.0"][l]["avg_tpot_ms"] for l in lambdas_str]

total_latency = {
    "slow": [sweep2_data["1.0"][l]["total_latency_s"] for l in lambdas_str],
    "fast": [sweep2_data["32.0"][l]["total_latency_s"] for l in lambdas_str]
}
prompt_thru = {
    "slow": [sweep2_data["1.0"][l]["prompt_thru_tok_s"] for l in lambdas_str],
    "fast": [sweep2_data["32.0"][l]["prompt_thru_tok_s"] for l in lambdas_str]
}
gen_thru = {
    "slow": [sweep2_data["1.0"][l]["gen_thru_tok_s"] for l in lambdas_str],
    "fast": [sweep2_data["32.0"][l]["gen_thru_tok_s"] for l in lambdas_str]
}

# Ensure output directory exists
os.makedirs("outputs", exist_ok=True)

# ----------------------------------------------------
# 1. Create the Pruning Trade-off Plot
# ----------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

# Colors matching Sweep 1's palette logic
colors = {
    "slow": "#EF4444",  # Coral/Red representing slow network
    "fast": "#10B981"   # Emerald/Green representing fast network
}

# Left Plot: TTFT
ax1.plot(lambdas, ttft_slow, label="Slow Interconnect (1.0 GB/s)", color=colors["slow"], marker="o", markersize=8, linewidth=2.5)
ax1.plot(lambdas, ttft_fast, label="Fast Interconnect (32.0 GB/s)", color=colors["fast"], marker="s", markersize=8, linewidth=2.5)
ax1.set_title("Mean Time-to-First-Token (TTFT)", fontsize=14, pad=15, fontweight='bold')
ax1.set_xlabel("Pruning Factor $\lambda_c$", fontsize=12, labelpad=10)
ax1.set_ylabel("Latency (ms)", fontsize=12, labelpad=10)
ax1.set_xticks(lambdas)
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend(fontsize=11, loc="best")

# Right Plot: TPOT
ax2.plot(lambdas, tpot_slow, label="Slow Interconnect (1.0 GB/s)", color=colors["slow"], marker="o", markersize=8, linewidth=2.5)
ax2.plot(lambdas, tpot_fast, label="Fast Interconnect (32.0 GB/s)", color=colors["fast"], marker="s", markersize=8, linewidth=2.5)
ax2.set_title("Mean Time-per-Output-Token (TPOT)", fontsize=14, pad=15, fontweight='bold')
ax2.set_xlabel("Pruning Factor $\lambda_c$", fontsize=12, labelpad=10)
ax2.set_ylabel("Latency (ms)", fontsize=12, labelpad=10)
ax2.set_xticks(lambdas)
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend(fontsize=11, loc="best")

plt.suptitle("Interconnect-Aware Gate Pruning Trade-off Trend (Sweep 2)", fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()

# Save plot
plot_path = "outputs/sweep2_pruning_plot.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Successfully saved Sweep 2 plot to {plot_path}")

# ----------------------------------------------------
# 2. Create the Table as an Image
# ----------------------------------------------------
fig_tbl, ax_tbl = plt.subplots(figsize=(12, 5))
ax_tbl.axis('off')

headers = ["Interconnect Speed", "Pruning Factor $\lambda_c$", "Total Latency (s)", "Prompt Throughput (tok/s)", "Gen Throughput (tok/s)", "Mean TTFT (ms)", "Mean TPOT (ms)"]
table_data = []

# Slow Link rows
for idx, l in enumerate(lambdas):
    table_data.append([
        "1.0 GB/s (Slow)",
        f"{l:.1f}",
        f"{total_latency['slow'][idx]:.3f}",
        f"{prompt_thru['slow'][idx]:.2f}",
        f"{gen_thru['slow'][idx]:.2f}",
        f"{ttft_slow[idx]:.2f}",
        f"{tpot_slow[idx]:.2f}"
    ])

# Fast Link rows
for idx, l in enumerate(lambdas):
    table_data.append([
        "32.0 GB/s (Fast)",
        f"{l:.1f}",
        f"{total_latency['fast'][idx]:.3f}",
        f"{prompt_thru['fast'][idx]:.2f}",
        f"{gen_thru['fast'][idx]:.2f}",
        f"{ttft_fast[idx]:.2f}",
        f"{tpot_fast[idx]:.2f}"
    ])

tbl = ax_tbl.table(
    cellText=table_data, 
    colLabels=headers, 
    loc='center', 
    cellLoc='center'
)

tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1.2, 1.6)

for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold', color='white', fontsize=12)
        cell.set_facecolor('#1E293B')
        cell.set_edgecolor('#0F172A')
    else:
        if row % 2 == 0:
            cell.set_facecolor('#F8FAFC')
        else:
            cell.set_facecolor('#FFFFFF')
        cell.set_edgecolor('#E2E8F0')
        
        # Color formatting for interconnect speed column
        if col == 0:
            cell.get_text().set_weight('bold')
            speed_text = cell.get_text().get_text()
            if "Slow" in speed_text:
                cell.get_text().set_color(colors["slow"])
            else:
                cell.get_text().set_color(colors["fast"])

ax_tbl.set_title("Sweep 2 Performance Metrics: Interconnect-Aware Gate Pruning Trend", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()

# Save table
table_path = "outputs/sweep2_results_table.png"
plt.savefig(table_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Successfully saved Sweep 2 table to {table_path}")
