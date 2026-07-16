#!/usr/bin/env python3
import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Load data from the sweep_results.json
results_json_path = "outputs/sweep_results.json"
if not os.path.exists(results_json_path):
    print(f"Error: {results_json_path} not found. Please run scratch/run_sweeps.py first.")
    exit(1)

with open(results_json_path, "r") as f:
    results = json.load(f)

sweep3_data = results["sweep3"]

bandwidths = [1.0, 4.0, 16.0, 32.0]
bandwidths_str = ["1.0", "4.0", "16.0", "32.0"]

ttft_enabled = []
tpot_enabled = []
ttft_disabled = []
tpot_disabled = []

total_latency = {"True": [], "False": []}
prompt_thru = {"True": [], "False": []}
gen_thru = {"True": [], "False": []}

for bw in bandwidths_str:
    # Enabled (True)
    m_en = sweep3_data[bw]["True"]
    ttft_enabled.append(m_en["avg_ttft_ms"])
    tpot_enabled.append(m_en["avg_tpot_ms"])
    total_latency["True"].append(m_en["total_latency_s"])
    prompt_thru["True"].append(m_en["prompt_thru_tok_s"])
    gen_thru["True"].append(m_en["gen_thru_tok_s"])

    # Disabled (False)
    m_dis = sweep3_data[bw]["False"]
    ttft_disabled.append(m_dis["avg_ttft_ms"])
    tpot_disabled.append(m_dis["avg_tpot_ms"])
    total_latency["False"].append(m_dis["total_latency_s"])
    prompt_thru["False"].append(m_dis["prompt_thru_tok_s"])
    gen_thru["False"].append(m_dis["gen_thru_tok_s"])

# Ensure output directory exists
os.makedirs("outputs", exist_ok=True)

# ----------------------------------------------------
# 1. Create the Crossover Trend Plot
# ----------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

colors = {
    "enabled": "#10B981",   # Emerald/Green representing HDFG Enabled
    "disabled": "#EF4444"   # Coral/Red representing HDFG Disabled
}

# Left Plot: TTFT
ax1.plot(bandwidths, ttft_enabled, label="HDFG Enabled", color=colors["enabled"], marker="o", markersize=8, linewidth=2.5)
ax1.plot(bandwidths, ttft_disabled, label="HDFG Disabled", color=colors["disabled"], marker="s", markersize=8, linewidth=2.5)
ax1.set_title("Mean Time-to-First-Token (TTFT)", fontsize=14, pad=15, fontweight='bold')
ax1.set_xlabel("Interconnect Bandwidth (GB/s)", fontsize=12, labelpad=10)
ax1.set_ylabel("Latency (ms)", fontsize=12, labelpad=10)
ax1.set_xscale("log")
ax1.set_xticks(bandwidths)
ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend(fontsize=11, loc="best")

# Right Plot: TPOT
ax2.plot(bandwidths, tpot_enabled, label="HDFG Enabled", color=colors["enabled"], marker="o", markersize=8, linewidth=2.5)
ax2.plot(bandwidths, tpot_disabled, label="HDFG Disabled", color=colors["disabled"], marker="s", markersize=8, linewidth=2.5)
ax2.set_title("Mean Time-per-Output-Token (TPOT)", fontsize=14, pad=15, fontweight='bold')
ax2.set_xlabel("Interconnect Bandwidth (GB/s)", fontsize=12, labelpad=10)
ax2.set_ylabel("Latency (ms)", fontsize=12, labelpad=10)
ax2.set_xscale("log")
ax2.set_xticks(bandwidths)
ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend(fontsize=11, loc="best")

plt.suptitle("Hierarchical Dispatch-Fetch Gating (HDFG) Crossover Trend (Sweep 3)", fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()

# Save plot
plot_path = "outputs/sweep3_crossover_plot.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Successfully saved Sweep 3 plot to {plot_path}")

# ----------------------------------------------------
# 2. Create the Table as an Image
# ----------------------------------------------------
fig_tbl, ax_tbl = plt.subplots(figsize=(12, 5))
ax_tbl.axis('off')

headers = ["Interconnect Speed", "HDFG Status", "Total Latency (s)", "Prompt Throughput (tok/s)", "Gen Throughput (tok/s)", "Mean TTFT (ms)", "Mean TPOT (ms)"]
table_data = []

for idx, bw in enumerate(bandwidths):
    # Enabled Row
    table_data.append([
        f"{bw:.1f} GB/s",
        "Enabled",
        f"{total_latency['True'][idx]:.3f}",
        f"{prompt_thru['True'][idx]:.2f}",
        f"{gen_thru['True'][idx]:.2f}",
        f"{ttft_enabled[idx]:.2f}",
        f"{tpot_enabled[idx]:.2f}"
    ])
    # Disabled Row
    table_data.append([
        f"{bw:.1f} GB/s",
        "Disabled",
        f"{total_latency['False'][idx]:.3f}",
        f"{prompt_thru['False'][idx]:.2f}",
        f"{gen_thru['False'][idx]:.2f}",
        f"{ttft_disabled[idx]:.2f}",
        f"{tpot_disabled[idx]:.2f}"
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
        
        # Color formatting for status column
        if col == 1:
            cell.get_text().set_weight('bold')
            status_text = cell.get_text().get_text()
            if status_text == "Enabled":
                cell.get_text().set_color(colors["enabled"])
            else:
                cell.get_text().set_color(colors["disabled"])

ax_tbl.set_title("Sweep 3 Performance Metrics: HDFG Enabled vs Disabled Trend", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()

# Save table
table_path = "outputs/sweep3_results_table.png"
plt.savefig(table_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Successfully saved Sweep 3 table to {table_path}")
