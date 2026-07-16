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

sweep1_data = results["sweep1"]

tp_scales = [1, 2, 4, 8]
policies = ["BALANCED", "RAND", "DATASET"]

ttft_data = {}
tpot_data = {}
total_latency = {}
prompt_thru = {}
gen_thru = {}

for policy in policies:
    ttft_data[policy] = [sweep1_data[str(tp)][policy]["avg_ttft_ms"] for tp in tp_scales]
    tpot_data[policy] = [sweep1_data[str(tp)][policy]["avg_tpot_ms"] for tp in tp_scales]
    total_latency[policy] = [sweep1_data[str(tp)][policy]["total_latency_s"] for tp in tp_scales]
    prompt_thru[policy] = [sweep1_data[str(tp)][policy]["prompt_thru_tok_s"] for tp in tp_scales]
    gen_thru[policy] = [sweep1_data[str(tp)][policy]["gen_thru_tok_s"] for tp in tp_scales]

# Ensure output directory exists
os.makedirs("outputs", exist_ok=True)

# ----------------------------------------------------
# 1. Create the Scaling Trend Plot (TTFT & TPOT)
# ----------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Custom aesthetic styling (light theme, modern clean design)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

colors = {
    "BALANCED": "#7C3AED",  # Violet
    "RAND": "#EF4444",      # Red/Coral
    "DATASET": "#0EA5E9"   # Light Blue / Teal
}
markers = {
    "BALANCED": "o",
    "RAND": "s",
    "DATASET": "^"
}

# Left Plot: Mean TTFT
for policy in policies:
    ax1.plot(
        tp_scales, ttft_data[policy], 
        label=policy, 
        color=colors[policy], 
        marker=markers[policy], 
        markersize=8, 
        linewidth=2.5
    )
ax1.set_title("Mean Time-to-First-Token (TTFT)", fontsize=14, pad=15, fontweight='bold')
ax1.set_xlabel("TP / EP Scale (Number of GPUs)", fontsize=12, labelpad=10)
ax1.set_ylabel("Latency (ms)", fontsize=12, labelpad=10)
ax1.set_xticks(tp_scales)
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend(fontsize=11, loc="upper right")

# Right Plot: Mean TPOT
for policy in policies:
    ax2.plot(
        tp_scales, tpot_data[policy], 
        label=policy, 
        color=colors[policy], 
        marker=markers[policy], 
        markersize=8, 
        linewidth=2.5
    )
ax2.set_title("Mean Time-per-Output-Token (TPOT)", fontsize=14, pad=15, fontweight='bold')
ax2.set_xlabel("TP / EP Scale (Number of GPUs)", fontsize=12, labelpad=10)
ax2.set_ylabel("Latency (ms)", fontsize=12, labelpad=10)
ax2.set_xticks(tp_scales)
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend(fontsize=11, loc="upper left")

plt.suptitle("Expert Routing Policies Performance Scaling Trend (Sweep 1)", fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()

# Save plot
plot_path = "outputs/sweep1_scaling_plot.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Successfully saved plot to {plot_path}")

# ----------------------------------------------------
# 2. Create the Table as an Image
# ----------------------------------------------------
fig_tbl, ax_tbl = plt.subplots(figsize=(12, 5))
ax_tbl.axis('off')

# Compile table data
headers = ["TP/EP Scale", "Routing Policy", "Total Latency (s)", "Prompt Throughput (tok/s)", "Gen Throughput (tok/s)", "Mean TTFT (ms)", "Mean TPOT (ms)"]
table_data = []

for idx, tp in enumerate(tp_scales):
    for policy in policies:
        table_data.append([
            f"TP={tp} / EP={tp}",
            policy,
            f"{total_latency[policy][idx]:.3f}",
            f"{prompt_thru[policy][idx]:.2f}",
            f"{gen_thru[policy][idx]:.2f}",
            f"{ttft_data[policy][idx]:.2f}",
            f"{tpot_data[policy][idx]:.2f}"
        ])

# Create beautiful table using matplotlib Table class
tbl = ax_tbl.table(
    cellText=table_data, 
    colLabels=headers, 
    loc='center', 
    cellLoc='center'
)

# Style table elements
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1.2, 1.6)

# Set colors: Header row, alternating data rows
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold', color='white', fontsize=12)
        cell.set_facecolor('#1E293B')  # Dark grey header
        cell.set_edgecolor('#0F172A')
    else:
        # Alternating rows
        if row % 2 == 0:
            cell.set_facecolor('#F8FAFC')
        else:
            cell.set_facecolor('#FFFFFF')
        cell.set_edgecolor('#E2E8F0')
        
        # Style policy names
        if col == 1:
            cell.get_text().set_weight('bold')
            policy_name = cell.get_text().get_text()
            if policy_name in colors:
                cell.get_text().set_color(colors[policy_name])

ax_tbl.set_title("Sweep 1 Performance Metrics: Expert Routing Policies Trend", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()

# Save table
table_path = "outputs/sweep1_results_table.png"
plt.savefig(table_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Successfully saved table to {table_path}")
