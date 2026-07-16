import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define file paths
llama4_pcie_paths = {
    1: "/home/gpu2/aditya_llmservingsim2.0/outputs/llama4_maverick_h100_tp1.csv",
    2: "/home/gpu2/aditya_llmservingsim2.0/outputs/llama4_maverick_h100_tp2_pcie.csv",
    4: "/home/gpu2/aditya_llmservingsim2.0/outputs/llama4_maverick_h100_tp4_pcie.csv",
    8: "/home/gpu2/aditya_llmservingsim2.0/outputs/llama4_maverick_h100_tp8_pcie.csv"
}

llama4_nvlink_paths = {
    1: "/home/gpu2/aditya_llmservingsim2.0/outputs/llama4_maverick_h100_tp1.csv",
    2: "/home/gpu2/aditya_llmservingsim2.0/outputs/llama4_maverick_h100_tp2_nvlink.csv",
    4: "/home/gpu2/aditya_llmservingsim2.0/outputs/llama4_maverick_h100_tp4_nvlink.csv",
    8: "/home/gpu2/aditya_llmservingsim2.0/outputs/llama4_maverick_h100_tp8_nvlink.csv"
}

# 1. Parse Llama-4-Maverick (PCIe)
llama4_pcie_ttft = []
llama4_pcie_tpot = []
for tp in [1, 2, 4, 8]:
    df = pd.read_csv(llama4_pcie_paths[tp])
    llama4_pcie_ttft.append(df['TTFT'].mean() / 1e6)  # convert to ms
    llama4_pcie_tpot.append(df['TPOT'].mean() / 1e6)  # convert to ms

# 2. Parse Llama-4-Maverick (NVLink)
llama4_nvlink_ttft = []
llama4_nvlink_tpot = []
for tp in [1, 2, 4, 8]:
    df = pd.read_csv(llama4_nvlink_paths[tp])
    llama4_nvlink_ttft.append(df['TTFT'].mean() / 1e6)  # convert to ms
    llama4_nvlink_tpot.append(df['TPOT'].mean() / 1e6)  # convert to ms

# Sleek modern styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=150)

# Curated premium color palette
c_pcie = '#E63946'      # Energetic Crimson Red
c_nvlink = '#1D3557'    # Sleek Deep Navy

tps = ['TP=1', 'TP=2', 'TP=4', 'TP=8']

# ----------------- TTFT Plot -----------------
# We use log scale to visualize extreme disparities gracefully (PCIe vs NVLink)
ax1.plot(tps, llama4_pcie_ttft, marker='o', markersize=8, color=c_pcie, linewidth=2.5, label='Llama-4-Maverick (PCIe: 16 GB/s, 20us)', linestyle='--', alpha=0.9)
ax1.plot(tps, llama4_nvlink_ttft, marker='d', markersize=8, color=c_nvlink, linewidth=2.5, label='Llama-4-Maverick (NVLink: 900 GB/s, 1.5us)', linestyle='-', alpha=0.95)

ax1.set_yscale('log')
ax1.set_title('Mean Time to First Token (TTFT)', fontsize=14, fontweight='bold', pad=15, color='#222222')
ax1.set_ylabel('Latency (ms, Log Scale)', fontsize=12, fontweight='semibold', color='#444444')
ax1.set_xlabel('Tensor Parallelism (TP) Degree', fontsize=12, fontweight='semibold', color='#444444')
ax1.grid(True, which="both", linestyle='--', alpha=0.5)
ax1.tick_params(axis='both', labelsize=11)

# Annotate points with offset to prevent overlap
for i, val in enumerate(llama4_pcie_ttft):
    ax1.annotate(f"{val:.2f}ms", (i, val), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold', fontsize=9, color=c_pcie)
for i, val in enumerate(llama4_nvlink_ttft):
    ax1.annotate(f"{val:.2f}ms", (i, val), textcoords="offset points", xytext=(0,-15), ha='center', fontweight='bold', fontsize=9, color=c_nvlink)

# ----------------- TPOT Plot -----------------
ax2.plot(tps, llama4_pcie_tpot, marker='o', markersize=8, color=c_pcie, linewidth=2.5, label='Llama-4-Maverick (PCIe: 16 GB/s, 20us)', linestyle='--', alpha=0.9)
ax2.plot(tps, llama4_nvlink_tpot, marker='d', markersize=8, color=c_nvlink, linewidth=2.5, label='Llama-4-Maverick (NVLink: 900 GB/s, 1.5us)', linestyle='-', alpha=0.95)

ax2.set_yscale('log')
ax2.set_title('Mean Time per Output Token (TPOT)', fontsize=14, fontweight='bold', pad=15, color='#222222')
ax2.set_ylabel('Latency (ms, Log Scale)', fontsize=12, fontweight='semibold', color='#444444')
ax2.set_xlabel('Tensor Parallelism (TP) Degree', fontsize=12, fontweight='semibold', color='#444444')
ax2.grid(True, which="both", linestyle='--', alpha=0.5)
ax2.tick_params(axis='both', labelsize=11)

# Annotate points
for i, val in enumerate(llama4_pcie_tpot):
    ax2.annotate(f"{val:.2f}ms", (i, val), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold', fontsize=9, color=c_pcie)
for i, val in enumerate(llama4_nvlink_tpot):
    ax2.annotate(f"{val:.2f}ms", (i, val), textcoords="offset points", xytext=(0,-15), ha='center', fontweight='bold', fontsize=9, color=c_nvlink)

# Add single legend for the entire figure
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=11, frameon=True, facecolor='#ffffff', edgecolor='#cccccc', bbox_to_anchor=(0.5, -0.08))

# General layout tuning
plt.suptitle('Llama-4-Maverick MoE serving latency scaling comparison on H100 GPU\nInterconnect Architecture Evaluation: PCIe vs NVLink (10-Request Dataset Workload)', fontsize=16, fontweight='bold', color='#111111', y=1.02)
plt.tight_layout()

# Save plot to artifacts directory
save_path = '/home/gpu2/.gemini/antigravity-ide/brain/f11ab176-e150-42df-8c77-945203395f18/llama4_scaling_comparison.png'
plt.savefig(save_path, bbox_inches='tight')
print(f"Plot successfully saved to {save_path}")
