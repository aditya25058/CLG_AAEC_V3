import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Sleek modern styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Premium color palette for each model
c_llama4 = '#E63946'    # Crimson Red
c_qwen3 = '#2A9D8F'     # Emerald Green/Teal
c_deepseek = '#1D3557'  # Deep Navy Blue

def parse_model_data(paths, tps):
    ttft_means = []
    tpot_means = []
    for tp in tps:
        path = paths[tp]
        if not os.path.exists(path):
            print(f"Warning: File {path} not found. Skipping TP={tp}")
            ttft_means.append(np.nan)
            tpot_means.append(np.nan)
            continue
        try:
            df = pd.read_csv(path)
            # convert from ns to ms
            ttft_means.append(df['TTFT'].mean() / 1e6)
            tpot_means.append(df['TPOT'].mean() / 1e6)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            ttft_means.append(np.nan)
            tpot_means.append(np.nan)
    return ttft_means, tpot_means

# --- MODEL DEFINITIONS ---

# 1. Llama-4 Maverick
llama4_pcie_paths = {
    1: "outputs/llama4_maverick_h100_tp1.csv",
    2: "outputs/llama4_maverick_h100_tp2_pcie.csv",
    4: "outputs/llama4_maverick_h100_tp4_pcie.csv",
    8: "outputs/llama4_maverick_h100_tp8_pcie.csv"
}
llama4_nvlink_paths = {
    1: "outputs/llama4_maverick_h100_tp1.csv",
    2: "outputs/llama4_maverick_h100_tp2_nvlink.csv",
    4: "outputs/llama4_maverick_h100_tp4_nvlink.csv",
    8: "outputs/llama4_maverick_h100_tp8_nvlink.csv"
}

# 2. Qwen-3 A22B
qwen3_pcie_paths = {
    1: "outputs/qwen3_a22b_h100_tp1.csv",
    2: "outputs/qwen3_a22b_h100_tp2_pcie.csv",
    4: "outputs/qwen3_a22b_h100_tp4_pcie.csv",
    8: "outputs/qwen3_a22b_h100_tp8_pcie.csv"
}
qwen3_nvlink_paths = {
    1: "outputs/qwen3_a22b_h100_tp1.csv",
    2: "outputs/qwen3_a22b_h100_tp2_nvlink.csv",
    4: "outputs/qwen3_a22b_h100_tp4_nvlink.csv",
    8: "outputs/qwen3_a22b_h100_tp8_nvlink.csv"
}

# 3. DeepSeek R1
deepseek_pcie_paths = {
    1: "outputs/deepseek_r1_h100_tp1_pcie_results.csv",
    2: "outputs/deepseek_r1_h100_tp2_pcie_results.csv",
    4: "outputs/deepseek_r1_h100_tp4_pcie_results.csv",
    8: "outputs/deepseek_r1_h100_tp8_pcie_results.csv"
}
deepseek_nvlink_paths = {
    1: "outputs/deepseek_r1_h100_tp1_nvlink_results.csv",
    2: "outputs/deepseek_r1_h100_tp2_nvlink_results.csv",
    4: "outputs/deepseek_r1_h100_tp4_nvlink_results.csv",
    8: "outputs/deepseek_r1_h100_tp8_nvlink_results.csv"
}

if __name__ == "__main__":
    tps = [1, 2, 4, 8]
    tps_labels = ['TP=1', 'TP=2', 'TP=4', 'TP=8']
    
    # Parse data for all three models
    l4_pcie_ttft, l4_pcie_tpot = parse_model_data(llama4_pcie_paths, tps)
    l4_nv_ttft, l4_nv_tpot = parse_model_data(llama4_nvlink_paths, tps)
    
    q3_pcie_ttft, q3_pcie_tpot = parse_model_data(qwen3_pcie_paths, tps)
    q3_nv_ttft, q3_nv_tpot = parse_model_data(qwen3_nvlink_paths, tps)
    
    ds_pcie_ttft, ds_pcie_tpot = parse_model_data(deepseek_pcie_paths, tps)
    ds_nv_ttft, ds_nv_tpot = parse_model_data(deepseek_nvlink_paths, tps)
    
    artifacts_dir = '/home/gpu2/.gemini/antigravity-ide/brain/f11ab176-e150-42df-8c77-945203395f18'
    
    # ----------------------------------------------------
    # Plot 1: TTFT Comparison (All Models PCIe vs NVLink)
    # ----------------------------------------------------
    fig_ttft, ax_ttft = plt.subplots(figsize=(12, 8), dpi=150)
    
    # Llama-4
    ax_ttft.plot(tps_labels, l4_pcie_ttft, marker='o', markersize=8, color=c_llama4, linewidth=2, 
                 label='Llama-4-Maverick (PCIe)', linestyle='--', alpha=0.9)
    ax_ttft.plot(tps_labels, l4_nv_ttft, marker='d', markersize=8, color=c_llama4, linewidth=2.5, 
                 label='Llama-4-Maverick (NVLink)', linestyle='-', alpha=0.95)
    
    # Qwen-3
    ax_ttft.plot(tps_labels, q3_pcie_ttft, marker='o', markersize=8, color=c_qwen3, linewidth=2, 
                 label='Qwen-3-235B-A22B (PCIe)', linestyle='--', alpha=0.9)
    ax_ttft.plot(tps_labels, q3_nv_ttft, marker='d', markersize=8, color=c_qwen3, linewidth=2.5, 
                 label='Qwen-3-235B-A22B (NVLink)', linestyle='-', alpha=0.95)
    
    # DeepSeek R1
    ax_ttft.plot(tps_labels, ds_pcie_ttft, marker='o', markersize=8, color=c_deepseek, linewidth=2, 
                 label='DeepSeek-R1-671B (PCIe)', linestyle='--', alpha=0.9)
    ax_ttft.plot(tps_labels, ds_nv_ttft, marker='d', markersize=8, color=c_deepseek, linewidth=2.5, 
                 label='DeepSeek-R1-671B (NVLink)', linestyle='-', alpha=0.95)
    
    ax_ttft.set_yscale('log')
    ax_ttft.set_title('Mean Time to First Token (TTFT) scaling Comparison\nInterconnect Performance Evaluation: PCIe vs NVLink on H100 GPUs', 
                       fontsize=15, fontweight='bold', pad=20, color='#111111')
    ax_ttft.set_ylabel('Latency (ms, Log Scale)', fontsize=13, fontweight='semibold', color='#333333')
    ax_ttft.set_xlabel('Tensor Parallelism (TP) Degree', fontsize=13, fontweight='semibold', color='#333333')
    ax_ttft.grid(True, which="both", linestyle='--', alpha=0.5)
    ax_ttft.tick_params(axis='both', labelsize=11)
    
    # Offset annotations to prevent overlaps
    for i in range(len(tps)):
        # Llama-4
        if not np.isnan(l4_pcie_ttft[i]):
            ax_ttft.annotate(f"{l4_pcie_ttft[i]:.1f}ms", (i, l4_pcie_ttft[i]), textcoords="offset points", xytext=(-25, 8), fontweight='bold', fontsize=8, color=c_llama4)
        if not np.isnan(l4_nv_ttft[i]):
            ax_ttft.annotate(f"{l4_nv_ttft[i]:.1f}ms", (i, l4_nv_ttft[i]), textcoords="offset points", xytext=(-25, -13), fontweight='bold', fontsize=8, color=c_llama4)
        # Qwen-3
        if not np.isnan(q3_pcie_ttft[i]):
            ax_ttft.annotate(f"{q3_pcie_ttft[i]:.1f}ms", (i, q3_pcie_ttft[i]), textcoords="offset points", xytext=(5, 8), fontweight='bold', fontsize=8, color=c_qwen3)
        if not np.isnan(q3_nv_ttft[i]):
            ax_ttft.annotate(f"{q3_nv_ttft[i]:.1f}ms", (i, q3_nv_ttft[i]), textcoords="offset points", xytext=(5, -13), fontweight='bold', fontsize=8, color=c_qwen3)
        # DeepSeek
        if not np.isnan(ds_pcie_ttft[i]):
            ax_ttft.annotate(f"{ds_pcie_ttft[i]:.1f}ms", (i, ds_pcie_ttft[i]), textcoords="offset points", xytext=(0, 12), ha='center', fontweight='bold', fontsize=8, color=c_deepseek)
        if not np.isnan(ds_nv_ttft[i]):
            ax_ttft.annotate(f"{ds_nv_ttft[i]:.1f}ms", (i, ds_nv_ttft[i]), textcoords="offset points", xytext=(0, -18), ha='center', fontweight='bold', fontsize=8, color=c_deepseek)

    ax_ttft.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#cccccc', fontsize=10)
    plt.tight_layout()
    ttft_path = os.path.join(artifacts_dir, 'all_models_ttft_comparison.png')
    plt.savefig(ttft_path, bbox_inches='tight')
    plt.close()
    print(f"TTFT Plot saved successfully to {ttft_path}")
    
    # ----------------------------------------------------
    # Plot 2: TPOT Comparison (All Models PCIe vs NVLink)
    # ----------------------------------------------------
    fig_tpot, ax_tpot = plt.subplots(figsize=(12, 8), dpi=150)
    
    # Llama-4
    ax_tpot.plot(tps_labels, l4_pcie_tpot, marker='o', markersize=8, color=c_llama4, linewidth=2, 
                 label='Llama-4-Maverick (PCIe)', linestyle='--', alpha=0.9)
    ax_tpot.plot(tps_labels, l4_nv_tpot, marker='d', markersize=8, color=c_llama4, linewidth=2.5, 
                 label='Llama-4-Maverick (NVLink)', linestyle='-', alpha=0.95)
    
    # Qwen-3
    ax_tpot.plot(tps_labels, q3_pcie_tpot, marker='o', markersize=8, color=c_qwen3, linewidth=2, 
                 label='Qwen-3-235B-A22B (PCIe)', linestyle='--', alpha=0.9)
    ax_tpot.plot(tps_labels, q3_nv_tpot, marker='d', markersize=8, color=c_qwen3, linewidth=2.5, 
                 label='Qwen-3-235B-A22B (NVLink)', linestyle='-', alpha=0.95)
    
    # DeepSeek R1
    ax_tpot.plot(tps_labels, ds_pcie_tpot, marker='o', markersize=8, color=c_deepseek, linewidth=2, 
                 label='DeepSeek-R1-671B (PCIe)', linestyle='--', alpha=0.9)
    ax_tpot.plot(tps_labels, ds_nv_tpot, marker='d', markersize=8, color=c_deepseek, linewidth=2.5, 
                 label='DeepSeek-R1-671B (NVLink)', linestyle='-', alpha=0.95)
    
    ax_tpot.set_yscale('log')
    ax_tpot.set_title('Mean Time per Output Token (TPOT) scaling Comparison\nInterconnect Performance Evaluation: PCIe vs NVLink on H100 GPUs', 
                       fontsize=15, fontweight='bold', pad=20, color='#111111')
    ax_tpot.set_ylabel('Latency (ms, Log Scale)', fontsize=13, fontweight='semibold', color='#333333')
    ax_tpot.set_xlabel('Tensor Parallelism (TP) Degree', fontsize=13, fontweight='semibold', color='#333333')
    ax_tpot.grid(True, which="both", linestyle='--', alpha=0.5)
    ax_tpot.tick_params(axis='both', labelsize=11)
    
    # Offset annotations to prevent overlaps
    for i in range(len(tps)):
        # Llama-4
        if not np.isnan(l4_pcie_tpot[i]):
            ax_tpot.annotate(f"{l4_pcie_tpot[i]:.2f}ms", (i, l4_pcie_tpot[i]), textcoords="offset points", xytext=(-25, 8), fontweight='bold', fontsize=8, color=c_llama4)
        if not np.isnan(l4_nv_tpot[i]):
            ax_tpot.annotate(f"{l4_nv_tpot[i]:.2f}ms", (i, l4_nv_tpot[i]), textcoords="offset points", xytext=(-25, -13), fontweight='bold', fontsize=8, color=c_llama4)
        # Qwen-3
        if not np.isnan(q3_pcie_tpot[i]):
            ax_tpot.annotate(f"{q3_pcie_tpot[i]:.2f}ms", (i, q3_pcie_tpot[i]), textcoords="offset points", xytext=(5, 8), fontweight='bold', fontsize=8, color=c_qwen3)
        if not np.isnan(q3_nv_tpot[i]):
            ax_tpot.annotate(f"{q3_nv_tpot[i]:.2f}ms", (i, q3_nv_tpot[i]), textcoords="offset points", xytext=(5, -13), fontweight='bold', fontsize=8, color=c_qwen3)
        # DeepSeek
        if not np.isnan(ds_pcie_tpot[i]):
            ax_tpot.annotate(f"{ds_pcie_tpot[i]:.2f}ms", (i, ds_pcie_tpot[i]), textcoords="offset points", xytext=(0, 12), ha='center', fontweight='bold', fontsize=8, color=c_deepseek)
        if not np.isnan(ds_nv_tpot[i]):
            ax_tpot.annotate(f"{ds_nv_tpot[i]:.2f}ms", (i, ds_nv_tpot[i]), textcoords="offset points", xytext=(0, -18), ha='center', fontweight='bold', fontsize=8, color=c_deepseek)

    ax_tpot.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#cccccc', fontsize=10)
    plt.tight_layout()
    tpot_path = os.path.join(artifacts_dir, 'all_models_tpot_comparison.png')
    plt.savefig(tpot_path, bbox_inches='tight')
    plt.close()
    print(f"TPOT Plot saved successfully to {tpot_path}")
