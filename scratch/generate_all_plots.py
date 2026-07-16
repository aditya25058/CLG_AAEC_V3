import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Sleek modern styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Premium color palette
c_pcie = '#E63946'      # Energetic Crimson Red
c_nvlink = '#1D3557'    # Sleek Deep Navy

def parse_model_data(paths, tps):
    ttft_means = []
    tpot_means = []
    latency_means = []
    for tp in tps:
        path = paths[tp]
        if not os.path.exists(path):
            print(f"Warning: File {path} not found. Skipping TP={tp}")
            ttft_means.append(np.nan)
            tpot_means.append(np.nan)
            latency_means.append(np.nan)
            continue
        try:
            df = pd.read_csv(path)
            # convert from ns to ms
            ttft_means.append(df['TTFT'].mean() / 1e6)
            tpot_means.append(df['TPOT'].mean() / 1e6)
            latency_means.append(df['latency'].mean() / 1e6)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            ttft_means.append(np.nan)
            tpot_means.append(np.nan)
            latency_means.append(np.nan)
    return ttft_means, tpot_means, latency_means

def generate_comparison_plots(model_id, model_name, pcie_paths, nvlink_paths, tps, save_filename):
    tps_labels = [f"TP={tp}" for tp in tps]
    
    # Parse data
    pcie_ttft, pcie_tpot, pcie_lat = parse_model_data(pcie_paths, tps)
    nv_ttft, nv_tpot, nv_lat = parse_model_data(nvlink_paths, tps)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=150)
    
    # ----------------- TTFT Plot -----------------
    ax1.plot(tps_labels, pcie_ttft, marker='o', markersize=8, color=c_pcie, linewidth=2.5, 
             label=f'{model_name} (PCIe: 16 GB/s, 20us)', linestyle='--', alpha=0.9)
    ax1.plot(tps_labels, nv_ttft, marker='d', markersize=8, color=c_nvlink, linewidth=2.5, 
             label=f'{model_name} (NVLink: 900 GB/s, 1.5us)', linestyle='-', alpha=0.95)
    
    ax1.set_yscale('log')
    ax1.set_title('Mean Time to First Token (TTFT)', fontsize=14, fontweight='bold', pad=15, color='#222222')
    ax1.set_ylabel('Latency (ms, Log Scale)', fontsize=12, fontweight='semibold', color='#444444')
    ax1.set_xlabel('Tensor Parallelism (TP) Degree', fontsize=12, fontweight='semibold', color='#444444')
    ax1.grid(True, which="both", linestyle='--', alpha=0.5)
    ax1.tick_params(axis='both', labelsize=11)
    
    # Annotate TTFT points
    for i, val in enumerate(pcie_ttft):
        if not np.isnan(val):
            ax1.annotate(f"{val:.2f}ms", (i, val), textcoords="offset points", xytext=(0,10), 
                         ha='center', fontweight='bold', fontsize=9, color=c_pcie)
    for i, val in enumerate(nv_ttft):
        if not np.isnan(val):
            ax1.annotate(f"{val:.2f}ms", (i, val), textcoords="offset points", xytext=(0,-15), 
                         ha='center', fontweight='bold', fontsize=9, color=c_nvlink)
            
    # ----------------- TPOT Plot -----------------
    ax2.plot(tps_labels, pcie_tpot, marker='o', markersize=8, color=c_pcie, linewidth=2.5, 
             label=f'{model_name} (PCIe: 16 GB/s, 20us)', linestyle='--', alpha=0.9)
    ax2.plot(tps_labels, nv_tpot, marker='d', markersize=8, color=c_nvlink, linewidth=2.5, 
             label=f'{model_name} (NVLink: 900 GB/s, 1.5us)', linestyle='-', alpha=0.95)
    
    ax2.set_yscale('log')
    ax2.set_title('Mean Time per Output Token (TPOT)', fontsize=14, fontweight='bold', pad=15, color='#222222')
    ax2.set_ylabel('Latency (ms, Log Scale)', fontsize=12, fontweight='semibold', color='#444444')
    ax2.set_xlabel('Tensor Parallelism (TP) Degree', fontsize=12, fontweight='semibold', color='#444444')
    ax2.grid(True, which="both", linestyle='--', alpha=0.5)
    ax2.tick_params(axis='both', labelsize=11)
    
    # Annotate TPOT points
    for i, val in enumerate(pcie_tpot):
        if not np.isnan(val):
            ax2.annotate(f"{val:.2f}ms", (i, val), textcoords="offset points", xytext=(0,10), 
                         ha='center', fontweight='bold', fontsize=9, color=c_pcie)
    for i, val in enumerate(nv_tpot):
        if not np.isnan(val):
            ax2.annotate(f"{val:.2f}ms", (i, val), textcoords="offset points", xytext=(0,-15), 
                         ha='center', fontweight='bold', fontsize=9, color=c_nvlink)
            
    # Legends
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=11, frameon=True, 
               facecolor='#ffffff', edgecolor='#cccccc', bbox_to_anchor=(0.5, -0.08))
    
    plt.suptitle(f'{model_name} serving latency scaling comparison on H100 GPU\nInterconnect Architecture Evaluation: PCIe vs NVLink (10-Request Workload)', 
                 fontsize=16, fontweight='bold', color='#111111', y=1.02)
    plt.tight_layout()
    
    # Save path in artifacts directory
    artifacts_dir = '/home/gpu2/.gemini/antigravity-ide/brain/f11ab176-e150-42df-8c77-945203395f18'
    save_path = os.path.join(artifacts_dir, save_filename)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Plot successfully saved to {save_path}")

# --- MODEL DEFINITIONS ---

# 1. Llama-4 Maverick
llama4_pcie = {
    1: "outputs/llama4_maverick_h100_tp1.csv",
    2: "outputs/llama4_maverick_h100_tp2_pcie.csv",
    4: "outputs/llama4_maverick_h100_tp4_pcie.csv",
    8: "outputs/llama4_maverick_h100_tp8_pcie.csv"
}
llama4_nvlink = {
    1: "outputs/llama4_maverick_h100_tp1.csv",
    2: "outputs/llama4_maverick_h100_tp2_nvlink.csv",
    4: "outputs/llama4_maverick_h100_tp4_nvlink.csv",
    8: "outputs/llama4_maverick_h100_tp8_nvlink.csv"
}

# 2. Qwen-3 A22B
qwen3_pcie = {
    1: "outputs/qwen3_a22b_h100_tp1.csv",
    2: "outputs/qwen3_a22b_h100_tp2_pcie.csv",
    4: "outputs/qwen3_a22b_h100_tp4_pcie.csv",
    8: "outputs/qwen3_a22b_h100_tp8_pcie.csv"
}
qwen3_nvlink = {
    1: "outputs/qwen3_a22b_h100_tp1.csv",
    2: "outputs/qwen3_a22b_h100_tp2_nvlink.csv",
    4: "outputs/qwen3_a22b_h100_tp4_nvlink.csv",
    8: "outputs/qwen3_a22b_h100_tp8_nvlink.csv"
}

# 3. Kimi-K2 (TP=8 is excluded/exception)
kimi_pcie = {
    1: "outputs/kimi_k2_h100_tp1.csv",
    2: "outputs/kimi_k2_h100_tp2_pcie.csv",
    4: "outputs/kimi_k2_h100_tp4_pcie.csv"
}
kimi_nvlink = {
    1: "outputs/kimi_k2_h100_tp1.csv",
    2: "outputs/kimi_k2_h100_tp2_nvlink.csv",
    4: "outputs/kimi_k2_h100_tp4_nvlink.csv"
}

# 4. DeepSeek R1
deepseek_pcie = {
    1: "outputs/deepseek_r1_h100_tp1_pcie_results.csv",
    2: "outputs/deepseek_r1_h100_tp2_pcie_results.csv",
    4: "outputs/deepseek_r1_h100_tp4_pcie_results.csv",
    8: "outputs/deepseek_r1_h100_tp8_pcie_results.csv"
}
deepseek_nvlink = {
    1: "outputs/deepseek_r1_h100_tp1_nvlink_results.csv",
    2: "outputs/deepseek_r1_h100_tp2_nvlink_results.csv",
    4: "outputs/deepseek_r1_h100_tp4_nvlink_results.csv",
    8: "outputs/deepseek_r1_h100_tp8_nvlink_results.csv"
}

if __name__ == "__main__":
    print("Generating Latency Scaling Comparison Plots...")
    
    # 1. Llama-4 Maverick
    print("\n--- Generating plots for Llama-4 Maverick ---")
    generate_comparison_plots("llama4", "Llama-4-Maverick-17B", llama4_pcie, llama4_nvlink, [1, 2, 4, 8], "llama4_scaling_comparison.png")
    
    # 2. Qwen-3 A22B
    print("\n--- Generating plots for Qwen-3 A22B ---")
    generate_comparison_plots("qwen3", "Qwen-3-235B-A22B", qwen3_pcie, qwen3_nvlink, [1, 2, 4, 8], "qwen3_scaling_comparison.png")
    
    # 3. DeepSeek R1
    print("\n--- Generating plots for DeepSeek R1 ---")
    generate_comparison_plots("deepseek", "DeepSeek-R1-671B", deepseek_pcie, deepseek_nvlink, [1, 2, 4, 8], "deepseek_scaling_comparison.png")
    
    print("\nAll comparison plots generated successfully inside artifacts directory!")
