# evaluation/scripts/e06_bandwidth_utilization.py
import os
import json
import sqlite3
import numpy as np

MODELS = {
    "qwen3_30b": {
        "db_path": "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/qwen3_30b_real_v2.db",
        "hidden_size": 2048,
        "active_experts": 8
    },
    "deepseek_v2_lite": {
        "db_path": "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/deepseek_lite_real.db",
        "hidden_size": 2048,
        "active_experts": 6
    }
}

def analyze_bandwidth_utilization(model_name: str, spec: dict):
    db_path = spec["db_path"]
    H = spec["hidden_size"]
    
    if not os.path.exists(db_path):
        print(f"Skipping {model_name} (database not found)")
        return
        
    print(f"Analyzing bandwidth utilization and transfer overlap for {model_name}...")
    
    # 1. Database query to load activations
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT energy_k_90, energy_k_50
        FROM activations
    """)
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        print("  No activation records found.")
        return
        
    # Column size in bytes = 3 matrices (gate_proj, up_proj, down_proj) * hidden_size * 2 bytes (bf16)
    column_bytes = 3 * H * 2
    
    # Bandwidth targets (GB/s)
    pcie_bw = 64.0
    nvlink_bw = 450.0
    
    # Standard attention computation window (latency in microseconds)
    # Typically 50 to 150 us on H100 depending on KV cache states and head counts
    attn_latencies_us = [50.0, 100.0, 150.0]
    
    results = {}
    
    # Let's assume on average we miss 'M' columns per active expert.
    # We sweep M from 16 to 128 to show bandwidth stress curves.
    miss_counts = [16, 32, 64, 128]
    
    for M in miss_counts:
        # Total transfer size per layer (active_experts * missed columns * column_bytes)
        active_exps = spec["active_experts"]
        total_transfer_bytes = active_exps * M * column_bytes
        total_transfer_kb = total_transfer_bytes / 1024.0
        
        # Calculate raw transfer times (microseconds)
        t_transfer_pcie = (total_transfer_bytes / (pcie_bw * 1e9)) * 1e6
        t_transfer_nvlink = (total_transfer_bytes / (nvlink_bw * 1e9)) * 1e6
        
        results[f"miss_{M}"] = {
            "transfer_kb": total_transfer_kb,
            "t_pcie_us": t_transfer_pcie,
            "t_nvlink_us": t_transfer_nvlink,
            "overlap_pcie": {},
            "overlap_nvlink": {}
        }
        
        print(f"\n  Missed Columns per Expert: {M} (Total payload: {total_transfer_kb:.2f} KB)")
        print(f"    Raw Transfer Latency -> PCIe Gen5: {t_transfer_pcie:.3f} us | NVLink: {t_transfer_nvlink:.3f} us")
        
        for t_attn in attn_latencies_us:
            # Overlap metrics
            pcie_hiding_ratio = min(t_transfer_pcie, t_attn) / t_transfer_pcie
            pcie_stall = max(0.0, t_transfer_pcie - t_attn)
            
            nvlink_hiding_ratio = min(t_transfer_nvlink, t_attn) / t_transfer_nvlink
            nvlink_stall = max(0.0, t_transfer_nvlink - t_attn)
            
            results[f"miss_{M}"]["overlap_pcie"][str(t_attn)] = {
                "hiding_ratio": pcie_hiding_ratio,
                "stall_us": pcie_stall
            }
            results[f"miss_{M}"]["overlap_nvlink"][str(t_attn)] = {
                "hiding_ratio": nvlink_hiding_ratio,
                "stall_us": nvlink_stall
            }
            
            print(f"    For Attention Window = {t_attn:.1f} us:")
            print(f"      PCIe Gen5 -> Hiding Ratio: {pcie_hiding_ratio*100:6.2f}% | Exposed Stall: {pcie_stall:6.2f} us")
            print(f"      NVLink    -> Hiding Ratio: {nvlink_hiding_ratio*100:6.2f}% | Exposed Stall: {nvlink_stall:6.2f} us")

    out_dir = f"/home/palakm/MoEServingSim/evaluation/results/e06_bandwidth/{model_name}"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "bandwidth_profile.json"), "w") as f:
        json.dump(results, f, indent=4)

def main():
    for name, spec in MODELS.items():
        analyze_bandwidth_utilization(name, spec)

if __name__ == "__main__":
    main()
