# evaluation/scripts/e20_honest_overlap_analysis.py
# Simulates the honest latency waterfall and overlap fraction under different interconnect configurations.
import os
import json
import math

# Qwen3-30B parameters
H = 2048
NL = 48

def run_honest_analysis():
    # Sweep space
    miss_sizes = [8, 16, 32, 64]
    batches = [1, 4, 8, 16]
    interconnects = {
        "PCIe_Gen4": 16.0,
        "PCIe_Gen5_x8": 32.0,
        "PCIe_Gen5_x16": 64.0,
        "CXL_3.0": 128.0
    }
    
    COLUMN_SIZE_BYTES = 3 * H * 2  # 12,288 bytes
    
    results = {}
    
    for name, bw in interconnects.items():
        results[name] = {}
        print(f"\n==========================================")
        print(f"INTERCONNECT: {name} ({bw} GB/s)")
        print(f"==========================================")
        print(f"{'Batch':<5} | {'Miss':<5} | {'Payload':<8} | {'Transfer':<8} | {'Overlap Window':<14} | {'Exposed':<8} | {'Overlap'}")
        print(f"{'Size':<5} | {'Cols':<5} | {'(MB)':<8} | {'(ms)':<8} | {'(ms)':<14} | {'Stall (ms)':<8} | {'Fraction'}")
        print("-" * 75)
        
        for B in batches:
            results[name][str(B)] = []
            
            # Autoregressive attention compute scales slightly with batch size
            t_attn_ms = (100.0 + 1.5 * (B - 1)) / 1000.0
            # FFN Phase 1 compute time scales with batch size
            t_ffn1_ms = (35.8 + 10.0 * (B - 1)) / 1000.0
            
            overlap_window_ms = t_attn_ms + t_ffn1_ms
            
            for M in miss_sizes:
                # Union active experts factor
                if B == 1:
                    union_experts = 8
                elif B == 4:
                    union_experts = 25
                elif B == 8:
                    union_experts = 40
                else:
                    union_experts = 64
                    
                total_cols = union_experts * M
                payload_bytes = total_cols * COLUMN_SIZE_BYTES
                payload_mb = payload_bytes / (1024**2)
                
                # Achieved transfer latency with PCIe overhead
                # We model PCIe launch overhead as 50 us (0.05 ms)
                launch_overhead_ms = 0.05
                t_copy_ms = (payload_bytes / (bw * 1e9)) * 1000.0 + launch_overhead_ms
                
                exposed_stall_ms = max(0.0, t_copy_ms - overlap_window_ms)
                
                overlap_fraction = max(0.0, 1.0 - (exposed_stall_ms / t_copy_ms))
                
                results[name][str(B)].append({
                    "miss_cols": M,
                    "payload_mb": payload_mb,
                    "copy_time_ms": t_copy_ms,
                    "overlap_window_ms": overlap_window_ms,
                    "exposed_stall_ms": exposed_stall_ms,
                    "overlap_fraction": overlap_fraction
                })
                
                print(f"{B:<5} | {M:<5} | {payload_mb:7.3f} | {t_copy_ms:8.4f} | {overlap_window_ms:14.4f} | {exposed_stall_ms:10.4f} | {overlap_fraction*100:6.1f}%")
                
    out_dir = "/home/palakm/MoEServingSim/evaluation/results/e20_overlap"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "honest_overlap_results.json"), "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_honest_analysis()
