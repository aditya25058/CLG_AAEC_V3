import os
import json
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

DB_PATH = "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/qwen3_30b_real_v2.db"
OUTPUT_DIR = "/home/palakm/MoEServingSim/qwen3_30b_plots"

def load_traces():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at {DB_PATH}")
    
    print("Loading sequential execution traces from DB...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT prompt_id, token_pos, layer, expert_id, active_indices, energy_k_50 
        FROM activations 
        ORDER BY prompt_id, token_pos, layer
    """)
    rows = cursor.fetchall()
    conn.close()
    
    evaluation_db = {}
    
    prompt_ids = sorted(list(set(row[0] for row in rows)))
    split_idx = len(prompt_ids) // 2
    eval_prompts = set(prompt_ids[split_idx:])
    
    for row in rows:
        p_id, t_pos, layer, exp_id, indices_str, k50 = row
        indices = json.loads(indices_str)[:k50]
        active_set = set(indices)
        
        if p_id in eval_prompts:
            if p_id not in evaluation_db:
                evaluation_db[p_id] = {}
            if t_pos not in evaluation_db[p_id]:
                evaluation_db[p_id][t_pos] = {}
            evaluation_db[p_id][t_pos][layer] = (exp_id, active_set)
            
    return evaluation_db

def run_pipeline_simulation(evaluation_db, batch_size=8, cache_size=32, use_adetr=True, gpu_flops=300e12):
    # Hardware bandwidths
    BW_NVLINK = 450e9    # 450 GB/s
    BW_PCIE = 64e9       # 64 GB/s (PCIe Gen5 x16)
    BW_RDMA = 12.5e9     # 12.5 GB/s (100G IB)
    
    COLUMN_SIZE_BYTES = 5120 * 2 * 3
    
    # ADETR vs No-ADETR variables:
    if use_adetr:
        COMPUTE_EFFICIENCY = 0.40  # 40% efficiency for contiguous coalesced GEMM
        DMA_LAUNCH_LATENCY_US = 0.5  # 0.5 us per contiguous transfer slice (we assume 1 contiguous slice per tier)
    else:
        COMPUTE_EFFICIENCY = 0.15  # 15% efficiency due to memory striding/gather overheads
        DMA_LAUNCH_LATENCY_US = 0.5  # 0.5 us per INDIVIDUAL column (strided copies)
        
    effective_flops = gpu_flops * COMPUTE_EFFICIENCY
    
    eval_prompt_ids = sorted(evaluation_db.keys())
    
    gpu_caches = {l: OrderedDict() for l in range(48)}
    t3_caches = {l: OrderedDict() for l in range(48)}
    
    total_latency_us = 0.0
    total_steps = 0
    
    num_batches = len(eval_prompt_ids) // batch_size
    if num_batches == 0:
        num_batches = 1
        
    for b_idx in range(num_batches):
        batch_prompts = eval_prompt_ids[b_idx * batch_size : min(len(eval_prompt_ids), (b_idx + 1) * batch_size)]
        if not batch_prompts:
            continue
            
        max_len = max(len(evaluation_db[p_id]) for p_id in batch_prompts)
        
        for t_pos in range(max_len):
            total_steps += 1
            
            for l in range(48):
                # Collect active columns
                batch_active_keys = set()
                for p_id in batch_prompts:
                    t_positions = sorted(evaluation_db[p_id].keys())
                    if t_pos < len(t_positions):
                        curr_t = t_positions[t_pos]
                        if l in evaluation_db[p_id][curr_t]:
                            exp_id, active_cols = evaluation_db[p_id][curr_t][l]
                            for col in active_cols:
                                batch_active_keys.add((exp_id, col))
                                
                if not batch_active_keys:
                    continue
                    
                t1_keys = set()
                t2_keys = set()
                t3_keys = set()
                t4_keys = set()
                t5_keys = set()
                
                t2_cache = gpu_caches[l]
                t3_cache = t3_caches[l]
                
                t2_capacity = cache_size * 128
                t3_capacity = cache_size * 128
                
                for key in batch_active_keys:
                    exp_id, col = key
                    if col < 8:
                        t1_keys.add(key)
                    elif key in t2_cache:
                        t2_keys.add(key)
                    elif key in t3_cache:
                        t3_keys.add(key)
                    elif col < 400:
                        t4_keys.add(key)
                    else:
                        t5_keys.add(key)
                        
                # Update caches
                for key in batch_active_keys:
                    exp_id, col = key
                    if col >= 8:
                        if key in t2_cache:
                            t2_cache.move_to_end(key)
                        else:
                            if len(t2_cache) >= t2_capacity:
                                evicted = t2_cache.popitem(last=False)[0]
                                if len(t3_cache) >= t3_capacity:
                                    t3_cache.popitem(last=False)
                                t3_cache[evicted] = True
                            t2_cache[key] = True
                            
                num_local = len(t1_keys) + len(t2_keys)
                num_t3 = len(t3_keys)
                num_t4 = len(t4_keys)
                num_t5 = len(t5_keys)
                num_total = len(batch_active_keys)
                
                # Compute MFLOPs
                mflops_local = (2.0 * 5120.0 * num_local) / 1e6
                mflops_phase2 = (2.0 * 5120.0 * (num_t3 + num_t4 + num_t5)) / 1e6
                
                t_comp_local = (mflops_local * 1e6 / effective_flops) * 1e6
                t_comp_phase2 = (mflops_phase2 * 1e6 / effective_flops) * 1e6
                
                # Transfer payloads and launch latency
                # With ADETR: only 1 DMA launch latency per active tier
                # Without ADETR: 1 DMA launch latency per individual column
                if use_adetr:
                    latency_t3 = DMA_LAUNCH_LATENCY_US if num_t3 > 0 else 0.0
                    latency_t4 = DMA_LAUNCH_LATENCY_US if num_t4 > 0 else 0.0
                    latency_t5 = DMA_LAUNCH_LATENCY_US if num_t5 > 0 else 0.0
                else:
                    latency_t3 = num_t3 * DMA_LAUNCH_LATENCY_US
                    latency_t4 = num_t4 * DMA_LAUNCH_LATENCY_US
                    latency_t5 = num_t5 * DMA_LAUNCH_LATENCY_US
                    
                t_trans_t3 = ((num_t3 * COLUMN_SIZE_BYTES) / BW_NVLINK) * 1e6 + latency_t3
                t_trans_t4 = ((num_t4 * COLUMN_SIZE_BYTES) / BW_PCIE) * 1e6 + latency_t4
                t_trans_t5 = ((num_t5 * COLUMN_SIZE_BYTES) / BW_RDMA) * 1e6 + latency_t5
                
                # Overlapped Wait Time
                longest_transfer = max(t_trans_t3, t_trans_t4, t_trans_t5)
                t_wait = max(0.0, longest_transfer - t_comp_local)
                
                # Total Latency with overlap
                t_overhead_sync = 1.0  # 1 us synchronization overhead
                latency = t_comp_local + t_wait + t_comp_phase2 + t_overhead_sync
                total_latency_us += latency
                
    return (total_latency_us / 1000.0) / max(1, total_steps)

def main():
    print("==================================================================")
    print("Simulating AAEC Pipelines: WITH ADETR vs. WITHOUT ADETR...")
    print("==================================================================")
    
    evaluation_db = load_traces()
    
    batch_sizes = [1, 2, 4, 8, 12, 25]
    cache_sizes = [8, 16, 32, 64]
    
    results = {}
    
    for cs in cache_sizes:
        results[cs] = {"with_adetr": [], "without_adetr": [], "speedup": []}
        for bs in batch_sizes:
            lat_with = run_pipeline_simulation(evaluation_db, batch_size=bs, cache_size=cs, use_adetr=True)
            lat_without = run_pipeline_simulation(evaluation_db, batch_size=bs, cache_size=cs, use_adetr=False)
            speedup = lat_without / lat_with
            
            results[cs]["with_adetr"].append((bs, lat_with))
            results[cs]["without_adetr"].append((bs, lat_without))
            results[cs]["speedup"].append((bs, speedup))
            
            print(f"Cache Size = {cs:2d} | Batch Size = {bs:2d} | Without ADETR = {lat_without:.4f} ms | With ADETR = {lat_with:.4f} ms | Speedup = {speedup:.2f}x")
            
    # Save results
    json_path = os.path.join(OUTPUT_DIR, "aaec_adetr_comparison.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved comparison results to: {json_path}")
    
    # Plotting comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Panel 1: Absolute Latency comparison at Cache Size = 32
    bs_list = [item[0] for item in results[32]["with_adetr"]]
    lat_with = [item[1] for item in results[32]["with_adetr"]]
    lat_without = [item[1] for item in results[32]["without_adetr"]]
    
    ax1.plot(bs_list, lat_without, 'o-', color="red", linewidth=2.5, label="AAEC Pipeline (Without ADETR)")
    ax1.plot(bs_list, lat_with, 's-', color="green", linewidth=2.5, label="AAEC Pipeline (With ADETR)")
    ax1.set_xscale('log', base=2)
    ax1.set_xticks(batch_sizes, labels=[str(b) for b in batch_sizes])
    ax1.set_xlabel("Batch Size (Tokens)")
    ax1.set_ylabel("FFN Latency per Token Step (ms)")
    ax1.set_title("Absolute Latency Comparison (Cache Size = 32)")
    ax1.grid(True, which="both", ls="--", alpha=0.3)
    ax1.legend()
    
    # Panel 2: ADETR Speedup vs Batch Size across Cache Sizes
    colors = {8: "red", 16: "orange", 32: "green", 64: "blue"}
    for cs in cache_sizes:
        bs_list = [item[0] for item in results[cs]["speedup"]]
        speedup_list = [item[1] for item in results[cs]["speedup"]]
        ax2.plot(bs_list, speedup_list, 'o-', color=colors[cs], linewidth=2.5, label=f"Cache Size = {cs}")
        
    ax2.set_xscale('log', base=2)
    ax2.set_xticks(batch_sizes, labels=[str(b) for b in batch_sizes])
    ax2.set_xlabel("Batch Size (Tokens)")
    ax2.set_ylabel("Speedup factor (x)")
    ax2.set_title("ADETR Speedup vs. Batch Size")
    ax2.grid(True, which="both", ls="--", alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "aaec_adetr_comparison.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved comparison plots to: {plot_path}")

if __name__ == "__main__":
    main()
