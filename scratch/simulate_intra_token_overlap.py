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

def run_intra_token_overlap_simulation(evaluation_db, batch_size=8, cache_size=32, gpu_flops=300e12):
    # Hardware bandwidths (from Image 2)
    BW_NVLINK = 450e9    # 450 GB/s
    BW_PCIE = 64e9       # 64 GB/s (PCIe Gen5 x16)
    BW_RDMA = 12.5e9     # 100 Gbps InfiniBand (12.5 GB/s)
    
    COLUMN_SIZE_BYTES = 5120 * 2 * 3  # BF16 for Gate, Up, Down matrices
    COMPUTE_EFFICIENCY = 0.40  # 40% of peak Tensor Core FLOPS
    effective_flops = gpu_flops * COMPUTE_EFFICIENCY
    
    eval_prompt_ids = sorted(evaluation_db.keys())
    
    # Precompute static column tuples for each expert
    static_cols_cache = {}
    for l in range(48):
        for e in range(128):
            static_cols_cache[(l, e)] = [(e, col) for col in range(768)]
            
    layer_cache_capacity = cache_size * 128
    gpu_caches = {l: OrderedDict() for l in range(48)}
    t3_caches = {l: OrderedDict() for l in range(48)}
    
    total_latency_overlapped_us = 0.0
    total_latency_sequential_us = 0.0
    total_steps = 0
    
    # We group tokens into batches of size 'batch_size'
    num_batches = len(eval_prompt_ids) // batch_size
    if num_batches == 0:
        num_batches = 1
        
    for b_idx in range(num_batches):
        batch_prompts = eval_prompt_ids[b_idx * batch_size : min(len(eval_prompt_ids), (b_idx + 1) * batch_size)]
        if not batch_prompts:
            continue
            
        # Find maximum length among these prompts
        max_len = max(len(evaluation_db[p_id]) for p_id in batch_prompts)
        
        for t_pos in range(max_len):
            total_steps += 1
            
            # Layer-by-layer execution
            for l in range(48):
                # Collect active columns for all tokens in this batch
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
                    
                # Classify batch active keys into tiers:
                t1_keys = set()
                t2_keys = set()
                t3_keys = set()
                t4_keys = set()
                t5_keys = set()
                
                t2_cache = gpu_caches[l]
                t3_cache = t3_caches[l]
                
                t2_capacity = cache_size * 128
                t3_capacity = cache_size * 128  # peer cache size
                
                for key in batch_active_keys:
                    exp_id, col = key
                    # T1: top 1% (columns 0 to 7)
                    if col < 8:
                        t1_keys.add(key)
                    # T2: local HBM LRU cache
                    elif key in t2_cache:
                        t2_keys.add(key)
                    # T3: peer GPU NVLink cache
                    elif key in t3_cache:
                        t3_keys.add(key)
                    # T4: host DRAM (columns 8 to 400)
                    elif col < 400:
                        t4_keys.add(key)
                    # T5: remote DRAM
                    else:
                        t5_keys.add(key)
                        
                # Update caches with the newly active keys
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
                            
                # Compute execution times and transfer payloads
                num_local = len(t1_keys) + len(t2_keys)
                num_t3 = len(t3_keys)
                num_t4 = len(t4_keys)
                num_t5 = len(t5_keys)
                num_total = len(batch_active_keys)
                
                # Compute MFLOPs: 2 ops (FMA) * Hidden Size (5120) * num_columns
                mflops_local = (2.0 * 5120.0 * num_local) / 1e6
                mflops_phase2 = (2.0 * 5120.0 * (num_t3 + num_t4 + num_t5)) / 1e6
                mflops_total = (2.0 * 5120.0 * num_total) / 1e6
                
                # Compute Times (us)
                t_comp_local = (mflops_local * 1e6 / effective_flops) * 1e6
                t_comp_phase2 = (mflops_phase2 * 1e6 / effective_flops) * 1e6
                t_comp_total = (mflops_total * 1e6 / effective_flops) * 1e6
                
                # Transfer Sizes (Bytes)
                size_t3 = num_t3 * COLUMN_SIZE_BYTES
                size_t4 = num_t4 * COLUMN_SIZE_BYTES
                size_t5 = num_t5 * COLUMN_SIZE_BYTES
                
                # Transfer Times (us)
                t_trans_t3 = (size_t3 / BW_NVLINK) * 1e6 if num_t3 > 0 else 0.0
                t_trans_t4 = (size_t4 / BW_PCIE) * 1e6 if num_t4 > 0 else 0.0
                t_trans_t5 = (size_t5 / BW_RDMA) * 1e6 if num_t5 > 0 else 0.0
                
                # Overlapped Wait Time
                longest_transfer = max(t_trans_t3, t_trans_t4, t_trans_t5)
                t_wait = max(0.0, longest_transfer - t_comp_local)
                
                # Total Latency with overlap
                t_overhead_sync = 1.0  # 1 us synchronization overhead
                latency_overlapped = t_comp_local + t_wait + t_comp_phase2 + t_overhead_sync
                
                # Total Latency sequential (no overlap)
                latency_sequential = t_trans_t3 + t_trans_t4 + t_trans_t5 + t_comp_total + t_overhead_sync
                
                total_latency_overlapped_us += latency_overlapped
                total_latency_sequential_us += latency_sequential
                
    avg_overlapped_ms = (total_latency_overlapped_us / 1000.0) / max(1, total_steps)
    avg_sequential_ms = (total_latency_sequential_us / 1000.0) / max(1, total_steps)
    speedup = avg_sequential_ms / avg_overlapped_ms
    
    return {
        "avg_overlapped_ms": avg_overlapped_ms,
        "avg_sequential_ms": avg_sequential_ms,
        "speedup": speedup
    }

def main():
    print("==================================================================")
    print("Simulating Predictor-less Intra-Token Overlap Serving Pipeline...")
    print("==================================================================")
    
    evaluation_db = load_traces()
    
    # Adjust batch sizes to fit within the 25 evaluation prompts
    batch_sizes = [1, 2, 4, 8, 12, 25]
    cache_sizes = [8, 16, 32, 64]
    
    results = {}
    
    for cs in cache_sizes:
        results[cs] = []
        for bs in batch_sizes:
            res = run_intra_token_overlap_simulation(evaluation_db, batch_size=bs, cache_size=cs)
            results[cs].append((bs, res["avg_overlapped_ms"], res["avg_sequential_ms"], res["speedup"]))
            print(f"Cache Size = {cs:2d} | Batch Size = {bs:2d} | Sequential = {res['avg_sequential_ms']:.4f} ms | Overlapped = {res['avg_overlapped_ms']:.4f} ms | Speedup = {res['speedup']:.2f}x")
            
    # Save raw results
    json_path = os.path.join(OUTPUT_DIR, "aaec_intra_token_overlap.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved raw results to: {json_path}")
    
    # Plot speedup vs batch size for different cache sizes
    plt.figure(figsize=(8, 5))
    colors = {8: "red", 16: "orange", 32: "green", 64: "blue"}
    for cs in cache_sizes:
        bs_list = [item[0] for item in results[cs]]
        speedup_list = [item[3] for item in results[cs]]
        plt.plot(bs_list, speedup_list, 'o-', color=colors[cs], linewidth=2, label=f"Cache Capacity = {cs} cols")
        
    plt.xscale('log', base=2)
    plt.xticks(batch_sizes, labels=[str(b) for b in batch_sizes])
    plt.xlabel("Execution Batch Size (Tokens)")
    plt.ylabel("Execution Overlap Speedup (x)")
    plt.title("Intra-Token Overlap Speedup vs Batch Size (Predictor-less)")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plot_path = os.path.join(OUTPUT_DIR, "aaec_intra_token_overlap.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved speedup plot to: {plot_path}")

if __name__ == "__main__":
    main()
