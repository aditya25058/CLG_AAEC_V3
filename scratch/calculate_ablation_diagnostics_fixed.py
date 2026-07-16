import os
import json
import sqlite3
import numpy as np
from collections import OrderedDict

spec = {
    "db_path": "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/qwen3_30b_real_v2.db",
    "num_layers": 48,
    "num_experts": 128,
    "intermediate_dim": 768,
    "active_experts": 8
}

def main():
    conn = sqlite3.connect(spec["db_path"])
    cursor = conn.cursor()
    cursor.execute("""
        SELECT prompt_id, token_pos, layer, expert_id, active_indices, energy_k_50
        FROM activations
        ORDER BY prompt_id, token_pos, layer
    """)
    rows = cursor.fetchall()
    conn.close()
    
    prompt_ids = sorted(list(set(row[0] for row in rows)))
    split_idx = len(prompt_ids) // 2
    eval_prompts = set(prompt_ids[split_idx:])
    
    eval_rows = [row for row in rows if row[0] in eval_prompts]
    
    # Group rows by (prompt_id, token_pos, layer) to load all active experts
    grouped_eval = {}
    for row in eval_rows:
        p_id, t_pos, layer, exp_id, indices_str, k50 = row
        active_cols = json.loads(indices_str)[:k50]
        
        key = (p_id, t_pos, layer)
        if key not in grouped_eval:
            grouped_eval[key] = []
        grouped_eval[key].append((exp_id, active_cols))
        
    # Sort keys to ensure chronological order
    sorted_keys = sorted(grouped_eval.keys())
    
    # ---------------------------------------------------------
    # 1. Monolithic Caching Analysis (no_aaec)
    # ---------------------------------------------------------
    cache_size = 32
    layer_capacity = cache_size
    NL = spec["num_layers"]
    
    caches_mono = {l: OrderedDict() for l in range(NL)}
    mono_misses = 0
    mono_hits = 0
    mono_miss_sizes = []
    
    for (p_id, t_pos, layer) in sorted_keys:
        cache = caches_mono[layer]
        active_experts = grouped_eval[(p_id, t_pos, layer)]
        
        step_misses = 0
        for exp_id, _ in active_experts:
            if exp_id in cache:
                mono_hits += 1
                cache.move_to_end(exp_id)
            else:
                mono_misses += 1
                step_misses += 1
                if len(cache) >= layer_capacity:
                    cache.popitem(last=False)
                cache[exp_id] = True
        if step_misses > 0:
            mono_miss_sizes.append(step_misses * 768)
            
    # ---------------------------------------------------------
    # 2. Slicing + LRU Analysis (slicing_lru)
    # ---------------------------------------------------------
    layer_capacity_col = cache_size * spec["num_experts"] # 32 * 128 = 4096 columns
    caches_col = {l: OrderedDict() for l in range(NL)}
    
    col_misses = 0
    col_hits = 0
    col_miss_sizes = []
    
    for (p_id, t_pos, layer) in sorted_keys:
        cache = caches_col[layer]
        active_experts = grouped_eval[(p_id, t_pos, layer)]
        
        step_misses = 0
        for exp_id, active_cols in active_experts:
            for col in active_cols:
                key = (exp_id, col)
                if key in cache:
                    col_hits += 1
                    cache.move_to_end(key)
                else:
                    col_misses += 1
                    step_misses += 1
                    if len(cache) >= layer_capacity_col:
                        cache.popitem(last=False)
                    cache[key] = True
        if step_misses > 0:
            col_miss_sizes.append(step_misses)
            
    # Print results
    COLUMN_SIZE_BYTES = 5120 * 2 # BF16
    link_bw = 8.0 # GB/s
    
    avg_mono_miss_cols = np.mean(mono_miss_sizes)
    avg_mono_miss_bytes = avg_mono_miss_cols * COLUMN_SIZE_BYTES
    avg_mono_time_us = (avg_mono_miss_bytes / (link_bw * 1e9)) * 1e6
    
    avg_col_miss_cols = np.mean(col_miss_sizes)
    avg_col_miss_bytes = avg_col_miss_cols * COLUMN_SIZE_BYTES
    avg_col_time_us = (avg_col_miss_bytes / (link_bw * 1e9)) * 1e6
    
    total_steps = len(sorted_keys)
    mono_miss_rate = len(mono_miss_sizes) / total_steps * 100
    col_miss_rate = len(col_miss_sizes) / total_steps * 100
    
    print("=" * 80)
    print(" EMPIRICAL ABLATION DIAGNOSTICS FOR QWEN3-30B (FIXED GROUPING)")
    print("=" * 80)
    print(f"Total layer execution steps evaluated: {total_steps}")
    print("\n--- CONFIGURATION 1: MONOLITHIC CACHING (No AAEC) ---")
    print(f"Total layer steps with cache misses: {len(mono_miss_sizes)} ({mono_miss_rate:.2f}%)")
    print(f"Average columns transferred per miss: {avg_mono_miss_cols:.1f} cols")
    print(f"Average payload transferred per miss: {avg_mono_miss_bytes / (1024*1024):.2f} MB")
    print(f"Average raw transfer time per miss:   {avg_mono_time_us:.2f} µs")
    
    print("\n--- CONFIGURATION 2: SLICING + LRU CACHING ---")
    print(f"Total layer steps with cache misses: {len(col_miss_sizes)} ({col_miss_rate:.2f}%)")
    print(f"Average columns transferred per miss: {avg_col_miss_cols:.1f} cols")
    print(f"Average payload transferred per miss: {avg_col_miss_bytes / (1024*1024):.2f} MB")
    print(f"Average raw transfer time per miss:   {avg_col_time_us:.2f} µs")
    
    print("\n--- KEY ANALYSIS ---")
    print(f"Miss size reduction: {avg_mono_miss_bytes / avg_col_miss_bytes:.1f}x smaller payload per miss!")
    print(f"Hiding window coverage: At T_attn = 50 us:")
    print(f"  - Monolithic miss exposes:  {max(0.0, avg_mono_time_us - 50.0):.2f} µs stall")
    print(f"  - Column-level miss exposes: {max(0.0, avg_col_time_us - 50.0):.2f} µs stall")
    print("=" * 80)

if __name__ == '__main__':
    main()
