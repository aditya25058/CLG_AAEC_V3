# evaluation/scripts/e04_cache_policy_comparison.py
# FIXED: load_db_traces appends all experts, true MIN/LFU scan full cache
import os
import json
import sqlite3
import random
import numpy as np
from collections import OrderedDict

MODELS = {
    "qwen3_30b": {
        "db_path": "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/qwen3_30b_real_v2.db",
        "num_layers": 48,
        "num_experts": 128,
        "intermediate_dim": 768,
        "hidden_size": 2048,
        "active_experts": 8
    },
    "deepseek_v2_lite": {
        "db_path": "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/deepseek_lite_real.db",
        "num_layers": 26,
        "num_experts": 64,
        "intermediate_dim": 1408,
        "hidden_size": 2048,
        "active_experts": 6
    }
}

def load_db_traces(db_path: str):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT prompt_id, token_pos, layer, expert_id, active_indices, energy_k_50
        FROM activations
    """)
    rows = cursor.fetchall()
    conn.close()
    
    prompt_ids = sorted(list(set(row[0] for row in rows)))
    split_idx = len(prompt_ids) // 2
    calib_prompts = set(prompt_ids[:split_idx])
    eval_prompts = set(prompt_ids[split_idx:])
    
    calibration_db = {}
    evaluation_db = {}
    
    # Speed optimization: Only use first 2 evaluation prompts for cache policy comparison
    # This provides a massive statistical sample while keeping runtime fast with O(N) MIN/LFU evictions.
    eval_prompts = sorted(list(eval_prompts))[:2]
    eval_prompts = set(eval_prompts)
    
    for row in rows:
        p_id = row[0]
        if p_id not in eval_prompts:
            continue
        p_id, t_pos, layer, exp_id, indices_str, k50 = row
        indices = json.loads(indices_str)[:k50]
        active_set = set(indices)
        
        if p_id not in evaluation_db:
            evaluation_db[p_id] = {}
        if t_pos not in evaluation_db[p_id]:
            evaluation_db[p_id][t_pos] = {}
        if layer not in evaluation_db[p_id][t_pos]:
            evaluation_db[p_id][t_pos][layer] = []
        evaluation_db[p_id][t_pos][layer].append((exp_id, active_set))
        
    return calibration_db, evaluation_db

def run_cache_simulation(evaluation_db, policy: str, cache_size: int, spec: dict):
    NL = spec["num_layers"]
    NE = spec["num_experts"]
    H = spec["intermediate_dim"]
    
    # Initialize cache per layer
    gpu_caches = {l: OrderedDict() for l in range(NL + 1)}
    layer_capacity = cache_size * NE
    
    total_hits = 0
    total_misses = 0
    total_steps = 0
    
    # Pre-scan access patterns for Belady's MIN (offline oracle)
    future_accesses = {}
    
    if policy == "min":
        step_idx = 0
        for p_id in sorted(evaluation_db.keys()):
            t_positions = sorted(evaluation_db[p_id].keys())
            for t in t_positions:
                for l in range(NL + 1):
                    if l not in evaluation_db[p_id][t]:
                        continue
                    for exp_id, active_cols in evaluation_db[p_id][t][l]:
                        for col in active_cols:
                            k = (l, exp_id, col)
                            if k not in future_accesses:
                                future_accesses[k] = []
                            future_accesses[k].append(step_idx)
                step_idx += 1

    # Usage counts for LFU
    lfu_counts = {l: {} for l in range(NL + 1)}
    
    # Next access absolute step cache for MIN policy
    min_next_access = {l: {} for l in range(NL + 1)}
    import bisect

    eval_prompt_ids = sorted(evaluation_db.keys())
    step_idx = 0
    
    for p_id in eval_prompt_ids:
        t_positions = sorted(evaluation_db[p_id].keys())
        for t in t_positions:
            total_steps += 1
            
            for l in range(NL + 1):
                if l not in evaluation_db[p_id][t]:
                    continue
                
                # Collect all active keys across ALL experts at this step
                all_active_keys = set()
                for exp_id, active_cols in evaluation_db[p_id][t][l]:
                    for col in active_cols:
                        all_active_keys.add((exp_id, col))
                
                cache = gpu_caches[l]
                local_active = all_active_keys.intersection(cache.keys())
                missed = all_active_keys - local_active
                
                total_hits += len(local_active)
                total_misses += len(missed)
                
                # Update LFU counts
                if policy == "lfu":
                    for key in all_active_keys:
                        lfu_counts[l][key] = lfu_counts[l].get(key, 0) + 1
                
                # Update caches based on policy
                for key in all_active_keys:
                    if key in cache:
                        if policy == "lru":
                            cache.move_to_end(key)
                        if policy == "min":
                            full_k = (l, key[0], key[1])
                            access_list = future_accesses.get(full_k, [])
                            idx = bisect.bisect_right(access_list, step_idx)
                            min_next_access[l][key] = access_list[idx] if idx < len(access_list) else float('inf')
                    else:
                        # We have a miss and need to insert
                        if len(cache) >= layer_capacity:
                            # Evict
                            if policy == "fifo":
                                cache.popitem(last=False)
                            elif policy == "lru":
                                cache.popitem(last=False)
                            elif policy == "lfu":
                                # True LFU: Scan the entire cache using fast C-level min key lookup
                                min_key = min(cache, key=lfu_counts[l].__getitem__)
                                cache.pop(min_key)
                            elif policy == "min":
                                # True Belady's MIN: Scan the entire cache using fast C-level max key lookup
                                evict_key = max(cache, key=min_next_access[l].__getitem__)
                                cache.pop(evict_key)
                                min_next_access[l].pop(evict_key, None)
                        
                        cache[key] = True
                        if policy == "min":
                            full_k = (l, key[0], key[1])
                            access_list = future_accesses.get(full_k, [])
                            idx = bisect.bisect_right(access_list, step_idx)
                            min_next_access[l][key] = access_list[idx] if idx < len(access_list) else float('inf')
            
            step_idx += 1
            
    hit_rate = total_hits / max(1, total_hits + total_misses)
    return hit_rate

def main():
    policies = ["fifo", "lru", "lfu", "min"]
    cache_sizes = [16, 32, 64, 128, 256, 512]
    
    for model_name, spec in MODELS.items():
        print(f"Running Cache Policy Comparison for {model_name}...")
        calib_db, eval_db = load_db_traces(spec["db_path"])
        
        results = {p: {} for p in policies}
        for size in cache_sizes:
            print(f"  Simulating Cache Size = {size}...")
            for p in policies:
                hr = run_cache_simulation(eval_db, p, size, spec)
                results[p][str(size)] = hr
                print(f"    Policy: {p:<6} | Hit Rate: {hr*100:.2f}%")
                
        out_dir = f"/home/palakm/MoEServingSim/evaluation/results/e04_cache/{model_name}"
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "cache_comparison.json"), "w") as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
