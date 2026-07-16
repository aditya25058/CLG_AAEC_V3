import os
import json
import sqlite3
import numpy as np
from collections import OrderedDict

MODELS = {
    "qwen3_30b": {
        "db_path": "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/qwen3_30b_real_v2.db",
        "num_layers": 48,
        "num_experts": 128,
        "intermediate_dim": 768,
        "active_experts": 8
    },
    "deepseek_v2_lite": {
        "db_path": "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/deepseek_lite_real.db",
        "num_layers": 27,
        "num_experts": 64,
        "intermediate_dim": 1408,
        "active_experts": 6
    }
}

def load_db_traces(db_path: str):
    conn = sqlite3.connect(db_path)
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
    calib_prompts = set(prompt_ids[:split_idx])
    eval_prompts = set(prompt_ids[split_idx:])
    
    calibration_db = {}
    evaluation_db = {}
    
    for row in rows:
        p_id, t_pos, layer, exp_id, indices_str, k50 = row
        indices = json.loads(indices_str)[:k50]
        active_set = set(indices)
        
        target_db = calibration_db if p_id in calib_prompts else evaluation_db
        
        if p_id not in target_db:
            target_db[p_id] = {}
        if t_pos not in target_db[p_id]:
            target_db[p_id][t_pos] = {}
            
        target_db[p_id][t_pos][layer] = (exp_id, active_set)
        
    return calibration_db, evaluation_db

def train_predictors(calibration_db, spec: dict):
    NL = spec["num_layers"]
    NE = spec["num_experts"]
    
    transition_matrix = np.zeros((NL, NE, NE))
    layer_expert_counts = np.zeros((NL, NE))
    expert_col_counts = {}
    
    for p_id in calibration_db:
        for t in calibration_db[p_id]:
            for l in range(NL):
                if l in calibration_db[p_id][t]:
                    exp_id, active_set = calibration_db[p_id][t][l]
                    if exp_id < NE:
                        layer_expert_counts[l, exp_id] += 1
                        
                        key = (l, exp_id)
                        if key not in expert_col_counts:
                            expert_col_counts[key] = {}
                        for col in active_set:
                            expert_col_counts[key][col] = expert_col_counts[key].get(col, 0) + 1
                        
                        if l > 0 and (l-1) in calibration_db[p_id][t]:
                            prev_exp, _ = calibration_db[p_id][t][l-1]
                            if prev_exp < NE:
                                transition_matrix[l, prev_exp, exp_id] += 1
                                
    # Normalize transition matrix
    for l in range(NL):
        for e in range(NE):
            row_sum = transition_matrix[l, e].sum()
            if row_sum > 0:
                transition_matrix[l, e] /= row_sum
            else:
                transition_matrix[l, e] = 1.0 / NE
                
    # Precompute top columns per expert based on calibration profiles
    top_cols_per_expert = {}
    for l in range(NL):
        for e in range(NE):
            key = (l, e)
            if key in expert_col_counts:
                sorted_cols = sorted(expert_col_counts[key].keys(), key=lambda x: expert_col_counts[key][x], reverse=True)
                if len(sorted_cols) < 768:
                    inactive = list(set(range(768)) - set(sorted_cols))
                    sorted_cols.extend(inactive)
                top_cols_per_expert[key] = sorted_cols
            else:
                top_cols_per_expert[key] = list(range(768))
                
    layer_0_most_frequent = int(np.argmax(layer_expert_counts[0]))
    return transition_matrix, top_cols_per_expert, layer_0_most_frequent

def run_ablation_simulation(
    evaluation_db,
    transition_matrix,
    top_cols_per_expert,
    layer_0_most_frequent,
    config: str, # "no_aaec", "slicing_only", "slicing_lru", "slicing_lru_prefetch"
    cache_size: int,
    spec: dict,
    link_bw_gb_s: float = 8.0
):
    NL = spec["num_layers"]
    NE = spec["num_experts"]
    
    COMPUTE_TIME_PER_LAYER_US = 50.0
    LATENCY_OVERHEAD_PER_DMA_US = 0.5
    COLUMN_SIZE_BYTES = 5120 * 2
    
    # Layer-wise caches
    gpu_caches = {l: OrderedDict() for l in range(NL)}
    
    # Cache capacity configurations
    if config == "no_aaec":
        layer_capacity = cache_size  # Cache whole experts (size = 32 experts)
    elif config == "slicing_only":
        layer_capacity = 0           # Cache size 0 (no cache)
    else:
        layer_capacity = cache_size * NE # Cache size = 32 * NE columns (column-granular)
        
    total_hits = 0
    total_misses = 0
    total_pushed_bytes = 0
    total_prefetched_bytes = 0
    total_stalls_us = 0.0
    total_steps = 0
    
    current_prefetch_queue = {}
    prev_token_active_cols = {}
    
    eval_prompt_ids = sorted(evaluation_db.keys())
    
    for p_id in eval_prompt_ids:
        t_positions = sorted(evaluation_db[p_id].keys())
        current_prefetch_queue.clear()
        prev_token_active_cols.clear()
        
        for idx, t in enumerate(t_positions):
            total_steps += 1
            
            for l in range(NL):
                if l not in evaluation_db[p_id][t]:
                    continue
                exp_id, active_cols = evaluation_db[p_id][t][l]
                
                cache = gpu_caches[l]
                
                if config == "no_aaec":
                    # Whole expert hits
                    active_keys = {exp_id}
                else:
                    # Column hits
                    active_keys = {(exp_id, col) for col in active_cols}
                
                if config == "slicing_only":
                    # Empty cache on every step
                    local_active = set()
                    missed = active_keys
                else:
                    local_active = active_keys.intersection(cache.keys())
                    missed = active_keys - local_active
                
                if config == "no_aaec":
                    hits = len(local_active) * 768
                    miss_cols_count = len(missed) * 768
                else:
                    hits = len(local_active)
                    miss_cols_count = len(missed)
                    
                total_hits += hits
                total_misses += miss_cols_count
                
                # Update cache
                if config != "slicing_only":
                    for key in active_keys:
                        if key in cache:
                            cache.move_to_end(key)
                        else:
                            if len(cache) >= layer_capacity:
                                cache.popitem(last=False)
                            cache[key] = True
                            
                # Calculate stalls
                if missed:
                    copy_size = miss_cols_count * COLUMN_SIZE_BYTES
                    copy_time = (copy_size / (link_bw_gb_s * 1e9)) * 1e6
                    total_pushed_bytes += copy_size
                    stall = max(0.0, (copy_time + LATENCY_OVERHEAD_PER_DMA_US) - COMPUTE_TIME_PER_LAYER_US)
                    total_stalls_us += stall
            
            # Prefetch logic
            current_prefetch_queue.clear()
            if idx < len(t_positions) - 1:
                # Prefetching is only enabled in slicing_lru_prefetch configuration
                if config != "slicing_lru_prefetch":
                    continue
                    
                for l in range(NL):
                    if l == 0:
                        pred_exp = layer_0_most_frequent
                    else:
                        if (l-1) in evaluation_db[p_id][t]:
                            prev_exp, _ = evaluation_db[p_id][t][l-1]
                            if prev_exp < NE:
                                pred_exp = int(np.argmax(transition_matrix[l, prev_exp]))
                            else:
                                pred_exp = 0
                        else:
                            pred_exp = 0
                            
                    cache = gpu_caches[l]
                    
                    # Column-level prefetch
                    temp_cols = prev_token_active_cols.get((l, pred_exp), set())
                    pred_cols_set = {(pred_exp, col) for col in temp_cols}
                    
                    # Fallback static columns
                    static_cols = set([(pred_exp, col) for col in top_cols_per_expert[(l, pred_exp)][:cache_size]])
                    predicted_keys = pred_cols_set.union(static_cols)
                    
                    missing = predicted_keys - cache.keys()
                    if missing:
                        current_prefetch_queue[l] = missing
                        total_prefetched_bytes += len(missing) * COLUMN_SIZE_BYTES
                        
            # Update temporal history
            prev_token_active_cols.clear()
            for l in range(NL):
                if l in evaluation_db[p_id][t]:
                    exp_id, active_cols = evaluation_db[p_id][t][l]
                    prev_token_active_cols[(l, exp_id)] = active_cols
                    
    hit_rate = total_hits / max(1, total_hits + total_misses)
    total_gb = (total_prefetched_bytes + total_pushed_bytes) / 1e9
    avg_stall = (total_stalls_us / 1000.0) / max(1, total_steps)
    
    return {
        "hit_rate": hit_rate,
        "avg_stall_ms": avg_stall,
        "total_transferred_gb": total_gb
    }

def main():
    configs = ["no_aaec", "slicing_only", "slicing_lru", "slicing_lru_prefetch"]
    
    for model_name, spec in MODELS.items():
        print(f"Running Full Ablation Sweep for {model_name}...")
        calib_db, eval_db = load_db_traces(spec["db_path"])
        trans_matrix, top_cols, l0_freq = train_predictors(calib_db, spec)
        
        results = {}
        for c in configs:
            res = run_ablation_simulation(eval_db, trans_matrix, top_cols, l0_freq, c, cache_size=32, spec=spec, link_bw_gb_s=8.0)
            results[c] = {
                "hit_rate": res["hit_rate"],
                "avg_stall_ms": res["avg_stall_ms"],
                "total_transferred_gb": res["total_transferred_gb"]
            }
            print(f"  Configuration: {c:<22} | Hit Rate: {res['hit_rate']*100:6.2f}% | Stall: {res['avg_stall_ms']:.4f} ms | Data: {res['total_transferred_gb']:.2f} GB")
            
        out_dir = f"/home/palakm/MoEServingSim/evaluation/results/e10_ablation/{model_name}"
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "full_ablation_results.json"), "w") as f:
            json.dump(results, f, indent=4)
        print("Success.")

if __name__ == "__main__":
    main()
