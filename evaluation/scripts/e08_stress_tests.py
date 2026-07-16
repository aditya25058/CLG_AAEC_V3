# evaluation/scripts/e08_stress_tests.py
# FIXED: load_db_traces appends all experts, DeepSeek layers=26
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
        "hidden_size": 2048,
        "active_experts": 8
    },
    "deepseek_v2_lite": {
        "db_path": "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/deepseek_lite_real.db",
        "num_layers": 27,
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
        if layer not in target_db[p_id][t_pos]:
            target_db[p_id][t_pos][layer] = []
        target_db[p_id][t_pos][layer].append((exp_id, active_set))
        
    return calibration_db, evaluation_db

def train_predictors(calibration_db, spec: dict):
    NL = spec["num_layers"]
    NE = spec["num_experts"]
    
    transition_matrix = np.zeros((NL + 1, NE, NE))
    layer_expert_counts = np.zeros((NL + 1, NE))
    
    for p_id in calibration_db:
        for t in calibration_db[p_id]:
            for l in calibration_db[p_id][t]:
                for exp_id, active_set in calibration_db[p_id][t][l]:
                    if exp_id < NE:
                        layer_expert_counts[l, exp_id] += 1
                        
                        if l > 0 and (l-1) in calibration_db[p_id][t]:
                            for prev_exp, _ in calibration_db[p_id][t][l-1]:
                                if prev_exp < NE:
                                    transition_matrix[l, prev_exp, exp_id] += 1
                                
    # Normalize transition matrix
    for l in range(NL + 1):
        for e in range(NE):
            row_sum = transition_matrix[l, e].sum()
            if row_sum > 0:
                transition_matrix[l, e] /= row_sum
            else:
                transition_matrix[l, e] = 1.0 / NE
                
    layer_0_most_frequent = int(np.argmax(layer_expert_counts[0]))
    return transition_matrix, layer_0_most_frequent

def run_cold_start_stress(evaluation_db, spec: dict):
    # Empty cache at step 0, monitor per-token hit rates for the first 64 tokens
    NL = spec["num_layers"]
    NE = spec["num_experts"]
    cache_size = 32
    layer_capacity = cache_size * NE
    
    gpu_caches = {l: OrderedDict() for l in range(NL + 1)}
    
    token_hits = np.zeros(64)
    token_totals = np.zeros(64)
    
    eval_prompt_ids = sorted(evaluation_db.keys())
    
    for p_id in eval_prompt_ids[:10]:
        t_positions = sorted(evaluation_db[p_id].keys())[:64]
        
        # Reset cache at start of prompt
        for l in range(NL + 1):
            gpu_caches[l].clear()
            
        for t_idx, t in enumerate(t_positions):
            if t_idx >= 64:
                break
                
            for l in evaluation_db[p_id][t]:
                cache = gpu_caches[l]
                
                # Collect all active keys across ALL experts at this step
                all_active_keys = set()
                for exp_id, active_cols in evaluation_db[p_id][t][l]:
                    for col in active_cols:
                        all_active_keys.add((exp_id, col))
                
                local_active = all_active_keys.intersection(cache.keys())
                hits = len(local_active)
                misses = len(all_active_keys) - hits
                
                token_hits[t_idx] += hits
                token_totals[t_idx] += (hits + misses)
                
                # Update cache
                for key in all_active_keys:
                    if key in cache:
                        cache.move_to_end(key)
                    else:
                        if len(cache) >= layer_capacity:
                            cache.popitem(last=False)
                        cache[key] = True
                        
    hit_rates_by_token = [float(token_hits[i] / max(1, token_totals[i])) for i in range(64)]
    return hit_rates_by_token

def main():
    for model_name, spec in MODELS.items():
        print(f"Running Cold Start Stress Test for {model_name}...")
        calib_db, eval_db = load_db_traces(spec["db_path"])
        trans_matrix, l0_freq = train_predictors(calib_db, spec)
        
        cold_start_hit_rates = run_cold_start_stress(eval_db, spec)
        print(f"  Stabilization hit rate after 10 tokens: {cold_start_hit_rates[10]*100:.2f}%")
        print(f"  Stabilization hit rate after 30 tokens: {cold_start_hit_rates[30]*100:.2f}%")
        
        out_dir = f"/home/palakm/MoEServingSim/evaluation/results/e08_stress/{model_name}"
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "cold_start_results.json"), "w") as f:
            json.dump({"cold_start_hit_rates": cold_start_hit_rates}, f, indent=4)

if __name__ == "__main__":
    main()
