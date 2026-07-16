import os
import json
import sqlite3
import numpy as np
from collections import OrderedDict

DB_PATH = "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/qwen3_30b_real_v2.db"

def load_traces():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT prompt_id, token_pos, layer, expert_id, active_indices, energy_k_50 
        FROM activations 
        ORDER BY prompt_id, token_pos, layer
    """)
    rows = cursor.fetchall()
    conn.close()
    
    calibration_db = {}
    evaluation_db = {}
    
    prompt_ids = sorted(list(set(row[0] for row in rows)))
    split_idx = len(prompt_ids) // 2
    calib_prompts = set(prompt_ids[:split_idx])
    eval_prompts = set(prompt_ids[split_idx:])
    
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

def train_predictor_and_profile(calibration_db):
    transition_matrix = np.zeros((48, 128, 128))
    layer_expert_counts = np.zeros((48, 128))
    expert_col_counts = {}
    
    for p_id in calibration_db:
        for t in calibration_db[p_id]:
            for l in range(48):
                if l in calibration_db[p_id][t]:
                    exp_id, active_set = calibration_db[p_id][t][l]
                    layer_expert_counts[l, exp_id] += 1
                    
                    key = (l, exp_id)
                    if key not in expert_col_counts:
                        expert_col_counts[key] = {}
                    for col in active_set:
                        expert_col_counts[key][col] = expert_col_counts[key].get(col, 0) + 1
                    
                    if l > 0 and (l-1) in calibration_db[p_id][t]:
                        prev_exp, _ = calibration_db[p_id][t][l-1]
                        transition_matrix[l, prev_exp, exp_id] += 1
                        
    # Normalize transition matrix
    for l in range(48):
        for e in range(128):
            row_sum = transition_matrix[l, e].sum()
            if row_sum > 0:
                transition_matrix[l, e] /= row_sum
            else:
                transition_matrix[l, e] = 1.0 / 128.0
                
    top_cols_per_expert = {}
    for l in range(48):
        for e in range(128):
            key = (l, e)
            if key in expert_col_counts:
                sorted_cols = sorted(expert_col_counts[key].keys(), key=lambda x: expert_col_counts[key][x], reverse=True)
                if len(sorted_cols) < 768:
                    inactive = list(set(range(768)) - set(sorted_cols))
                    sorted_cols.extend(inactive)
                top_cols_per_expert[key] = sorted_cols
            else:
                top_cols_per_expert[key] = list(range(768))
                
    layer_0_most_frequent = np.argmax(layer_expert_counts[0])
            
    return transition_matrix, top_cols_per_expert, layer_0_most_frequent

def run_interleaved_simulation(evaluation_db, transition_matrix, top_cols_per_expert, layer_0_most_frequent, policy="aaec", cache_size=16, concurrency=16):
    COLUMN_SIZE_BYTES = 5120 * 2
    eval_prompt_ids = sorted(evaluation_db.keys())[:concurrency]
    
    # Interleave prompt tokens: construct a sequence of (prompt_id, token_pos)
    interleaved_tokens = []
    max_tokens = max(len(evaluation_db[p_id]) for p_id in eval_prompt_ids)
    
    for t_idx in range(max_tokens):
        for p_id in eval_prompt_ids:
            t_positions = sorted(evaluation_db[p_id].keys())
            if t_idx < len(t_positions):
                interleaved_tokens.append((p_id, t_positions[t_idx]))
                
    # Precompute static column tuples for each expert
    static_cols_cache = {}
    for l in range(48):
        for e in range(128):
            static_cols_cache[(l, e)] = [(e, col) for col in top_cols_per_expert[(l, e)]]
            
    layer_cache_capacity = cache_size * 128
    gpu_caches = {l: OrderedDict() for l in range(48)}
    
    total_misses = 0
    total_hits = 0
    total_steps = 0
    
    # Prefetch state per prompt request
    current_prefetch_queues = {p_id: {} for p_id in eval_prompt_ids}
    prev_token_active_cols = {p_id: {} for p_id in eval_prompt_ids}

    for p_id, t in interleaved_tokens:
        total_steps += 1
        
        # Execute token t of prompt p_id
        for l in range(48):
            if l not in evaluation_db[p_id][t]:
                continue
            exp_id, active_cols = evaluation_db[p_id][t][l]
            
            cache = gpu_caches[l]
            active_keys = {(exp_id, col) for col in active_cols}
            
            missed = active_keys - cache.keys()
            pref_hits = set()
            
            if policy == "aaec" and l in current_prefetch_queues[p_id]:
                pref_hits = missed.intersection(current_prefetch_queues[p_id][l])
                missed = missed - pref_hits
                
            hits = len(active_keys) - len(missed)
            total_hits += hits
            total_misses += len(missed)
            
            # Update global cache state
            for key in active_keys:
                if key in cache:
                    cache.move_to_end(key)
                else:
                    if len(cache) >= layer_cache_capacity:
                        cache.popitem(last=False)
                    cache[key] = True
                    
        # Prepare prefetch queue for the NEXT token of THIS prompt p_id
        current_prefetch_queues[p_id].clear()
        
        # Find next token index for this prompt
        t_positions = sorted(evaluation_db[p_id].keys())
        current_idx = t_positions.index(t)
        
        if current_idx < len(t_positions) - 1:
            # Predict next expert causally using the current token's expert
            for l in range(48):
                if l == 0:
                    pred_exp = layer_0_most_frequent
                else:
                    if (l-1) in evaluation_db[p_id][t]:
                        prev_exp, _ = evaluation_db[p_id][t][l-1]
                        probs = transition_matrix[l, prev_exp]
                        pred_exp = np.argmax(probs)
                    else:
                        pred_exp = 0
                        
                cache = gpu_caches[l]
                
                # Speculative Gating Check
                if l > 0 and (l-1) in evaluation_db[p_id][t]:
                    prev_exp, _ = evaluation_db[p_id][t][l-1]
                    confidence = transition_matrix[l, prev_exp, pred_exp]
                else:
                    confidence = 1.0
                if confidence < 0.05:
                    continue
                    
                # Temporal speculative prior
                temp_cols = prev_token_active_cols[p_id].get((l, pred_exp), set())
                pred_cols_set = {(pred_exp, col) for col in temp_cols}
                
                static_cols = set(static_cols_cache[(l, pred_exp)][:cache_size])
                predicted_keys = pred_cols_set.union(static_cols)
                
                missing = predicted_keys - cache.keys()
                if missing:
                    current_prefetch_queues[p_id][l] = missing
                    
        # Update temporal prior for next token of this request
        prev_token_active_cols[p_id].clear()
        for l in range(48):
            if l in evaluation_db[p_id][t]:
                exp_id, active_cols = evaluation_db[p_id][t][l]
                prev_token_active_cols[p_id][(l, exp_id)] = active_cols

    hit_rate = total_hits / max(1, total_hits + total_misses)
    return hit_rate

def main():
    print("Loading traces...")
    calibration_db, evaluation_db = load_traces()
    transition_matrix, top_cols_per_expert, layer_0_most_frequent = train_predictor_and_profile(calibration_db)
    
    print("\n--- CONTINUOUS BATCHING INTERLEAVED STUDY ---")
    for cs in [8, 16, 32]:
        for conc in [1, 4, 8, 16]:
            hr_lru = run_interleaved_simulation(evaluation_db, transition_matrix, top_cols_per_expert, layer_0_most_frequent, policy="lru", cache_size=cs, concurrency=conc)
            hr_aaec = run_interleaved_simulation(evaluation_db, transition_matrix, top_cols_per_expert, layer_0_most_frequent, policy="aaec", cache_size=cs, concurrency=conc)
            print(f"Cache Size = {cs:2d} | Concurrency = {conc:2d} | LRU Hit = {hr_lru*100:5.2f}% | AAEC Hit = {hr_aaec*100:5.2f}% | Delta = {(hr_aaec - hr_lru)*100:+5.2f}%")

if __name__ == "__main__":
    main()
