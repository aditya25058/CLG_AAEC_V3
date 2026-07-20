# evaluation/scripts/e14_distributed_prefetcher.py
# Simulates reactive, predictive (Markov), predictive (Temporal), and Oracle prefetchers.
import os
import json
import sqlite3
import numpy as np
from collections import OrderedDict, defaultdict

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

def load_db_traces_with_split(db_path: str):
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
    train_prompts = set(prompt_ids[:split_idx])
    eval_prompts = set(prompt_ids[split_idx:])
    eval_prompts = sorted(list(eval_prompts))[:5]
    eval_prompts = set(eval_prompts)
    
    train_db = []
    eval_db = {}
    
    for row in rows:
        p_id, t_pos, layer, exp_id, indices_str, k50 = row
        indices = json.loads(indices_str)[:k50]
        active_set = set(indices)
        
        if p_id in train_prompts:
            train_db.append((p_id, t_pos, layer, exp_id, active_set))
        else:
            if p_id not in eval_prompts:
                continue
            if p_id not in eval_db:
                eval_db[p_id] = {}
            if t_pos not in eval_db[p_id]:
                eval_db[p_id][t_pos] = {}
            if layer not in eval_db[p_id][t_pos]:
                eval_db[p_id][t_pos][layer] = []
            eval_db[p_id][t_pos][layer].append((exp_id, active_set))
            
    return train_db, eval_db

def train_transition_predictor(train_db, num_layers: int, num_experts: int):
    transition_counts = {l: defaultdict(lambda: defaultdict(int)) for l in range(1, num_layers + 1)}
    popularity = {l: defaultdict(int) for l in range(num_layers + 1)}
    
    steps = defaultdict(lambda: defaultdict(list))
    for p_id, t_pos, layer, exp_id, _ in train_db:
        steps[(p_id, t_pos)][layer].append(exp_id)
        popularity[layer][exp_id] += 1
        
    for step_key, layers_data in steps.items():
        for l in range(1, num_layers + 1):
            if (l - 1) in layers_data and l in layers_data:
                for prev_exp in layers_data[l - 1]:
                    for curr_exp in layers_data[l]:
                        transition_counts[l][prev_exp][curr_exp] += 1
                
    predictor = {}
    for l in range(1, num_layers + 1):
        predictor[l] = {}
        for prev_exp in range(num_experts):
            if prev_exp in transition_counts[l] and transition_counts[l][prev_exp]:
                predicted = max(transition_counts[l][prev_exp].items(), key=lambda x: x[1])[0]
                predictor[l][prev_exp] = predicted
            else:
                if popularity[l]:
                    predictor[l][prev_exp] = max(popularity[l].items(), key=lambda x: x[1])[0]
                else:
                    predictor[l][prev_exp] = 0
                    
    predictor[0] = {}
    if popularity[0]:
        most_popular_l0 = max(popularity[0].items(), key=lambda x: x[1])[0]
    else:
        most_popular_l0 = 0
    for prev_exp in range(num_experts):
        predictor[0][prev_exp] = most_popular_l0
        
    return predictor

def run_distributed_prefetcher_simulation(
    eval_db,
    predictor,
    prefetch_policy: str, # "reactive", "markov", "temporal", "oracle"
    world_size: int,
    spec: dict
):
    NL = spec["num_layers"]
    NE = spec["num_experts"]
    H = spec["hidden_size"]
    I = spec["intermediate_dim"]
    
    experts_per_node = NE // world_size if world_size > 0 else NE
    
    LOCAL_BW_GBPS = 64.0
    NETWORK_BW_GBPS = 10.0
    LOCAL_LATENCY_US = 0.5
    NETWORK_LATENCY_US = 5.0
    COMPUTE_TIME_PER_LAYER_US = 50.0
    ATTENTION_COMPUTE_TIME_US = 100.0
    
    COLUMN_SIZE_BYTES = 3 * H * 2
    
    cache_capacity_cols = 32
    gpu_caches = {l: OrderedDict() for l in range(NL + 1)}
    column_capacity = cache_capacity_cols * NE
    
    total_steps = 0
    network_bytes_transferred = 0
    total_stalls_us = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    eval_prompt_ids = sorted(eval_db.keys())
    
    # Track temporal active history (layer -> set of active keys in previous token)
    prev_token_active = defaultdict(set)
    
    for p_id in eval_prompt_ids:
        t_positions = sorted(eval_db[p_id].keys())
        prev_token_active.clear()
        
        for l in range(NL + 1):
            gpu_caches[l].clear()
            
        for t in t_positions:
            total_steps += 1
            
            # Simulated cache prefetch window insertion (before executing current step)
            current_prefetch_keys = defaultdict(set)
            
            if prefetch_policy != "reactive":
                # Predict what keys to prefetch for this step across layers
                for l in range(NL + 1):
                    if l in eval_db[p_id][t]:
                        experts_at_step = eval_db[p_id][t][l]
                        
                        # Identify predicted expert
                        pred_exp_id = 0
                        if prefetch_policy == "oracle":
                            # Oracle: perfect prediction of all experts at current step
                            for exp_id, active_cols in experts_at_step:
                                current_prefetch_keys[l].update({(exp_id, col) for col in active_cols})
                            continue
                        elif prefetch_policy == "temporal":
                            # Temporal: use active experts of the same layer from previous token
                            prev_exps = prev_token_active[l]
                            if prev_exps:
                                for exp_id in prev_exps:
                                    current_prefetch_keys[l].update({(exp_id, col) for col in range(cache_capacity_cols)})
                            continue
                        elif prefetch_policy == "markov":
                            # Markov: prediction based on previous layer's active expert
                            if l == 0:
                                pred_exp_id = predictor[0][0]
                            elif (l-1) in eval_db[p_id][t]:
                                prev_exp = eval_db[p_id][t][l-1][0][0]
                                pred_exp_id = predictor[l].get(prev_exp, 0)
                            else:
                                pred_exp_id = predictor[l].get(0, 0)
                                
                            current_prefetch_keys[l].update({(pred_exp_id, col) for col in range(cache_capacity_cols)})
            
            # Apply prefetching transfers to cache (simulates pre-fetching ahead of compute)
            for l, pref_keys in current_prefetch_keys.items():
                cache = gpu_caches[l]
                missing = {k for k in pref_keys if k not in cache}
                if missing:
                    # Count bandwidth
                    for k_exp, k_col in missing:
                        node_id = k_exp // experts_per_node if experts_per_node > 0 else 0
                        if node_id != 0:
                            network_bytes_transferred += COLUMN_SIZE_BYTES
                    # Load into cache
                    for key in missing:
                        if len(cache) >= column_capacity:
                            cache.popitem(last=False)
                        cache[key] = True
            
            # Run FFN compute and calculate actual latency/stalls
            for l in range(NL + 1):
                if l not in eval_db[p_id][t]:
                    continue
                experts_at_step = eval_db[p_id][t][l]
                cache = gpu_caches[l]
                
                # Active columns at this step
                active_keys = set()
                for exp_id, active_cols in experts_at_step:
                    for col in active_cols:
                        active_keys.add((exp_id, col))
                        
                local_active = {k for k in active_keys if k in cache}
                missed_keys = active_keys - local_active
                missed_bytes = len(missed_keys) * COLUMN_SIZE_BYTES
                
                # Update cache hit accuracy metrics
                if prefetch_policy != "reactive":
                    total_predictions += len(active_keys)
                    correct_predictions += len(local_active)
                
                # Update LRU
                for key in active_keys:
                    if key in cache:
                        cache.move_to_end(key)
                    else:
                        if len(cache) >= column_capacity:
                            cache.popitem(last=False)
                        cache[key] = True
                
                # Calculate stalls for this layer
                if missed_keys:
                    # Count bandwidth for misses
                    for k_exp, k_col in missed_keys:
                        node_id = k_exp // experts_per_node if experts_per_node > 0 else 0
                        if node_id != 0:
                            network_bytes_transferred += COLUMN_SIZE_BYTES
                            
                    # Load time
                    # Identify if miss needs remote network or local bandwidth
                    has_remote = False
                    for k_exp, k_col in missed_keys:
                        node_id = k_exp // experts_per_node if experts_per_node > 0 else 0
                        if node_id != 0:
                            has_remote = True
                            break
                            
                    if has_remote:
                        t_transfer = (missed_bytes / (NETWORK_BW_GBPS * 1e9)) * 1e6 + NETWORK_LATENCY_US
                    else:
                        t_transfer = (missed_bytes / (LOCAL_BW_GBPS * 1e9)) * 1e6 + LOCAL_LATENCY_US
                        
                    # Since we prefetched, the copy is overlapped with Attention compute + layer compute
                    overlap_window = COMPUTE_TIME_PER_LAYER_US
                    if prefetch_policy != "reactive":
                        overlap_window += ATTENTION_COMPUTE_TIME_US
                        
                    stall = max(0.0, t_transfer - overlap_window)
                    total_stalls_us += stall
                    
            # Record active history for temporal predictor next step
            prev_token_active.clear()
            for l in eval_db[p_id][t]:
                for exp_id, _ in eval_db[p_id][t][l]:
                    prev_token_active[l].add(exp_id)
                            
    total_network_gb = network_bytes_transferred / 1e9
    avg_stall_ms = (total_stalls_us / 1000.0) / max(1, total_steps)
    
    BASE_COMPUTE_TIME_MS = 1.5
    avg_total_latency_ms = BASE_COMPUTE_TIME_MS + (avg_stall_ms * NL)
    throughput_tokens_sec = 1000.0 / avg_total_latency_ms
    
    prefetch_hit_rate = (correct_predictions / max(1, total_predictions)) * 100.0 if prefetch_policy != "reactive" else 0.0
    
    return {
        "network_gb": total_network_gb,
        "avg_stall_ms": avg_stall_ms,
        "throughput": throughput_tokens_sec,
        "prefetch_hit_rate": prefetch_hit_rate
    }

def main():
    world_sizes = [4, 8, 16]
    policies = ["reactive", "markov", "temporal", "oracle"]
    
    for model_name, spec in MODELS.items():
        print(f"\n==========================================")
        print(f"DISTRIBUTED PREFETCHER SIMULATION FOR {model_name.upper()}")
        print(f"==========================================")
        train_db, eval_db = load_db_traces_with_split(spec["db_path"])
        predictor = train_transition_predictor(train_db, spec["num_layers"], spec["num_experts"])
        
        results = {p: {} for p in policies}
        
        for p in policies:
            print(f"Policy: {p.upper()}")
            for ws in world_sizes:
                res = run_distributed_prefetcher_simulation(eval_db, predictor, p, ws, spec)
                results[p][str(ws)] = res
                print(f"  Nodes: {ws:<2} | Net Moved: {res['network_gb']:6.2f} GB | Stall: {res['avg_stall_ms']:.4f} ms | Tps: {res['throughput']:.2f} | Hit: {res['prefetch_hit_rate']:.1f}%")
            
        out_dir = f"/home/palakm/MoEServingSim/evaluation/results/e14_prefetcher/{model_name}"
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "prefetcher_results.json"), "w") as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
