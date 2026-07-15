# evaluation/scripts/e14_distributed_prefetcher.py
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
    
    train_db = []
    eval_db = {}
    
    for row in rows:
        p_id, t_pos, layer, exp_id, indices_str, k50 = row
        indices = json.loads(indices_str)[:k50]
        active_set = set(indices)
        
        if p_id in train_prompts:
            train_db.append((p_id, t_pos, layer, exp_id, active_set))
        else:
            if p_id not in eval_db:
                eval_db[p_id] = {}
            if t_pos not in eval_db[p_id]:
                eval_db[p_id][t_pos] = {}
            eval_db[p_id][t_pos][layer] = (exp_id, active_set)
            
    return train_db, eval_db

def train_transition_predictor(train_db, num_layers: int, num_experts: int):
    # Transition count tracker: transition_counts[layer][prev_exp][curr_exp]
    transition_counts = {l: defaultdict(lambda: defaultdict(int)) for l in range(1, num_layers)}
    popularity = {l: defaultdict(int) for l in range(num_layers)}
    
    # Sort training DB by prompt_id, token_pos, layer
    steps = defaultdict(dict)
    for p_id, t_pos, layer, exp_id, _ in train_db:
        steps[(p_id, t_pos)][layer] = exp_id
        popularity[layer][exp_id] += 1
        
    for step_key, layers_data in steps.items():
        for l in range(1, num_layers):
            if (l - 1) in layers_data and l in layers_data:
                prev_exp = layers_data[l - 1]
                curr_exp = layers_data[l]
                transition_counts[l][prev_exp][curr_exp] += 1
                
    # Compile predictor table: predictor[layer][prev_exp] -> predicted_exp
    predictor = {}
    for l in range(1, num_layers):
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
                    
    # Layer 0 predictor defaults to general popularity
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
    use_prefetch: bool,
    world_size: int,
    spec: dict
):
    NL = spec["num_layers"]
    NE = spec["num_experts"]
    H = spec["hidden_size"]
    I = spec["intermediate_dim"]
    
    experts_per_node = NE // world_size
    
    # Timing and bandwidth configurations
    LOCAL_BW_GBPS = 64.0        # PCIe Gen5
    NETWORK_BW_GBPS = 10.0      # 100 Gbps network
    LOCAL_LATENCY_US = 0.5
    NETWORK_LATENCY_US = 5.0
    COMPUTE_TIME_PER_LAYER_US = 50.0    # Phase 1 compute window
    ATTENTION_COMPUTE_TIME_US = 100.0   # Attention block compute window
    
    COLUMN_SIZE_BYTES = 3 * H * 2
    
    # Initialize cache per layer (Size 32 columns/expert)
    cache_capacity_cols = 32
    gpu_caches = {l: OrderedDict() for l in range(NL)}
    column_capacity = cache_capacity_cols * NE
    
    total_steps = 0
    network_bytes_transferred = 0
    total_stalls_us = 0.0
    
    eval_prompt_ids = sorted(eval_db.keys())
    
    for p_id in eval_prompt_ids:
        t_positions = sorted(eval_db[p_id].keys())
        
        # Reset cache on prompt boundary
        for l in range(NL):
            gpu_caches[l].clear()
            
        for t in t_positions:
            total_steps += 1
            
            # Predictor state: track what we predicted for each layer
            predicted_experts_for_step = {}
            # Generate predictions for this token step
            predicted_experts_for_step[0] = predictor[0][0]
            for l in range(1, NL):
                if (l-1) in eval_db[p_id][t]:
                    prev_exp, _ = eval_db[p_id][t][l-1]
                    predicted_experts_for_step[l] = predictor[l][prev_exp]
                else:
                    predicted_experts_for_step[l] = predictor[l][0]
            
            # Step loop through layers
            for l in range(NL):
                if l not in eval_db[p_id][t]:
                    continue
                exp_id, active_cols = eval_db[p_id][t][l]
                
                # Check actual location
                node_id = exp_id // experts_per_node
                is_local = (node_id == 0)
                
                cache = gpu_caches[l]
                
                # Caching logic
                active_keys = {(exp_id, col) for col in active_cols}
                local_active = active_keys.intersection(cache.keys())
                missed_keys = active_keys - local_active
                
                # Calculate actual miss bytes needed
                missed_bytes = len(missed_keys) * COLUMN_SIZE_BYTES
                
                if not use_prefetch:
                    # --- REACTIVE AAEC ---
                    if missed_keys:
                        for key in missed_keys:
                            if len(cache) >= column_capacity:
                                cache.popitem(last=False)
                            cache[key] = True
                        
                        if not is_local:
                            network_bytes_transferred += missed_bytes
                            t_transfer = (missed_bytes / (NETWORK_BW_GBPS * 1e9)) * 1e6 + NETWORK_LATENCY_US
                        else:
                            t_transfer = (missed_bytes / (LOCAL_BW_GBPS * 1e9)) * 1e6 + LOCAL_LATENCY_US
                            
                        stall = max(0.0, t_transfer - COMPUTE_TIME_PER_LAYER_US)
                        total_stalls_us += stall
                    else:
                        for key in active_keys:
                            cache.move_to_end(key)
                            
                else:
                    # --- PREDICTIVE AAEC ---
                    pred_exp_id = predicted_experts_for_step[l]
                    pred_node_id = pred_exp_id // experts_per_node
                    pred_is_local = (pred_node_id == 0)
                    
                    pred_active_cols = active_cols if pred_exp_id == exp_id else set(range(cache_capacity_cols))
                    pred_keys = {(pred_exp_id, col) for col in pred_active_cols}
                    pred_missed_keys = pred_keys - set(cache.keys())
                    pred_missed_bytes = len(pred_missed_keys) * COLUMN_SIZE_BYTES
                    
                    # Trigger speculative prefetch network traffic if remote
                    if not pred_is_local:
                        network_bytes_transferred += pred_missed_bytes
                        
                    is_correct_prediction = (pred_exp_id == exp_id)
                    
                    if is_correct_prediction:
                        for key in pred_missed_keys:
                            if len(cache) >= column_capacity:
                                cache.popitem(last=False)
                            cache[key] = True
                            
                        for key in active_keys:
                            if key in cache:
                                cache.move_to_end(key)
                                
                        if missed_keys:
                            overlap_window = ATTENTION_COMPUTE_TIME_US + COMPUTE_TIME_PER_LAYER_US
                            if not is_local:
                                t_transfer = (missed_bytes / (NETWORK_BW_GBPS * 1e9)) * 1e6 + NETWORK_LATENCY_US
                            else:
                                t_transfer = (missed_bytes / (LOCAL_BW_GBPS * 1e9)) * 1e6 + LOCAL_LATENCY_US
                                
                            stall = max(0.0, t_transfer - overlap_window)
                            total_stalls_us += stall
                    else:
                        # Misprediction
                        if missed_keys:
                            for key in missed_keys:
                                if len(cache) >= column_capacity:
                                    cache.popitem(last=False)
                                cache[key] = True
                                
                            if not is_local:
                                network_bytes_transferred += missed_bytes
                                t_transfer = (missed_bytes / (NETWORK_BW_GBPS * 1e9)) * 1e6 + NETWORK_LATENCY_US
                            else:
                                t_transfer = (missed_bytes / (LOCAL_BW_GBPS * 1e9)) * 1e6 + LOCAL_LATENCY_US
                                
                            stall = max(0.0, t_transfer - COMPUTE_TIME_PER_LAYER_US)
                            total_stalls_us += stall
                            
    # End-to-end statistics
    total_network_gb = network_bytes_transferred / 1e9
    avg_stall_ms = (total_stalls_us / 1000.0) / max(1, total_steps)
    
    # Prefetch hit rate (only computed on remote nodes where prefetch is relevant)
    # Or globally: total correct predictions / total steps
    
    # Calculate throughput (tokens/sec)
    BASE_COMPUTE_TIME_MS = 1.5
    avg_total_latency_ms = BASE_COMPUTE_TIME_MS + (avg_stall_ms * NL)
    throughput_tokens_sec = 1000.0 / avg_total_latency_ms
    
    # Calculate prefetch hit rate from simulation runs
    # To compute prefetch success rate, we track how many times the predicted expert matched actual expert
    correct_predictions = 0
    total_predictions = 0
    for p_id in eval_prompt_ids:
        t_positions = sorted(eval_db[p_id].keys())
        for t in t_positions:
            for l in range(NL):
                if l not in eval_db[p_id][t]:
                    continue
                exp_id, _ = eval_db[p_id][t][l]
                pred_exp_id = predictor[l][eval_db[p_id][t][l-1][0]] if l > 0 and (l-1) in eval_db[p_id][t] else predictor[l][0]
                if pred_exp_id == exp_id:
                    correct_predictions += 1
                total_predictions += 1
                
    prefetch_hit_rate = (correct_predictions / max(1, total_predictions)) * 100.0
    
    return {
        "network_gb": total_network_gb,
        "avg_stall_ms": avg_stall_ms,
        "throughput": throughput_tokens_sec,
        "prefetch_hit_rate": prefetch_hit_rate
    }

def main():
    world_sizes = [4, 8, 16]
    
    for model_name, spec in MODELS.items():
        print(f"Running Distributed Prefetcher Simulation for {model_name}...")
        train_db, eval_db = load_db_traces_with_split(spec["db_path"])
        
        predictor = train_transition_predictor(train_db, spec["num_layers"], spec["num_experts"])
        
        results = {
            "reactive": {},
            "predictive": {}
        }
        
        for ws in world_sizes:
            print(f"  Testing cluster scale: {ws} nodes...")
            res_react = run_distributed_prefetcher_simulation(eval_db, predictor, False, ws, spec)
            res_pred = run_distributed_prefetcher_simulation(eval_db, predictor, True, ws, spec)
            
            results["reactive"][str(ws)] = res_react
            results["predictive"][str(ws)] = res_pred
            
            print(f"    Reactive   | Net Moved: {res_react['network_gb']:6.2f} GB | Stall: {res_react['avg_stall_ms']:.4f} ms | Tps: {res_react['throughput']:.2f}")
            print(f"    Predictive | Net Moved: {res_pred['network_gb']:6.2f} GB | Stall: {res_pred['avg_stall_ms']:.4f} ms | Tps: {res_pred['throughput']:.2f} | Pred Acc: {res_pred['prefetch_hit_rate']:.1f}%")
            
        out_dir = f"/home/palakm/MoEServingSim/evaluation/results/e14_prefetcher/{model_name}"
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "prefetcher_results.json"), "w") as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
