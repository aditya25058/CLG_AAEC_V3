# evaluation/scripts/e14_distributed_prefetcher.py
# FIXED: load_db_traces_with_split appends all experts, train uses all experts,
#        simulation iterates all experts, DeepSeek layers=26
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
    # Group training data by (prompt_id, token_pos) -> {layer -> [expert_ids]}
    transition_counts = {l: defaultdict(lambda: defaultdict(int)) for l in range(1, num_layers + 1)}
    popularity = {l: defaultdict(int) for l in range(num_layers + 1)}
    
    steps = defaultdict(lambda: defaultdict(list))
    for p_id, t_pos, layer, exp_id, _ in train_db:
        steps[(p_id, t_pos)][layer].append(exp_id)
        popularity[layer][exp_id] += 1
        
    for step_key, layers_data in steps.items():
        for l in range(1, num_layers + 1):
            if (l - 1) in layers_data and l in layers_data:
                # FIXED: use ALL previous-layer experts as context
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
    use_prefetch: bool,
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
    
    for p_id in eval_prompt_ids:
        t_positions = sorted(eval_db[p_id].keys())
        
        for l in range(NL + 1):
            gpu_caches[l].clear()
            
        for t in t_positions:
            total_steps += 1
            
            for l in eval_db[p_id][t]:
                experts_at_step = eval_db[p_id][t][l]
                cache = gpu_caches[l]
                
                # Process ALL active experts at this step
                for exp_id, active_cols in experts_at_step:
                    node_id = exp_id // experts_per_node if experts_per_node > 0 else 0
                    is_local = (node_id == 0)
                    
                    active_keys = {(exp_id, col) for col in active_cols}
                    local_active = {k for k in active_keys if k in cache}
                    missed_keys = active_keys - local_active
                    missed_bytes = len(missed_keys) * COLUMN_SIZE_BYTES
                    
                    # Compute prediction accuracy
                    if l in predictor:
                        if l == 0:
                            pred_exp_id = predictor[0][0]
                        elif (l-1) in eval_db[p_id][t]:
                            prev_exp = eval_db[p_id][t][l-1][0][0]  # first expert of prev layer
                            pred_exp_id = predictor[l].get(prev_exp, 0)
                        else:
                            pred_exp_id = predictor[l].get(0, 0)
                        
                        total_predictions += 1
                        if pred_exp_id == exp_id:
                            correct_predictions += 1
                    
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
                                if key in cache:
                                    cache.move_to_end(key)
                                
                    else:
                        # --- PREDICTIVE AAEC ---
                        if l in predictor:
                            if l == 0:
                                pred_exp_id_pf = predictor[0][0]
                            elif (l-1) in eval_db[p_id][t]:
                                prev_exp = eval_db[p_id][t][l-1][0][0]
                                pred_exp_id_pf = predictor[l].get(prev_exp, 0)
                            else:
                                pred_exp_id_pf = predictor[l].get(0, 0)
                        else:
                            pred_exp_id_pf = 0
                        
                        pred_node_id = pred_exp_id_pf // experts_per_node if experts_per_node > 0 else 0
                        pred_is_local = (pred_node_id == 0)
                        
                        pred_active_cols = active_cols if pred_exp_id_pf == exp_id else set(range(cache_capacity_cols))
                        pred_keys = {(pred_exp_id_pf, col) for col in pred_active_cols}
                        pred_missed_keys = {k for k in pred_keys if k not in cache}
                        pred_missed_bytes = len(pred_missed_keys) * COLUMN_SIZE_BYTES
                        
                        if not pred_is_local:
                            network_bytes_transferred += pred_missed_bytes
                            
                        is_correct_prediction = (pred_exp_id_pf == exp_id)
                        
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
                            
    total_network_gb = network_bytes_transferred / 1e9
    avg_stall_ms = (total_stalls_us / 1000.0) / max(1, total_steps)
    
    BASE_COMPUTE_TIME_MS = 1.5
    avg_total_latency_ms = BASE_COMPUTE_TIME_MS + (avg_stall_ms * NL)
    throughput_tokens_sec = 1000.0 / avg_total_latency_ms
    
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
