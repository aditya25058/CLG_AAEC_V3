# evaluation/scripts/e18_predictor_sweep.py
# Sweeps predictor accuracy from 0% to 100% to evaluate impact on stall and network bandwidth overhead.
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

def load_db_eval_traces(db_path: str):
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
    eval_prompts = set(prompt_ids[split_idx:])
    eval_prompts = sorted(list(eval_prompts))[:5]
    eval_prompts = set(eval_prompts)
    
    eval_db = {}
    for row in rows:
        p_id, t_pos, layer, exp_id, indices_str, k50 = row
        indices = json.loads(indices_str)[:k50]
        active_set = set(indices)
        
        if p_id in eval_prompts:
            if p_id not in eval_db:
                eval_db[p_id] = {}
            if t_pos not in eval_db[p_id]:
                eval_db[p_id][t_pos] = {}
            if layer not in eval_db[p_id][t_pos]:
                eval_db[p_id][t_pos][layer] = []
            eval_db[p_id][t_pos][layer].append((exp_id, active_set))
            
    return eval_db

def run_sweep_simulation(eval_db, target_accuracy, granularity, spec):
    NL = spec["num_layers"]
    NE = spec["num_experts"]
    H = spec["hidden_size"]
    I = spec["intermediate_dim"]
    
    LINK_BW_GBPS = 16.0  # PCIe Gen4 (16 GB/s)
    COMPUTE_TIME_PER_LAYER_US = 50.0
    ATTENTION_COMPUTE_TIME_US = 100.0
    COLUMN_SIZE_BYTES = 3 * H * 2
    
    cache_capacity_cols = 32
    gpu_caches = {l: OrderedDict() for l in range(NL + 1)}
    
    if granularity == "expert":
        layer_capacity = cache_capacity_cols # Expert cache size (number of experts)
    else:
        layer_capacity = cache_capacity_cols * NE # Column cache capacity (total columns)
        
    total_steps = 0
    total_bytes_moved = 0
    total_stalls_us = 0.0
    
    # Pre-seed random generator for deterministic simulation
    np.random.seed(42)
    
    eval_prompt_ids = sorted(eval_db.keys())
    
    for p_id in eval_prompt_ids:
        t_positions = sorted(eval_db[p_id].keys())
        
        for l in range(NL + 1):
            gpu_caches[l].clear()
            
        for t in t_positions:
            total_steps += 1
            
            # Predict and prefetch
            predicted_prefetches = defaultdict(set)
            
            for l in range(NL + 1):
                if l not in eval_db[p_id][t]:
                    continue
                experts_at_step = eval_db[p_id][t][l]
                
                for exp_id, active_cols in experts_at_step:
                    # Roll target accuracy
                    is_correct = np.random.rand() < target_accuracy
                    
                    if is_correct:
                        pred_exp = exp_id
                    else:
                        # Pick a random incorrect expert
                        incorrect = list(set(range(NE)) - {exp_id})
                        pred_exp = np.random.choice(incorrect) if incorrect else 0
                        
                    if granularity == "expert":
                        predicted_prefetches[l].add(pred_exp)
                    else:
                        predicted_prefetches[l].update({(pred_exp, col) for col in range(cache_capacity_cols)})
                        
            # Apply prefetch transfers to cache
            for l, keys in predicted_prefetches.items():
                cache = gpu_caches[l]
                missing = {k for k in keys if k not in cache}
                if missing:
                    if granularity == "expert":
                        total_bytes_moved += len(missing) * I * COLUMN_SIZE_BYTES
                    else:
                        total_bytes_moved += len(missing) * COLUMN_SIZE_BYTES
                    for key in missing:
                        if len(cache) >= layer_capacity:
                            cache.popitem(last=False)
                        cache[key] = True
                        
            # Execute step compute and measure stalls
            for l in range(NL + 1):
                if l not in eval_db[p_id][t]:
                    continue
                experts_at_step = eval_db[p_id][t][l]
                cache = gpu_caches[l]
                
                if granularity == "expert":
                    active_keys = {exp_id for exp_id, _ in experts_at_step}
                else:
                    active_keys = set()
                    for exp_id, active_cols in experts_at_step:
                        for col in active_cols:
                            active_keys.add((exp_id, col))
                            
                local_active = {k for k in active_keys if k in cache}
                missed_keys = active_keys - local_active
                
                # Update LRU
                for key in active_keys:
                    if key in cache:
                        cache.move_to_end(key)
                    else:
                        if len(cache) >= layer_capacity:
                            cache.popitem(last=False)
                        cache[key] = True
                        
                # Stall calculation
                if missed_keys:
                    if granularity == "expert":
                        miss_bytes = len(missed_keys) * I * COLUMN_SIZE_BYTES
                    else:
                        miss_bytes = len(missed_keys) * COLUMN_SIZE_BYTES
                        
                    total_bytes_moved += miss_bytes
                    t_transfer = (miss_bytes / (LINK_BW_GBPS * 1e9)) * 1e6
                    overlap_window = COMPUTE_TIME_PER_LAYER_US + ATTENTION_COMPUTE_TIME_US
                    stall = max(0.0, t_transfer - overlap_window)
                    total_stalls_us += stall
                    
    avg_stall_ms = (total_stalls_us / 1000.0) / max(1, total_steps)
    total_gb = total_bytes_moved / 1e9
    
    BASE_COMPUTE_TIME_MS = 1.5
    avg_total_latency_ms = BASE_COMPUTE_TIME_MS + (avg_stall_ms * NL)
    throughput = 1000.0 / avg_total_latency_ms
    
    return {
        "avg_stall_ms": avg_stall_ms,
        "total_gb": total_gb,
        "throughput": throughput
    }

def main():
    accuracies = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    for model_name, spec in MODELS.items():
        print(f"\n==========================================")
        print(f"PREDICTOR ACCURACY SWEEP FOR {model_name.upper()}")
        print(f"==========================================")
        eval_db = load_db_eval_traces(spec["db_path"])
        
        results = {
            "column_level": [],
            "expert_level": []
        }
        
        for granularity in ["column", "expert"]:
            print(f"Granularity: {granularity.upper()}")
            for acc in accuracies:
                res = run_sweep_simulation(eval_db, acc, granularity, spec)
                results[f"{granularity}_level"].append({
                    "target_accuracy": acc,
                    "avg_stall_ms": res["avg_stall_ms"],
                    "total_gb": res["total_gb"],
                    "throughput": res["throughput"]
                })
                print(f"  Acc: {acc*100:3.0f}% | Stall: {res['avg_stall_ms']:.4f} ms | Data: {res['total_gb']:6.2f} GB | Throughput: {res['throughput']:.2f} tps")
                
        out_dir = f"/home/palakm/MoEServingSim/evaluation/results/e18_sweep/{model_name}"
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "predictor_sweep_results.json"), "w") as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
