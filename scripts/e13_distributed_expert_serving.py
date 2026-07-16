# evaluation/scripts/e13_distributed_expert_serving.py
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
        ORDER BY prompt_id, token_pos, layer
    """)
    rows = cursor.fetchall()
    conn.close()
    
    prompt_ids = sorted(list(set(row[0] for row in rows)))
    split_idx = len(prompt_ids) // 2
    eval_prompts = set(prompt_ids[split_idx:])
    eval_prompts = sorted(list(eval_prompts))[:5]
    eval_prompts = set(eval_prompts)
    
    evaluation_db = {}
    for row in rows:
        p_id, t_pos, layer, exp_id, indices_str, k50 = row
        if p_id not in eval_prompts:
            continue
        indices = json.loads(indices_str)[:k50]
        active_set = set(indices)
        
        if p_id not in evaluation_db:
            evaluation_db[p_id] = {}
        if t_pos not in evaluation_db[p_id]:
            evaluation_db[p_id][t_pos] = {}
        if layer not in evaluation_db[p_id][t_pos]:
            evaluation_db[p_id][t_pos][layer] = []
        evaluation_db[p_id][t_pos][layer].append((exp_id, active_set))
        
    return evaluation_db

def run_distributed_simulation(
    evaluation_db,
    system_type: str,
    cache_capacity_cols_per_exp: int,
    spec: dict
):
    NL = spec["num_layers"]
    NE = spec["num_experts"]
    H = spec["hidden_size"]
    I = spec["intermediate_dim"]
    
    # 4 Nodes: Experts are partitioned evenly
    experts_per_node = NE // 4
    
    LOCAL_BW_GBPS = 64.0
    NETWORK_BW_GBPS = 10.0
    LOCAL_LATENCY_US = 0.5
    NETWORK_LATENCY_US = 5.0
    COMPUTE_TIME_PER_LAYER_US = 50.0
    
    COLUMN_SIZE_BYTES = 3 * H * 2
    FULL_EXPERT_SIZE_BYTES = I * COLUMN_SIZE_BYTES
    
    gpu_caches = {l: OrderedDict() for l in range(NL + 1)}
    
    total_cols_capacity = cache_capacity_cols_per_exp * NE
    
    if system_type == "expert_cache":
        expert_capacity = max(1, total_cols_capacity // I)
    else:
        column_capacity = total_cols_capacity
        
    total_steps = 0
    network_bytes_transferred = 0
    network_fetches_count = 0
    total_stalls_us = 0.0
    
    eval_prompt_ids = sorted(evaluation_db.keys())
    
    for p_id in eval_prompt_ids:
        t_positions = sorted(evaluation_db[p_id].keys())
        
        for l in range(NL + 1):
            gpu_caches[l].clear()
            
        for t in t_positions:
            total_steps += 1
            
            for l in evaluation_db[p_id][t]:
                experts_at_step = evaluation_db[p_id][t][l]
                
                # Process ALL active experts at this step
                for exp_id, active_cols in experts_at_step:
                    node_id = exp_id // experts_per_node if experts_per_node > 0 else 0
                    is_local = (node_id == 0)
                    
                    cache = gpu_caches[l]
                    
                    if system_type == "demand":
                        transferred_bytes = FULL_EXPERT_SIZE_BYTES
                        
                        if not is_local:
                            network_bytes_transferred += transferred_bytes
                            network_fetches_count += 1
                            t_transfer = (transferred_bytes / (NETWORK_BW_GBPS * 1e9)) * 1e6 + NETWORK_LATENCY_US
                        else:
                            t_transfer = (transferred_bytes / (LOCAL_BW_GBPS * 1e9)) * 1e6 + LOCAL_LATENCY_US
                            
                        stall = max(0.0, t_transfer - COMPUTE_TIME_PER_LAYER_US)
                        total_stalls_us += stall
                        
                    elif system_type == "expert_cache":
                        if exp_id in cache:
                            cache.move_to_end(exp_id)
                            stall = 0.0
                        else:
                            transferred_bytes = FULL_EXPERT_SIZE_BYTES
                            
                            if len(cache) >= expert_capacity:
                                cache.popitem(last=False)
                            cache[exp_id] = True
                            
                            if not is_local:
                                network_bytes_transferred += transferred_bytes
                                network_fetches_count += 1
                                t_transfer = (transferred_bytes / (NETWORK_BW_GBPS * 1e9)) * 1e6 + NETWORK_LATENCY_US
                            else:
                                t_transfer = (transferred_bytes / (LOCAL_BW_GBPS * 1e9)) * 1e6 + LOCAL_LATENCY_US
                                
                            stall = max(0.0, t_transfer - COMPUTE_TIME_PER_LAYER_US)
                            total_stalls_us += stall
                            
                    elif system_type == "aaec_column_cache":
                        active_keys = {(exp_id, col) for col in active_cols}
                        local_active = {k for k in active_keys if k in cache}
                        missed_keys = active_keys - local_active
                        
                        if missed_keys:
                            missed_bytes = len(missed_keys) * COLUMN_SIZE_BYTES
                            
                            for key in missed_keys:
                                if len(cache) >= column_capacity:
                                    cache.popitem(last=False)
                                cache[key] = True
                            
                            if not is_local:
                                network_bytes_transferred += missed_bytes
                                network_fetches_count += 1
                                t_transfer = (missed_bytes / (NETWORK_BW_GBPS * 1e9)) * 1e6 + NETWORK_LATENCY_US
                            else:
                                t_transfer = (missed_bytes / (LOCAL_BW_GBPS * 1e9)) * 1e6 + LOCAL_LATENCY_US
                                
                            stall = max(0.0, t_transfer - COMPUTE_TIME_PER_LAYER_US)
                            total_stalls_us += stall
                        else:
                            for key in active_keys:
                                cache.move_to_end(key)
                        
    total_network_gb = network_bytes_transferred / 1e9
    avg_fetch_size_kb = (network_bytes_transferred / max(1, network_fetches_count)) / 1024.0
    avg_stall_ms = (total_stalls_us / 1000.0) / max(1, total_steps)
    
    BASE_COMPUTE_TIME_MS = 1.5
    avg_total_latency_ms = BASE_COMPUTE_TIME_MS + (avg_stall_ms * NL)
    throughput_tokens_sec = 1000.0 / avg_total_latency_ms
    
    return {
        "network_gb": total_network_gb,
        "avg_fetch_kb": avg_fetch_size_kb,
        "avg_stall_ms": avg_stall_ms,
        "throughput": throughput_tokens_sec
    }

def main():
    systems = ["demand", "expert_cache", "aaec_column_cache"]
    cache_sizes = [16, 32, 64, 128, 256, 512]
    
    for model_name, spec in MODELS.items():
        print(f"Running Distributed Serving Simulation for {model_name}...")
        eval_db = load_db_traces(spec["db_path"])
        
        results = {sys_type: {} for sys_type in systems}
        
        for size in cache_sizes:
            print(f"  Testing Cache Capacity equivalent to {size} cols/expert...")
            for sys_type in systems:
                res = run_distributed_simulation(eval_db, sys_type, size, spec)
                results[sys_type][str(size)] = res
                print(f"    System: {sys_type:<18} | Net Moved: {res['network_gb']:6.2f} GB | Avg Fetch: {res['avg_fetch_kb']:7.1f} KB | Stall: {res['avg_stall_ms']:.4f} ms | Tps: {res['throughput']:.2f}")
                
        out_dir = f"/home/palakm/MoEServingSim/evaluation/results/e13_distributed/{model_name}"
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "distributed_results.json"), "w") as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
