# evaluation/scripts/e15_batch_scaling_tradeoffs.py
# Simulates I/O cost per token and KV Cache / Weight Cache memory contention as Batch Size scales.
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

# Fixed physical memory budget
VRAM_BUDGET_BYTES = 24 * 1024 * 1024 * 1024  # 24 GB standard GPU (e.g. RTX 3090/4090)
SEQ_LEN = 1024  # Average sequence length for KV cache calculation
LINK_BW_GBPS = 64.0  # PCIe Gen5 (64 GB/s)

def load_all_db_traces(db_path: str):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT prompt_id, token_pos, layer, expert_id, active_indices, energy_k_50
        FROM activations
        ORDER BY prompt_id, token_pos, layer
    """)
    rows = cursor.fetchall()
    conn.close()
    
    db = {}
    for row in rows:
        p_id, t_pos, layer, exp_id, indices_str, k50 = row
        indices = json.loads(indices_str)[:k50]
        active_set = set(indices)
        
        if p_id not in db:
            db[p_id] = {}
        if t_pos not in db[p_id]:
            db[p_id][t_pos] = {}
        if layer not in db[p_id][t_pos]:
            db[p_id][t_pos][layer] = []
        db[p_id][t_pos][layer].append((exp_id, active_set))
        
    return db

def run_batch_simulation(db, batch_size, spec):
    NL = spec["num_layers"]
    NE = spec["num_experts"]
    I = spec["intermediate_dim"]
    H = spec["hidden_size"]
    
    # Calculate KV cache size
    # 2 (K and V) * num_layers * num_heads * d_head * 2 bytes (BF16) * seq_len * batch_size
    # hidden_size = num_heads * d_head = 2048
    kv_bytes_per_user = 2 * NL * H * 2 * SEQ_LEN
    kv_total_bytes = kv_bytes_per_user * batch_size
    kv_total_gb = kv_total_bytes / (1024**3)
    
    # Remaining weight VRAM capacity
    weight_vram_bytes = max(1 * 1024**3, VRAM_BUDGET_BYTES - kv_total_bytes)
    weight_vram_gb = weight_vram_bytes / (1024**3)
    
    # Cache size in columns per expert (C)
    # total_VRAM = NL * NE * C * 3 * H * 2 bytes
    column_size_bytes = 3 * H * 2
    total_experts = NL * NE
    cache_size = int(weight_vram_bytes // (total_experts * column_size_bytes))
    cache_size = max(4, min(I, cache_size))  # Bound between 4 and full intermediate_dim
    
    layer_capacity = cache_size * NE
    
    # Retrieve all distinct prompt IDs and group them into batches
    prompt_ids = list(db.keys())
    np.random.seed(42)
    
    # Select prompts to form the batch
    if len(prompt_ids) < batch_size:
        # Recycle prompts if batch size exceeds available prompts
        batch_prompts = [prompt_ids[i % len(prompt_ids)] for i in range(batch_size)]
    else:
        batch_prompts = list(np.random.choice(prompt_ids, batch_size, replace=False))
        
    # Get distinct token positions
    t_positions = sorted(list(db[prompt_ids[0]].keys()))
    
    gpu_caches = [OrderedDict() for _ in range(NL)]
    
    total_hits = 0
    total_misses = 0
    total_stalls_us = 0
    total_steps = 0
    total_data_moved = 0
    
    # Simulate step-by-step decoding
    for t in t_positions:
        total_steps += 1
        
        for l in range(NL):
            cache = gpu_caches[l]
            
            # Form active union of columns across all users in the batch
            active_keys = set()
            for p_id in batch_prompts:
                # If prompt has finished or is missing this step, skip
                if t not in db[p_id] or l not in db[p_id][t]:
                    continue
                for exp_id, active_cols in db[p_id][t][l]:
                    for col in active_cols:
                        active_keys.add((exp_id, col))
            
            if not active_keys:
                continue
                
            local_active = {k for k in active_keys if k in cache}
            missed = active_keys - local_active
            
            total_hits += len(local_active)
            total_misses += len(missed)
            
            # Update cache with LRU
            for key in active_keys:
                if key in cache:
                    cache.move_to_end(key)
                else:
                    if len(cache) >= layer_capacity:
                        cache.popitem(last=False)
                    cache[key] = True
            
            if missed:
                copy_size = len(missed) * column_size_bytes
                total_data_moved += copy_size
                copy_time = (copy_size / (LINK_BW_GBPS * 1e9)) * 1e6
                
                # FFN compute time scales with batch size
                # 35.8 us base + 10 us per additional batch user
                ffn_time = 35.8 + 10.0 * (batch_size - 1)
                overlap_window = 100.0 + ffn_time  # 100 us attention window
                
                stall = max(0.0, (copy_time + 50.0) - overlap_window)  # 50 us scheduling overhead
                total_stalls_us += stall
                
    total_queries_steps = total_steps * NL
    hit_rate = (total_hits / max(1, total_hits + total_misses)) * 100.0
    avg_stall_ms = (total_stalls_us / 1000.0) / max(1, total_steps)
    total_gb = total_data_moved / (1024**3)
    avg_io_per_token_mb = (total_data_moved / max(1, total_steps)) / (1024**2)
    
    return {
        "kv_gb": kv_total_gb,
        "weight_gb": weight_vram_gb,
        "cache_size": cache_size,
        "hit_rate": hit_rate,
        "stall_ms": avg_stall_ms,
        "data_gb": total_gb,
        "io_token_mb": avg_io_per_token_mb
    }

def main():
    batches = [1, 2, 4, 8, 16, 32, 64]
    
    for model_name, spec in MODELS.items():
        print(f"\n==========================================")
        print(f"BATCH SCALING SIMULATION FOR {model_name.upper()}")
        print(f"==========================================")
        db = load_all_db_traces(spec["db_path"])
        
        print(f"{'Batch':<6} | {'KV Cache':<9} | {'VRAM WCache':<11} | {'Cache Size':<10} | {'Hit Rate':<8} | {'Avg Stall':<9} | {'Total I/O':<9} | {'I/O/Token':<10}")
        print(f"{'Size':<6} | {'(GB)':<9} | {'(GB)':<11} | {'(cols/exp)':<10} | {'(%)':<8} | {'(ms)':<9} | {'(GB)':<9} | {'(MB/step)':<10}")
        print("-" * 90)
        
        for B in batches:
            res = run_batch_simulation(db, B, spec)
            print(f"{B:<6} | {res['kv_gb']:7.2f} | {res['weight_gb']:9.2f} | {res['cache_size']:10} | {res['hit_rate']:7.2f}% | {res['stall_ms']:8.4f} | {res['data_gb']:8.2f} | {res['io_token_mb']:9.2f}")
            
if __name__ == "__main__":
    main()
