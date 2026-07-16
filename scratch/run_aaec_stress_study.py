import os
import json
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

DB_PATH = "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/qwen3_30b_real_v2.db"
OUTPUT_DIR = "/home/palakm/MoEServingSim/qwen3_30b_plots"

def load_traces():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at {DB_PATH}")
    
    print("Loading sequential execution traces from DB...")
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
    print("Training predictor on calibration set...")
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
                
    # Precompute top columns per expert based on calibration profiles
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

def run_stress_simulation(evaluation_db, transition_matrix, top_cols_per_expert, layer_0_most_frequent, policy="aaec", cache_size=16, link_bw_gb_s=2.0):
    COMPUTE_TIME_PER_LAYER_US = 40.0
    LATENCY_OVERHEAD_PER_DMA_US = 0.5
    COLUMN_SIZE_BYTES = 5120 * 2  # BF16
    
    eval_prompt_ids = sorted(evaluation_db.keys())
    
    # Precompute static column tuples for each expert
    static_cols_cache = {}
    for l in range(48):
        for e in range(128):
            static_cols_cache[(l, e)] = [(e, col) for col in top_cols_per_expert[(l, e)]]
            
    layer_cache_capacity = cache_size * 128
    gpu_caches = {l: OrderedDict() for l in range(48)}
    
    total_misses = 0
    total_hits = 0
    total_stalls_us = 0.0
    total_steps = 0
    
    # Prefetch metrics trackers
    pref_hits_count = 0
    pref_misses_count = 0
    total_prefetched_columns = 0
    
    current_prefetch_queue = {}
    prev_token_active_cols = {}

    for p_id in eval_prompt_ids:
        t_positions = sorted(evaluation_db[p_id].keys())
        current_prefetch_queue.clear()
        prev_token_active_cols.clear()
        
        for idx, t in enumerate(t_positions):
            total_steps += 1
            
            # Execute token t
            for l in range(48):
                if l not in evaluation_db[p_id][t]:
                    continue
                exp_id, active_cols = evaluation_db[p_id][t][l]
                
                cache = gpu_caches[l]
                active_keys = {(exp_id, col) for col in active_cols}
                
                # Dynamic prefetch matching
                missed = active_keys - cache.keys()
                pref_hits = set()
                
                if l in current_prefetch_queue:
                    pref_hits = missed.intersection(current_prefetch_queue[l])
                    pref_hits_count += len(pref_hits)
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
                    
                # Calculate stalls
                if missed:
                    copy_size_bytes = len(missed) * COLUMN_SIZE_BYTES
                    copy_time_us = (copy_size_bytes / (link_bw_gb_s * 1e9)) * 1e6
                    stall_us = max(0.0, (copy_time_us + LATENCY_OVERHEAD_PER_DMA_US) - COMPUTE_TIME_PER_LAYER_US)
                    total_stalls_us += stall_us
            
            # Record misses in prefetch queue (columns prefetched but NOT activated in the FFN step)
            for l in current_prefetch_queue:
                if l in evaluation_db[p_id][t]:
                    exp_id, active_cols = evaluation_db[p_id][t][l]
                    active_keys = {(exp_id, col) for col in active_cols}
                    pref_misses = current_prefetch_queue[l] - active_keys
                    pref_misses_count += len(pref_misses)

            # Prepare prefetch queue for next step
            current_prefetch_queue.clear()
            
            if idx < len(t_positions) - 1:
                # ─── REAL CAUSAL PREDICTOR ───
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
                    
                    if l > 0 and (l-1) in evaluation_db[p_id][t]:
                        prev_exp, _ = evaluation_db[p_id][t][l-1]
                        confidence = transition_matrix[l, prev_exp, pred_exp]
                    else:
                        confidence = 1.0
                        
                    # Gating
                    if confidence < 0.05:
                        continue
                        
                    # Temporal-aware speculative prior
                    temp_cols = prev_token_active_cols.get((l, pred_exp), set())
                    pred_cols_set = {(pred_exp, col) for col in temp_cols}
                    
                    static_cols = set(static_cols_cache[(l, pred_exp)][:cache_size])
                    predicted_keys = pred_cols_set.union(static_cols)
                    
                    missing = predicted_keys - cache.keys()
                    if missing:
                        current_prefetch_queue[l] = missing
                        total_prefetched_columns += len(missing)
            
            # Update temporal prior for next token
            prev_token_active_cols.clear()
            for l in range(48):
                if l in evaluation_db[p_id][t]:
                    exp_id, active_cols = evaluation_db[p_id][t][l]
                    prev_token_active_cols[(l, exp_id)] = active_cols

    hit_rate = total_hits / max(1, total_hits + total_misses)
    avg_stall_per_token_ms = (total_stalls_us / 1000.0) / max(1, total_steps)
    
    # Calculate Precision & Recall
    precision = pref_hits_count / max(1, pref_hits_count + pref_misses_count)
    recall = pref_hits_count / max(1, pref_hits_count + total_misses)
    wasted_prefetch_gb = (pref_misses_count * COLUMN_SIZE_BYTES) / 1e9
        
    return {
        "hit_rate": hit_rate,
        "precision": precision,
        "recall": recall,
        "wasted_prefetch_gb": wasted_prefetch_gb,
        "avg_stall_per_token_ms": avg_stall_per_token_ms
    }

def main():
    print("==================================================================")
    print("Executing SOSP AAEC Stress-Test Sweeps (AAEC ONLY)...")
    print("==================================================================")
    
    calibration_db, evaluation_db = load_traces()
    transition_matrix, top_cols_per_expert, layer_0_most_frequent = train_predictor_and_profile(calibration_db)
    
    cache_sizes = [4, 8, 16, 32]
    bandwidths = [1.0, 2.0, 4.0]
    
    stress_results = {}
    
    for bw in bandwidths:
        stress_results[bw] = []
        for cs in cache_sizes:
            print(f"Simulating AAEC | Bandwidth = {bw:.1f} GB/s | Cache Size = {cs:2d}...")
            res = run_stress_simulation(evaluation_db, transition_matrix, top_cols_per_expert, layer_0_most_frequent, policy="aaec", cache_size=cs, link_bw_gb_s=bw)
            stress_results[bw].append({
                "cache_size": cs,
                "hit_rate": res["hit_rate"],
                "precision": res["precision"],
                "recall": res["recall"],
                "wasted_prefetch_gb": res["wasted_prefetch_gb"],
                "avg_stall_ms": res["avg_stall_per_token_ms"]
            })
                
    # Save raw JSON results
    json_path = os.path.join(OUTPUT_DIR, "aaec_stress_study.json")
    with open(json_path, "w") as f:
        json.dump(stress_results, f, indent=4)
    print(f"\nSaved raw stress study results to: {json_path}")
    
    # Plotting results: 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {1.0: "red", 2.0: "orange", 4.0: "green"}
    markers = {1.0: "o", 2.0: "s", 4.0: "D"}
    
    # 1. Effective Hit Rate (%) vs. Cache Size
    ax_hr = axs[0, 0]
    for bw in bandwidths:
        cs_list = [item["cache_size"] for item in stress_results[bw]]
        hr_list = [item["hit_rate"] * 100 for item in stress_results[bw]]
        ax_hr.plot(cs_list, hr_list, marker=markers[bw], color=colors[bw], linewidth=2, label=f"{bw:.1f} GB/s")
    ax_hr.set_title("AAEC Effective Cache Hit Rate vs. Cache Size")
    ax_hr.set_xlabel("Cache Size (Columns per Expert)")
    ax_hr.set_ylabel("Hit Rate (%)")
    ax_hr.set_xticks(cache_sizes)
    ax_hr.grid(True, alpha=0.3)
    ax_hr.legend()
    
    # 2. Prefetch Precision & Recall vs. Cache Size
    ax_pr = axs[0, 1]
    for bw in [2.0]:  # Precision/Recall are mostly independent of bandwidth speed (depend on cache hit state)
        cs_list = cache_sizes
        prec_list = [item["precision"] * 100 for item in stress_results[bw]]
        rec_list = [item["recall"] * 100 for item in stress_results[bw]]
        ax_pr.plot(cs_list, prec_list, 'o--', color='blue', linewidth=2, label="AAEC Precision")
        ax_pr.plot(cs_list, rec_list, 's-', color='darkblue', linewidth=2, label="AAEC Recall")
    ax_pr.set_title("AAEC Prefetch Precision & Recall vs. Cache Size (2.0 GB/s)")
    ax_pr.set_xlabel("Cache Size (Columns per Expert)")
    ax_pr.set_ylabel("Rate (%)")
    ax_pr.set_xticks(cache_sizes)
    ax_pr.grid(True, alpha=0.3)
    ax_pr.legend()
    
    # 3. Wasted Prefetch (GB) vs. Cache Size
    ax_waste = axs[1, 0]
    for bw in bandwidths:
        cs_list = [item["cache_size"] for item in stress_results[bw]]
        waste_list = [item["wasted_prefetch_gb"] for item in stress_results[bw]]
        ax_waste.plot(cs_list, waste_list, marker=markers[bw], color=colors[bw], linewidth=2, label=f"{bw:.1f} GB/s")
    ax_waste.set_title("AAEC Wasted Prefetch Weight Traffic vs. Cache Size")
    ax_waste.set_xlabel("Cache Size (Columns per Expert)")
    ax_waste.set_ylabel("Wasted Data (GB)")
    ax_waste.set_xticks(cache_sizes)
    ax_waste.grid(True, alpha=0.3)
    ax_waste.legend()
    
    # 4. Serving Stall Latency vs. Cache Size
    ax_stall = axs[1, 1]
    for bw in bandwidths:
        cs_list = [item["cache_size"] for item in stress_results[bw]]
        stall_list = [item["avg_stall_ms"] for item in stress_results[bw]]
        ax_stall.plot(cs_list, stall_list, marker=markers[bw], color=colors[bw], linewidth=2, label=f"{bw:.1f} GB/s")
    ax_stall.set_title("AAEC Average Serving Stall Latency vs. Cache Size")
    ax_stall.set_xlabel("Cache Size (Columns per Expert)")
    ax_stall.set_ylabel("Avg Stall Latency (ms/token)")
    ax_stall.set_xticks(cache_sizes)
    ax_stall.grid(True, alpha=0.3)
    ax_stall.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "aaec_stress_study.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved stress study plots to: {plot_path}")

if __name__ == "__main__":
    main()
