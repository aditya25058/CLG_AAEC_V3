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
    
    print(f"Calibration prompts: {calib_prompts}")
    print(f"Evaluation prompts: {eval_prompts}")
    
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
    print("Training causal predictor and profiling column popularity on calibration set...")
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
                
    # Calculate layer entropy for AdapMoE cache allocation
    layer_entropies = np.zeros(48)
    for l in range(48):
        total = layer_expert_counts[l].sum()
        if total > 0:
            probs = layer_expert_counts[l] / total
            probs = probs[probs > 0]
            layer_entropies[l] = -np.sum(probs * np.log2(probs))
        else:
            layer_entropies[l] = 0.0
            
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
                
    # Find most frequent expert in Layer 0 for non-oracle Layer 0 predictor
    layer_0_most_frequent = np.argmax(layer_expert_counts[0])
            
    return transition_matrix, layer_entropies, top_cols_per_expert, layer_0_most_frequent

def run_sota_simulation(evaluation_db, transition_matrix, layer_entropies, top_cols_per_expert, layer_0_most_frequent, policy="aaec", cache_size=32, link_bw_gb_s=8.0, cpu_compute_delay_us=10.0):
    COMPUTE_TIME_PER_LAYER_US = 40.0
    LATENCY_OVERHEAD_PER_DMA_US = 0.5
    COLUMN_SIZE_BYTES = 5120 * 2  # BF16
    
    eval_prompt_ids = sorted(evaluation_db.keys())
    
    # Precompute static column tuples for each expert
    static_cols_cache = {}
    for l in range(48):
        for e in range(128):
            static_cols_cache[(l, e)] = [(e, col) for col in top_cols_per_expert[(l, e)]]
            
    # Calculate cache budgets per layer
    layer_cache_capacities = {}
    if policy == "adapmoe":
        entropy_sum = layer_entropies.sum()
        for l in range(48):
            fraction = layer_entropies[l] / entropy_sum if entropy_sum > 0 else 1.0 / 48.0
            layer_cache_capacities[l] = int(fraction * cache_size * 128 * 48)
    else:
        for l in range(48):
            layer_cache_capacities[l] = cache_size * 128
            
    # Initialize Global Column Cache per layer as OrderedDict for O(1) LRU ops
    gpu_caches = {l: OrderedDict() for l in range(48)}
    
    # ─── PowerInfer-2 static cache initialization ───
    if policy == "powerinfer2":
        for l in range(48):
            capacity = layer_cache_capacities[l]
            added = 0
            col_idx = 0
            while added < capacity and col_idx < 768:
                for e in range(128):
                    if added >= capacity:
                        break
                    col = top_cols_per_expert[(l, e)][col_idx]
                    key = (e, col)
                    gpu_caches[l][key] = True
                    added += 1
                col_idx += 1
                
    total_misses = 0
    total_hits = 0
    total_prefetched_bytes = 0
    total_pushed_bytes = 0
    total_stalls_us = 0.0
    total_steps = 0
    
    # Predictor accuracy logging
    pred_correct_per_layer = np.zeros(48)
    pred_total_per_layer = np.zeros(48)
    
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
                
                if policy == "demand":
                    # Direct Offloading: Cache is disabled. Every token fetches on demand.
                    missed = active_keys
                    hits = 0
                    total_misses += len(missed)
                    
                    if missed:
                        copy_size_bytes = len(missed) * COLUMN_SIZE_BYTES
                        copy_time_us = (copy_size_bytes / (link_bw_gb_s * 1e9)) * 1e6
                        total_pushed_bytes += copy_size_bytes
                        stall_us = max(0.0, (copy_time_us + LATENCY_OVERHEAD_PER_DMA_US) - COMPUTE_TIME_PER_LAYER_US)
                        total_stalls_us += stall_us
                        
                elif policy == "powerinfer2":
                    local_active = active_keys.intersection(cache.keys())
                    missed = active_keys - local_active
                    
                    hits = len(local_active)
                    total_hits += hits
                    total_misses += len(missed)
                    
                    if missed:
                        cpu_latency = len(missed) * cpu_compute_delay_us
                        total_stalls_us += cpu_latency
                        
                else:
                    # GPU-only caching / prefetching policies
                    missed = active_keys - cache.keys()
                    pref_hits = set()
                    
                    if policy in ["promoe", "hobbit", "aaec"] and l in current_prefetch_queue:
                        pref_hits = missed.intersection(current_prefetch_queue[l])
                        missed = missed - pref_hits
                        
                    hits = len(active_keys) - len(missed)
                    total_hits += hits
                    total_misses += len(missed)
                    
                    # Update global cache state
                    capacity = layer_cache_capacities[l]
                    for key in active_keys:
                        if key in cache:
                            cache.move_to_end(key)
                        else:
                            if len(cache) >= capacity:
                                cache.popitem(last=False)
                            cache[key] = True
                        
                    # Calculate stalls
                    if missed:
                        copy_size_bytes = len(missed) * COLUMN_SIZE_BYTES
                        if policy == "hobbit":
                            copy_size_bytes = len(missed) * (5120 // 2)
                        
                        copy_time_us = (copy_size_bytes / (link_bw_gb_s * 1e9)) * 1e6
                        total_pushed_bytes += copy_size_bytes
                        
                        dequant_us = len(missed) * 2.0 if policy == "hobbit" else 0.0
                        
                        stall_us = max(0.0, (copy_time_us + LATENCY_OVERHEAD_PER_DMA_US + dequant_us) - COMPUTE_TIME_PER_LAYER_US)
                        total_stalls_us += stall_us
            
            # Prepare prefetch queue for next step
            current_prefetch_queue.clear()
            
            if idx < len(t_positions) - 1:
                # ─── REAL CAUSAL PREDICTOR ───
                for l in range(48):
                    if l == 0:
                        # Non-oracle Layer 0 predictor: use the most frequent expert in calibration
                        pred_exp = layer_0_most_frequent
                    else:
                        if (l-1) in evaluation_db[p_id][t]:
                            prev_exp, _ = evaluation_db[p_id][t][l-1]
                            probs = transition_matrix[l, prev_exp]
                            pred_exp = np.argmax(probs)
                        else:
                            pred_exp = 0
                            
                    # Log predictor accuracy
                    if l in evaluation_db[p_id][t]:
                        actual_exp, _ = evaluation_db[p_id][t][l]
                        pred_correct_per_layer[l] += (pred_exp == actual_exp)
                        pred_total_per_layer[l] += 1
                            
                    cache = gpu_caches[l]
                    
                    if policy == "promoe":
                        pred_cols = static_cols_cache[(l, pred_exp)][:cache_size]
                        missing = set(pred_cols) - cache.keys()
                        if missing:
                            current_prefetch_queue[l] = missing
                            total_prefetched_bytes += len(missing) * COLUMN_SIZE_BYTES
                            
                    elif policy == "hobbit":
                        pred_cols = static_cols_cache[(l, pred_exp)][:cache_size]
                        missing = set(pred_cols) - cache.keys()
                        if missing:
                            current_prefetch_queue[l] = missing
                            total_prefetched_bytes += len(missing) * (5120 // 2)
                            
                    elif policy == "aaec":
                        if l > 0 and (l-1) in evaluation_db[p_id][t]:
                            prev_exp, _ = evaluation_db[p_id][t][l-1]
                            confidence = transition_matrix[l, prev_exp, pred_exp]
                        else:
                            confidence = 1.0
                            
                        if confidence < 0.05:
                            continue
                            
                        # Column-level temporal prior
                        temp_cols = prev_token_active_cols.get((l, pred_exp), set())
                        pred_cols_set = {(pred_exp, col) for col in temp_cols}
                        
                        # Fallback static prior
                        static_cols = set(static_cols_cache[(l, pred_exp)][:cache_size])
                        predicted_keys = pred_cols_set.union(static_cols)
                        
                        missing = predicted_keys - cache.keys()
                        if missing:
                            current_prefetch_queue[l] = missing
                            total_prefetched_bytes += len(missing) * COLUMN_SIZE_BYTES
            
            # Update temporal prior for next token
            prev_token_active_cols.clear()
            for l in range(48):
                if l in evaluation_db[p_id][t]:
                    exp_id, active_cols = evaluation_db[p_id][t][l]
                    prev_token_active_cols[(l, exp_id)] = active_cols

    # Log predictor accuracy stats
    layer_accuracies = {}
    for l in range(48):
        if pred_total_per_layer[l] > 0:
            layer_accuracies[l] = pred_correct_per_layer[l] / pred_total_per_layer[l]
            
    hit_rate = total_hits / max(1, total_hits + total_misses)
    if policy == "demand":
        hit_rate = 0.0
        
    total_transferred_gb = (total_prefetched_bytes + total_pushed_bytes) / 1e9
    avg_stall_per_token_ms = (total_stalls_us / 1000.0) / max(1, total_steps)
    
    if policy == "hobbit":
        wasted_prefetch_gb = max(0.0, (total_prefetched_bytes - (total_hits * (5120 // 2))) / 1e9)
    else:
        wasted_prefetch_gb = max(0.0, (total_prefetched_bytes - (total_hits * COLUMN_SIZE_BYTES)) / 1e9)
        
    return {
        "hit_rate": hit_rate,
        "total_transferred_gb": total_transferred_gb,
        "wasted_prefetch_gb": wasted_prefetch_gb,
        "avg_stall_per_token_ms": avg_stall_per_token_ms,
        "layer_accuracies": layer_accuracies
    }

def main():
    print("==================================================================")
    print("Executing SOSP-Ready Causal SOTA Comparison Sweeps (Cache Size = 32)...")
    print("==================================================================")
    
    calibration_db, evaluation_db = load_traces()
    transition_matrix, layer_entropies, top_cols_per_expert, layer_0_most_frequent = train_predictor_and_profile(calibration_db)
    
    link_speeds = [2.0, 8.0, 16.0, 64.0]
    # We added 'lru' (caching only, no prefetching) and mapped 'demand' to true direct offloading (no caching)
    baselines = ["demand", "lru", "powerinfer2", "promoe", "adapmoe", "hobbit", "aaec"]
    
    results = {b: [] for b in baselines}
    
    for bw in link_speeds:
        print(f"\nSimulating Link Bandwidth = {bw:.1f} GB/s...")
        for b in baselines:
            res = run_sota_simulation(evaluation_db, transition_matrix, layer_entropies, top_cols_per_expert, layer_0_most_frequent, b, cache_size=32, link_bw_gb_s=bw)
            results[b].append(res)
            
    print("\n" + "="*165)
    print(f"{'Bandwidth (GB/s)':<18} | {'Policy':<30} | {'Effective Hit Rate':<20} | {'PCIe Stall/Token (ms)':<22} | {'Total Data (GB)':<16} | {'Wasted Prefetch (GB)':<20}")
    print("-"*165)
    for idx, bw in enumerate(link_speeds):
        for b in baselines:
            r = results[b][idx]
            name_map = {
                "demand": "Direct Offloading (Demand)",
                "lru": "LRU Cache (No Prefetch)",
                "powerinfer2": "PowerInfer-2",
                "promoe": "ProMoE",
                "adapmoe": "AdapMoE",
                "hobbit": "HOBBIT",
                "aaec": "AAEC (Ours)"
            }
            print(f"{bw:<18.1f} | {name_map[b]:<30} | {r['hit_rate']*100:<18.2f}% | {r['avg_stall_per_token_ms']:<20.4f} | {r['total_transferred_gb']:<16.4f} | {r['wasted_prefetch_gb']:<20.4f}")
        print("-"*165)
        
    # Print Predictor Accuracy metrics for reviewers
    predictor_stats = results["aaec"][1]["layer_accuracies"]
    print("\n" + "="*50)
    print("REVIEWER TELEMETRY: CAUSAL PREDICTOR ACCURACY PER LAYER")
    print("="*50)
    total_acc = 0.0
    for l in range(48):
        acc = predictor_stats.get(l, 0.0) * 100
        total_acc += acc
        print(f"  Layer {l:2d}: {acc:6.2f}% Accuracy")
    print("-"*50)
    print(f"  Overall Average Predictor Accuracy = {total_acc/48:.2f}%")
    print("="*50)
        
    # Generate SOTA Comparison Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    policies_labels = ["Demand", "LRU (No Pref)", "PowerInfer-2", "ProMoE", "AdapMoE", "HOBBIT", "AAEC (Ours)"]
    hit_rates = [results[b][1]['hit_rate']*100 for b in baselines]
    ax1.bar(policies_labels, hit_rates, color=['black', 'gray', 'orange', 'purple', 'blue', 'brown', 'green'], alpha=0.8)
    ax1.set_title("Effective Cache Hit Rate (%) at 8.0 GB/s")
    ax1.set_ylabel("Hit Rate (%)")
    ax1.set_ylim(0, 100)
    for i, v in enumerate(hit_rates):
        ax1.text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold', fontsize=9)
        
    stalls = [results[b][1]['avg_stall_per_token_ms'] for b in baselines]
    ax2.bar(policies_labels, stalls, color=['black', 'gray', 'orange', 'purple', 'blue', 'brown', 'green'], alpha=0.8)
    ax2.set_title("Average PCIe Stall per Token (ms) at 8.0 GB/s")
    ax2.set_ylabel("Stall Latency (ms)")
    for i, v in enumerate(stalls):
        ax2.text(i, v + 0.1, f"{v:.3f}ms", ha='center', fontweight='bold', fontsize=9)
        
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "aaec_sota_comparisons.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\nSaved SOTA comparison plot to: {plot_path}")
    
    # Save results to JSON
    json_path = os.path.join(OUTPUT_DIR, "aaec_sota_comparisons.json")
    with open(json_path, "w") as f:
        # Strip numpy collections for JSON compatibility
        serializable_results = {}
        for b in baselines:
            serializable_results[b] = []
            for item in results[b]:
                serializable_results[b].append({
                    "hit_rate": item["hit_rate"],
                    "total_transferred_gb": item["total_transferred_gb"],
                    "wasted_prefetch_gb": item["wasted_prefetch_gb"],
                    "avg_stall_per_token_ms": item["avg_stall_per_token_ms"]
                })
        json.dump(serializable_results, f, indent=4)
    print(f"Saved results to: {json_path}")
    
    # ─── EXP 2: PowerInfer-2 CPU Latency Sensitivity Sweep ───
    print("\n[EXP 2] PowerInfer-2 CPU Latency Sensitivity Sweep (at 8.0 GB/s)...")
    cpu_delays = [2.0, 5.0, 10.0, 15.0, 20.0, 30.0]
    pi_stalls = []
    for delay in cpu_delays:
        res = run_sota_simulation(evaluation_db, transition_matrix, layer_entropies, top_cols_per_expert, layer_0_most_frequent, "powerinfer2", cache_size=32, link_bw_gb_s=8.0, cpu_compute_delay_us=delay)
        pi_stalls.append(res["avg_stall_per_token_ms"])
        print(f"  CPU Delay {delay:<6} us/neuron: Avg Stall = {res['avg_stall_per_token_ms']:.3f} ms/token")
        
    # Generate Sensitivity Plot
    plt.figure(figsize=(7, 4.5))
    plt.plot(cpu_delays, pi_stalls, 'o-', color='orange', linewidth=2.5, markersize=8, label='PowerInfer-2 Stall')
    plt.axhline(y=results["aaec"][1]["avg_stall_per_token_ms"], color='green', linestyle='--', label='AAEC (Ours)')
    plt.xlabel("CPU Execution Delay per Pinned Miss (us)")
    plt.ylabel("Average Serving Stall Latency (ms/token)")
    plt.title("PowerInfer-2 CPU Compute Delay Sensitivity vs. AAEC")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    pi_plot_path = os.path.join(OUTPUT_DIR, "powerinfer2_sensitivity.png")
    plt.savefig(pi_plot_path, dpi=150)
    plt.close()
    print(f"Saved sensitivity plot to: {pi_plot_path}")

if __name__ == "__main__":
    main()
