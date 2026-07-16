#!/usr/bin/env python3
import os
import json
import sqlite3
import numpy as np
import matplotlib.pyplot as plt

def main():
    db_path = "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/qwen3_30b_real_v2.db"
    transitions_path = "/home/palakm/MoEServingSim/qwen3_30b_plots/experiment_8_expert_transitions.json"
    out_dir = "/home/palakm/MoEServingSim/qwen3_30b_plots"
    os.makedirs(out_dir, exist_ok=True)

    print("=== LOADING REAL WORKLOAD DATA ===")
    if not os.path.exists(transitions_path) or not os.path.exists(db_path):
        print("Error: Missing database or transition matrix JSON.")
        return

    # 1. Load transition probability matrix
    with open(transitions_path, "r") as f:
        transition_data = json.load(f)
    transition_prob = np.array(transition_data["transition_probability_matrix"]) # [128, 128]

    # 2. Connect to database and load all sequential traces into memory
    print("Loading sequential execution traces from DB...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # We load prompt_id, layer, token_pos, expert_id, and active columns
    cursor.execute("""
        SELECT prompt_id, layer, token_pos, expert_id, active_indices, energy_k_50 
        FROM activations 
        ORDER BY prompt_id, token_pos, layer
    """)
    rows = cursor.fetchall()
    
    # Organize trace: trace_db[prompt_id][token_pos][layer] = (expert_id, active_cols_set)
    trace_db = {}
    expert_overall_freqs = {} # expert_id -> column -> count (for static initialization)

    for row in rows:
        p_id, layer, t_pos, exp_id, indices_str, k50 = row
        indices = json.loads(indices_str)[:k50] # 50% energy set
        active_set = set(indices)
        
        if p_id not in trace_db:
            trace_db[p_id] = {}
        if t_pos not in trace_db[p_id]:
            trace_db[p_id][t_pos] = {}
            
        trace_db[p_id][t_pos][layer] = (exp_id, active_set)
        
        # Accumulate column frequencies for initialization
        if exp_id not in expert_overall_freqs:
            expert_overall_freqs[exp_id] = {}
        for col in active_set:
            expert_overall_freqs[exp_id][col] = expert_overall_freqs[exp_id].get(col, 0) + 1
            
    conn.close()
    
    # Find active prompt IDs and token lengths
    prompt_ids = sorted(trace_db.keys())
    print(f"Loaded sequential traces for {len(prompt_ids)} prompts.")

    # Precompute overall top columns per expert for cache initialization
    top_cols_per_expert = {}
    for e in range(128):
        if e in expert_overall_freqs:
            sorted_cols = sorted(expert_overall_freqs[e].keys(), key=lambda x: expert_overall_freqs[e][x], reverse=True)
            top_cols_per_expert[e] = sorted_cols
        else:
            top_cols_per_expert[e] = list(range(128))

    # System parameters
    COMPUTE_TIME_PER_LAYER_US = 40.0  # Phase 1 compute time
    LATENCY_OVERHEAD_PER_DMA_US = 0.5  # PCIe DMA launch overhead
    COLUMN_SIZE_BYTES = 5120 * 2  # BF16 weight channel

    # ----------------------------------------------------
    # TRACE-PLAYBACK SIMULATION RUNNER
    # ----------------------------------------------------
    def run_simulation(policy="demand", threshold=0.10, cache_size=128, link_bw_gb_s=64.0):
        total_misses = 0
        total_hits = 0
        total_prefetched_bytes = 0
        total_pushed_bytes = 0
        total_stalls_us = 0.0
        total_steps = 0

        # Initialize GPU warm caches per layer (capacity = cache_size columns)
        gpu_cache = {}
        for l in range(48):
            for e in range(128):
                gpu_cache[(l, e)] = list(top_cols_per_expert[e][:cache_size])

        current_prefetch_queue = {}

        for p_id in prompt_ids:
            t_positions = sorted(trace_db[p_id].keys())
            current_prefetch_queue.clear()
            
            for idx, t in enumerate(t_positions):
                total_steps += 1
                
                # A. Execute current step
                for l in range(48):
                    # In some steps, some layers might not have recorded activations if no routing happened
                    if l not in trace_db[p_id][t]:
                        continue
                        
                    exp_id, active_cols = trace_db[p_id][t][l]

                    cache_list = gpu_cache[(l, exp_id)]
                    cache_set = set(cache_list)

                    # Determine misses
                    missed = active_cols - cache_set
                    # Check if missed columns were speculatively prefetched in previous step
                    if policy == "speculative" and (l, exp_id) in current_prefetch_queue:
                        pref_hits = missed.intersection(current_prefetch_queue[(l, exp_id)])
                        missed = missed - pref_hits

                    hits = len(active_cols) - len(missed)
                    total_hits += hits
                    total_misses += len(missed)

                    # Update cache with LRU policy
                    all_newly_accessed = list(active_cols)
                    for nid in all_newly_accessed:
                        if nid in cache_list:
                            cache_list.remove(nid)
                        else:
                            if len(cache_list) >= cache_size:
                                cache_list.pop(0)
                        cache_list.append(nid)

                    # Process stalls for remaining misses
                    if missed:
                        copy_size_bytes = len(missed) * COLUMN_SIZE_BYTES
                        copy_time_us = (copy_size_bytes / (link_bw_gb_s * 1e9)) * 1e6
                        total_pushed_bytes += copy_size_bytes
                        stall_us = max(0.0, (copy_time_us + LATENCY_OVERHEAD_PER_DMA_US) - COMPUTE_TIME_PER_LAYER_US)
                        total_stalls_us += stall_us

                # B. Prepare prefetch queue for the NEXT step (t+1)
                current_prefetch_queue = {}
                if policy == "speculative" and idx < len(t_positions) - 1:
                    next_t = t_positions[idx + 1]
                    for l in range(48):
                        if l not in trace_db[p_id][t]:
                            continue
                        current_exp, _ = trace_db[p_id][t][l]
                        probs = transition_prob[current_exp]
                        
                        # Find predicted experts
                        predicted_experts = np.where(probs >= threshold)[0]
                        for pred_exp in predicted_experts:
                            cache_set = set(gpu_cache[(l, pred_exp)])
                            # Prefetch the top active columns for this predicted expert
                            top_cols = set(top_cols_per_expert[pred_exp][:cache_size])
                            missing = top_cols - cache_set
                            if missing:
                                if (l, pred_exp) not in current_prefetch_queue:
                                    current_prefetch_queue[(l, pred_exp)] = set()
                                current_prefetch_queue[(l, pred_exp)].update(missing)
                                total_prefetched_bytes += len(missing) * COLUMN_SIZE_BYTES

        hit_rate = total_hits / max(1, total_hits + total_misses)
        total_transferred_gb = (total_prefetched_bytes + total_pushed_bytes) / 1e9
        avg_stall_per_token_ms = (total_stalls_us / 1000.0) / max(1, total_steps)
        wasted_prefetch_gb = max(0.0, (total_prefetched_bytes - (total_hits * COLUMN_SIZE_BYTES)) / 1e9)

        return {
            "hit_rate": hit_rate,
            "total_transferred_gb": total_transferred_gb,
            "wasted_prefetch_gb": wasted_prefetch_gb,
            "avg_stall_per_token_ms": avg_stall_per_token_ms
        }

    # ----------------------------------------------------
    # SWEEP INTERCONNECT SPEEDS
    # ----------------------------------------------------
    # Link speeds (GB/s): 2.0 (16Gbs Network), 8.0 (PCIe Gen3), 16.0 (PCIe Gen4), 64.0 (PCIe Gen5)
    link_speeds = [2.0, 8.0, 16.0, 64.0]
    results_demand = []
    results_spec_low = []
    results_spec_high = []

    print("\nStarting sequential trace simulation sweep across interconnect speeds...")
    for bw in link_speeds:
        print(f"  Simulating at Link Bandwidth = {bw:.1f} GB/s...")
        res_d = run_simulation(policy="demand", link_bw_gb_s=bw)
        res_sl = run_simulation(policy="speculative", threshold=0.10, link_bw_gb_s=bw)
        res_sh = run_simulation(policy="speculative", threshold=0.02, link_bw_gb_s=bw)
        
        results_demand.append(res_d)
        results_spec_low.append(res_sl)
        results_spec_high.append(res_sh)

    # ----------------------------------------------------
    # OUTPUT RESULTS
    # ----------------------------------------------------
    print("\n" + "="*145)
    print(f"{'Bandwidth (GB/s)':<18} | {'Policy':<30} | {'Effective Hit Rate':<20} | {'PCIe Stall/Token (ms)':<22} | {'Total Data (GB)':<16} | {'Wasted Prefetch (GB)':<20}")
    print("-"*145)
    for idx, bw in enumerate(link_speeds):
        d, sl, sh = results_demand[idx], results_spec_low[idx], results_spec_high[idx]
        print(f"{bw:<18.1f} | {'Demand-Driven (Baseline)':<30} | {d['hit_rate']*100:<18.2f}% | {d['avg_stall_per_token_ms']:<20.4f} | {d['total_transferred_gb']:<16.4f} | {d['wasted_prefetch_gb']:<20.4f}")
        print(f"{'':<18} | {'Speculative (Threshold=0.10)':<30} | {sl['hit_rate']*100:<18.2f}% | {sl['avg_stall_per_token_ms']:<20.4f} | {sl['total_transferred_gb']:<16.4f} | {sl['wasted_prefetch_gb']:<20.4f}")
        print(f"{'':<18} | {'Speculative (Threshold=0.02)':<30} | {sh['hit_rate']*100:<18.2f}% | {sh['avg_stall_per_token_ms']:<20.4f} | {sh['total_transferred_gb']:<16.4f} | {sh['wasted_prefetch_gb']:<20.4f}")
        print("-"*145)
    print("="*145)

    # Save to JSON
    report_data = {
        "link_speeds": link_speeds,
        "demand": results_demand,
        "speculative_low": results_spec_low,
        "speculative_high": results_spec_high
    }
    with open(os.path.join(out_dir, "predictive_prefetching_sweep_results.json"), "w") as f:
        json.dump(report_data, f, indent=4)
    print(f"Results JSON saved to: {out_dir}/predictive_prefetching_sweep_results.json")

    # Generate plot
    print("Generating performance plot...")
    plt.figure(figsize=(7.5, 4.5))
    
    stalls_d = [r['avg_stall_per_token_ms'] for r in results_demand]
    stalls_sl = [r['avg_stall_per_token_ms'] for r in results_spec_low]
    stalls_sh = [r['avg_stall_per_token_ms'] for r in results_spec_high]

    plt.plot(link_speeds, stalls_d, marker='o', color='#e63946', linewidth=2.5, label='Demand-Driven (Baseline)')
    plt.plot(link_speeds, stalls_sl, marker='s', color='#f1a7a1', linewidth=2.0, linestyle='--', label='Speculative (Threshold=0.10)')
    plt.plot(link_speeds, stalls_sh, marker='^', color='#1d3557', linewidth=2.5, label='Speculative (Threshold=0.02)')

    plt.xscale('log')
    plt.xticks(link_speeds, [f"{bw:.1f} GB/s" for bw in link_speeds])
    plt.xlabel('Interconnect Bandwidth (GB/s)', fontweight='bold')
    plt.ylabel('Average PCIe Stall per Token (ms)', fontweight='bold')
    plt.title('PCIe Stall Latency vs. Interconnect Bandwidth (Real Traces)', fontsize=12, fontweight='bold', pad=15)
    plt.grid(True, which="both", ls='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "predictive_prefetch_comparison.png"), dpi=200)
    plt.close()
    print(f"Comparative plot saved to: {out_dir}/predictive_prefetch_comparison.png")

if __name__ == "__main__":
    main()
