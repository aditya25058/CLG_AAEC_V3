import sqlite3
import json
import os
import numpy as np
from collections import OrderedDict, defaultdict

DB_PATH = "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/qwen3_30b_real_v2.db"
OUTPUT_DIR = "/home/palakm/MoEServingSim/qwen3_30b_plots"

def main():
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at {DB_PATH}")
        return

    print("Connecting to trace database...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Load all trace activations
    print("Loading traces from database...")
    cursor.execute("""
        SELECT prompt_id, token_pos, layer, expert_id, active_indices, energy_k_50, energy_k_90
        FROM activations
        ORDER BY prompt_id, token_pos, layer
    """)
    rows = cursor.fetchall()
    conn.close()

    print(f"Loaded {len(rows)} expert activation records.")

    # 1. Precompute column popularity per (layer, expert_id) to initialize warm caches
    expert_col_frequencies = defaultdict(lambda: defaultdict(int))
    for row in rows:
        p_id, t_pos, layer, exp_id, indices_str, k50, k90 = row
        indices = json.loads(indices_str)
        # Use top 90% energy columns as popularity base
        for col in indices[:k90]:
            expert_col_frequencies[(layer, exp_id)][col] += 1

    # Sort columns by frequency per expert
    warm_init_columns = {}
    for (layer, exp_id), col_freqs in expert_col_frequencies.items():
        sorted_cols = sorted(col_freqs.keys(), key=lambda x: col_freqs[x], reverse=True)
        warm_init_columns[(layer, exp_id)] = sorted_cols

    # 2. Organize rows into trace_db[prompt_id][token_pos][layer] = list of (expert_id, active_cols_50, active_cols_90)
    trace_db = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    prompt_ids = set()

    for row in rows:
        p_id, t_pos, layer, exp_id, indices_str, k50, k90 = row
        prompt_ids.add(p_id)
        indices = json.loads(indices_str)
        active_50 = set(indices[:k50])
        active_90 = set(indices[:k90])
        trace_db[p_id][t_pos][layer].append((exp_id, active_50, active_90))

    sorted_prompt_ids = sorted(list(prompt_ids))
    print(f"Total prompts in trace: {len(sorted_prompt_ids)}")

    # Split prompts: First half for warm-up/profiling, second half for evaluation (just like original simulator)
    split_idx = len(sorted_prompt_ids) // 2
    eval_prompts = sorted_prompt_ids[split_idx:]
    print(f"Evaluating on {len(eval_prompts)} prompts...")

    # Hardware specs
    PCIE_BW = 64.0  # GB/s (PCIe Gen5 x16 duplex)
    DMA_LAUNCH_OVERHEAD_US = 2.5  # us per DMA copy command
    COLUMN_SIZE_BYTES = 5120 * 2 * 3  # BF16 for gate_proj, up_proj, down_proj = 30,720 bytes (30.72 KB)

    # We evaluate for both 50% energy and 90% energy thresholds
    energy_levels = ["50%", "90%"]
    cache_sizes = [32, 64, 128, 256]

    print("\n" + "="*80)
    print("EXPERIMENT 1 & 2: MULTI-EXPERT COLUMN OVERLAP & BUDGET FEASIBILITY")
    print("="*80)

    for energy_lvl in energy_levels:
        print(f"\n>>> Energy Target: {energy_lvl} Output Reconstruction Quality")
        
        for c_size in cache_sizes:
            # Initialize GPU cache per (layer, expert_id)
            gpu_cache = {}
            for l in range(48):
                for e in range(128):
                    init_cols = warm_init_columns.get((l, e), [])
                    gpu_cache[(l, e)] = list(init_cols[:c_size])

            total_layers_executed = 0
            sum_required_cols = 0
            sum_missed_cols = 0
            sum_missed_bytes = 0
            sum_hits = 0
            
            strided_latencies = []
            packed_latencies = []
            
            # Run simulation on evaluation prompts
            for p_id in eval_prompts:
                t_positions = sorted(trace_db[p_id].keys())
                for t in t_positions:
                    for l in range(48):
                        if l not in trace_db[p_id][t]:
                            continue
                        
                        experts_data = trace_db[p_id][t][l]
                        if not experts_data:
                            continue

                        total_layers_executed += 1
                        
                        # Set of missed columns for each active expert at this step
                        layer_required_cols = 0
                        layer_missed_cols = 0
                        active_experts_with_misses = 0

                        for exp_id, cols_50, cols_90 in experts_data:
                            active_cols = cols_50 if energy_lvl == "50%" else cols_90
                            cache_list = gpu_cache[(l, exp_id)]
                            cache_set = set(cache_list)
                            
                            required = len(active_cols)
                            missed_set = active_cols - cache_set
                            missed = len(missed_set)
                            hits = required - missed
                            
                            layer_required_cols += required
                            layer_missed_cols += missed
                            sum_hits += hits
                            
                            if missed > 0:
                                active_experts_with_misses += 1

                            # Update cache (LRU)
                            for col in active_cols:
                                if col in cache_list:
                                    cache_list.remove(col)
                                else:
                                    if len(cache_list) >= c_size:
                                        cache_list.pop(0)
                                cache_list.append(col)

                        sum_required_cols += layer_required_cols
                        sum_missed_cols += layer_missed_cols
                        
                        # Missed bytes
                        layer_bytes = layer_missed_cols * COLUMN_SIZE_BYTES
                        sum_missed_bytes += layer_bytes

                        # Compute PCIe latency for this step/layer:
                        # 1. Strided PCIe latency:
                        #    For each missed column, copy W_gate, W_up, W_down separately (3 separate transfers)
                        #    Number of DMA transfers = layer_missed_cols * 3
                        #    Launch overhead = num_transfers * DMA_LAUNCH_OVERHEAD_US
                        #    Transmission time = layer_bytes / PCIE_BW
                        num_strided_transfers = layer_missed_cols * 3
                        transmission_time_us = (layer_bytes / (PCIE_BW * 1e9)) * 1e6
                        strided_lat_us = (num_strided_transfers * DMA_LAUNCH_OVERHEAD_US) + transmission_time_us
                        strided_latencies.append(strided_lat_us)

                        # 2. Packed (ADETR-coalesced) PCIe latency:
                        #    For each active expert with misses, copy its missed columns in 1 contiguous transfer.
                        #    Number of DMA transfers = active_experts_with_misses
                        #    Launch overhead = num_transfers * DMA_LAUNCH_OVERHEAD_US
                        packed_lat_us = (active_experts_with_misses * DMA_LAUNCH_OVERHEAD_US) + transmission_time_us
                        packed_latencies.append(packed_lat_us)

            avg_required_cols = sum_required_cols / total_layers_executed
            avg_missed_cols = sum_missed_cols / total_layers_executed
            avg_missed_bytes = sum_missed_bytes / total_layers_executed
            hit_rate = sum_hits / max(1, sum_hits + sum_missed_cols)
            
            avg_strided_lat_us = np.mean(strided_latencies)
            avg_packed_lat_us = np.mean(packed_latencies)
            speedup = avg_strided_lat_us / max(1e-3, avg_packed_lat_us)
            
            # Hiding window success rate (<= 100 us)
            strided_success_rate = np.mean([1.0 if lat <= 100.0 else 0.0 for lat in strided_latencies]) * 100.0
            packed_success_rate = np.mean([1.0 if lat <= 100.0 else 0.0 for lat in packed_latencies]) * 100.0

            print(f"  Cache Size = {c_size} columns per expert:")
            print(f"    - Hit Rate:                 {hit_rate*100:.2f}%")
            print(f"    - Avg Required Cols/Layer:  {avg_required_cols:.1f} columns")
            print(f"    - Avg Missed Cols/Layer:    {avg_missed_cols:.1f} columns")
            print(f"    - Avg Payload Size/Layer:   {avg_missed_bytes / 1024:.1f} KB")
            print(f"    - Avg Strided PCIe Latency: {avg_strided_lat_us:.2f} us")
            print(f"    - Avg Packed PCIe Latency:  {avg_packed_lat_us:.2f} us (Speedup: {speedup:.2f}x)")
            print(f"    - Strided Hiding Success Rate (<= 100us): {strided_success_rate:.2f}%")
            print(f"    - Packed Hiding Success Rate (<= 100us):  {packed_success_rate:.2f}%")

if __name__ == "__main__":
    main()
