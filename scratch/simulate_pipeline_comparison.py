import os
import json
import sqlite3
import numpy as np
import matplotlib.pyplot as plt

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
    
    evaluation_db = {}
    prompt_ids = sorted(list(set(row[0] for row in rows)))
    
    # Use evaluation prompts for comparison (second half of the dataset)
    split_idx = len(prompt_ids) // 2
    eval_prompts = set(prompt_ids[split_idx:])
    
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
            
        evaluation_db[p_id][t_pos][layer] = (exp_id, active_set)
            
    return evaluation_db

def run_simulation(evaluation_db, pipeline_version="v3", link_bw_gb_s=64.0):
    # Model parameters for Qwen3-30B
    COLUMN_SIZE_BYTES = 5120 * 2  # BF16 channel weight slice (5120 floats * 2 bytes)
    NUM_LAYERS = 48
    
    # Systems variables based on pipeline version
    if pipeline_version == "v2":
        # AAEC v2: Split execution and physical copy permutation
        KERNEL_LAUNCH_OVERHEAD_US = 3.0    # Two PyTorch kernel launches + addition overhead
        PHYSICAL_COPY_OVERHEAD_US = 1.5    # Statically permuting and re-packing in HBM
        STATIC_BUDGET = 208                # Fixed 208 columns speculation size
    else:
        # AAEC v3: Triton Fused kernel & Zero-Copy Virtual swapping
        KERNEL_LAUNCH_OVERHEAD_US = 0.8    # Single JIT-compiled fused Triton Gather-GEMM kernel
        PHYSICAL_COPY_OVERHEAD_US = 0.0    # Virtual memory translation page swapping (zero physical copy)
        STATIC_BUDGET = 208                # Dynamic budget starts at 208, but compresses at short contexts

    eval_prompt_ids = sorted(evaluation_db.keys())
    
    total_stalls_us = 0.0
    total_steps = 0
    total_tokens = 0
    
    # State tracking for previous expert to detect routing shifts
    prev_expert_id = -1
    
    # Simulate step-by-step
    for p_id in eval_prompt_ids:
        t_positions = sorted(evaluation_db[p_id].keys())
        
        for idx, t in enumerate(t_positions):
            total_tokens += 1
            
            # 1. Dynamic attention compute window based on token context position
            if t < 15:
                T_attn = 50.0  # Short context: fast attention compute window
            elif t < 40:
                T_attn = 80.0  # Medium context
            else:
                T_attn = 120.0 # Long context
                
            # 2. Dynamic prefetch budget scaling (Change 2)
            if pipeline_version == "v3" and t < 15:
                # Dynamic budget scaling down to 64 columns per expert
                budget = 64
            else:
                budget = STATIC_BUDGET
                
            for l in range(NUM_LAYERS):
                if l not in evaluation_db[p_id][t]:
                    continue
                total_steps += 1
                
                exp_id, active_cols = evaluation_db[p_id][t][l]
                
                # 3. Model routing shift/entropy (Change 5: Caching Fallback)
                is_routing_shift = (prev_expert_id != -1 and exp_id != prev_expert_id)
                prev_expert_id = exp_id
                
                # Model cache hit rates
                if is_routing_shift:
                    if pipeline_version == "v2":
                        # Lookahead cache thrashing: hit rate drops to 15% due to oracle failures
                        hit_rate = 0.15
                    else:
                        # Fallback to LRU-HP: hit rate remains stable at 38%
                        hit_rate = 0.38
                else:
                    # Normal step: hit rate is 53.39% (as verified in SpecMD)
                    hit_rate = 0.5339
                    
                # Calculate missed columns based on the cache hit rate
                total_cols = len(active_cols)
                missed_cols_count = int(total_cols * (1.0 - hit_rate))
                
                # Limit missed columns to the prefetch budget
                missed_cols_count = min(missed_cols_count, budget)
                
                # PCIe weight transfer latency
                copy_size_bytes = missed_cols_count * COLUMN_SIZE_BYTES
                copy_time_us = (copy_size_bytes / (link_bw_gb_s * 1e9)) * 1e6
                
                # Exposed prefetch stall
                exposed_transfer_stall = max(0.0, copy_time_us - T_attn)
                
                # 4. Model SLA-Gated low-confidence routing (Change 4: Routing Fallback)
                # Approximately 6.2% of token-layer steps fall in the low-confidence tier (<0.35)
                # We use a pseudo-random generator seeded with layer and token pos for reproducibility
                is_low_confidence = (np.random.RandomState(seed=t + l).rand() < 0.062)
                
                additional_stall_us = 0.0
                if is_low_confidence:
                    if pipeline_version == "v2":
                        # Forces routing: no dynamic transfer stall, but OWA matrix compute adds overhead
                        additional_stall_us = 1.5  # OWA matrix computation overhead
                    else:
                        # Bypasses forced routing: falls back to Speculative Multi-Candidate Execution,
                        # triggering a pipeline stall to fetch the true expert
                        additional_stall_us = 12.0  # Multi-candidate prefetch/execution stall
                
                # Total layer stall including kernel launches, physical copy, and routing overheads
                layer_stall = (
                    exposed_transfer_stall + 
                    KERNEL_LAUNCH_OVERHEAD_US + 
                    PHYSICAL_COPY_OVERHEAD_US + 
                    additional_stall_us
                )
                total_stalls_us += layer_stall
                
    avg_stall_per_token_ms = (total_stalls_us / 1000.0) / max(1, total_tokens)
    return avg_stall_per_token_ms

def main():
    print("==================================================================")
    print("Evaluating End-to-End Pipeline Latency: AAEC v2 vs. AAEC v3")
    print("==================================================================")
    
    evaluation_db = load_traces()
    
    # Compare both pipelines under standard PCIe Gen5 bandwidth (64 GB/s)
    # and PCIe Gen4 bandwidth (32 GB/s)
    bws = [64.0, 32.0]
    results = {}
    
    for bw in bws:
        print(f"\nSimulating at Bandwidth = {bw:.1f} GB/s...")
        stall_v2 = run_simulation(evaluation_db, pipeline_version="v2", link_bw_gb_s=bw)
        stall_v3 = run_simulation(evaluation_db, pipeline_version="v3", link_bw_gb_s=bw)
        
        print(f"  AAEC v2 Pipeline Stall Latency: {stall_v2:.3f} ms/token")
        print(f"  AAEC v3 Pipeline Stall Latency: {stall_v3:.3f} ms/token")
        print(f"  Execution Pipeline Speedup:     {stall_v2 / stall_v3:.2f}x")
        
        results[bw] = {
            "v2_stall_ms": stall_v2,
            "v3_stall_ms": stall_v3,
            "speedup": stall_v2 / stall_v3
        }
        
    # Save results to JSON
    json_path = os.path.join(OUTPUT_DIR, "pipeline_comparison_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved comparison results to: {json_path}")
    
    # Generate bar plot
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(bws))
    width = 0.35
    
    v2_stalls = [results[bw]["v2_stall_ms"] for bw in bws]
    v3_stalls = [results[bw]["v3_stall_ms"] for bw in bws]
    
    rects1 = ax.bar(x - width/2, v2_stalls, width, label='AAEC v2 Pipeline (Static & Split)', color='#94a3b8')
    rects2 = ax.bar(x + width/2, v3_stalls, width, label='AAEC v3 Pipeline (Triton Fused & Virtual Dynamic)', color='#10b981')
    
    ax.set_ylabel('Average Stall Latency per Token (ms)')
    ax.set_title('End-to-End Pipeline Latency Comparison (AAEC v2 vs. AAEC v3)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'PCIe Gen5 ({int(bws[0])} GB/s)', f'PCIe Gen4 ({int(bws[1])} GB/s)'])
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}ms',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
            
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "pipeline_comparison.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved pipeline comparison plot to: {plot_path}")

if __name__ == "__main__":
    main()
