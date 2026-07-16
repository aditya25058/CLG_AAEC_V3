import os
import json
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def main():
    db_path = "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/qwen3_30b_real_v2.db"
    out_dir = "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)
    
    print("Loading activations data...")
    cursor = conn.execute(
        "SELECT layer, expert_id, prompt_id, token_pos, active_indices, "
        "energy_k_50, intermediate_dim FROM activations ORDER BY prompt_id, token_pos, layer, expert_id"
    )
    rows = cursor.fetchall()
    print(f"Loaded {len(rows)} records.")
    
    # ---------------------------------------------------------
    # 1. Expert Revisit Distance Distribution
    # ---------------------------------------------------------
    print("\n--- Running Expert Revisit Distance Study ---")
    # Group token_pos by (layer, expert_id, prompt_id)
    visits = defaultdict(list)
    for r in rows:
        key = (r[0], r[1], r[2])
        visits[key].append(r[3])
        
    revisit_distances = []
    for key, pos_list in visits.items():
        sorted_pos = sorted(pos_list)
        for i in range(len(sorted_pos) - 1):
            dist = sorted_pos[i+1] - sorted_pos[i]
            revisit_distances.append(dist)
            
    if revisit_distances:
        revisit_distances = np.array(revisit_distances)
        print("Expert Revisit Distance Stats (in tokens):")
        print(f"  Mean:    {np.mean(revisit_distances):.2f} tokens")
        print(f"  Std:     {np.std(revisit_distances):.2f} tokens")
        print(f"  Median:  {np.median(revisit_distances):.1f} tokens")
        print(f"  P90:     {np.percentile(revisit_distances, 90):.1f} tokens")
        print(f"  P95:     {np.percentile(revisit_distances, 95):.1f} tokens")
        print(f"  P99:     {np.percentile(revisit_distances, 99):.1f} tokens")
        print(f"  Maximum: {np.max(revisit_distances)} tokens")
        
        # Plot distribution
        plt.figure(figsize=(7, 4.5))
        plt.hist(revisit_distances, bins=np.logspace(0, np.log10(max(revisit_distances)), 30), color='#0077b6', edgecolor='black', alpha=0.8)
        plt.xscale('log')
        plt.xlabel('Revisit Distance (Tokens)', fontsize=11, fontweight='bold')
        plt.ylabel('Frequency (Log Scale)', fontsize=11, fontweight='bold')
        plt.yscale('log')
        plt.title('Expert Revisit Distance Distribution', fontsize=12, fontweight='bold', pad=12)
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "expert_revisit_distance.png"), dpi=200)
        plt.close()
    else:
        print("No revisits found.")

    # ---------------------------------------------------------
    # 2. Extended Working-Set Growth W(n) up to 512
    # ---------------------------------------------------------
    print("\n--- Running Extended Working-Set Growth Study ---")
    seq_activations = defaultdict(list)
    for r in rows:
        key = (r[0], r[1], r[2])
        indices = json.loads(r[4])[:r[5]] # 50% energy set
        seq_activations[key].append((r[3], set(indices)))
        
    n_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    growth_stats = defaultdict(list)
    
    for key, tokens_data in seq_activations.items():
        sorted_tokens = sorted(tokens_data, key=lambda x: x[0])
        cumulative_set = set()
        for idx, (t_pos, active_set) in enumerate(sorted_tokens):
            cumulative_set.update(active_set)
            step = idx + 1
            if step in n_values:
                growth_stats[step].append(len(cumulative_set) / 768.0 * 100.0)
                
    actual_n = [n for n in n_values if n in growth_stats and len(growth_stats[n]) > 0]
    print("Extended Working-Set Growth W(n):")
    for n in actual_n:
        arr = np.array(growth_stats[n])
        print(f"  n = {n:3d}: Mean = {np.mean(arr):6.2f}%, Median = {np.median(arr):6.2f}%, P90 = {np.percentile(arr, 90):6.2f}%, P95 = {np.percentile(arr, 95):6.2f}%")
        
    # Plot Extended Working-Set Growth
    plt.figure(figsize=(7, 4.5))
    y_means = [np.mean(growth_stats[n]) for n in actual_n]
    y_p90 = [np.percentile(growth_stats[n], 90) for n in actual_n]
    y_p10 = [np.percentile(growth_stats[n], 10) for n in actual_n]
    plt.plot(actual_n, y_means, marker='o', linewidth=2.5, color='#1d3557', label='Mean Footprint')
    plt.fill_between(actual_n, y_p10, y_p90, color='#1d3557', alpha=0.15, label='P10 - P90 range')
    plt.xscale('log')
    plt.xticks(actual_n, [str(n) for n in actual_n])
    plt.xlabel('Sequence Length (Tokens)', fontsize=11, fontweight='bold')
    plt.ylabel('Unique Neuron Footprint (%)', fontsize=11, fontweight='bold')
    plt.title('Asymptotic Working-Set Growth W(n) (50% Energy)', fontsize=12, fontweight='bold', pad=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "working_set_growth_512.png"), dpi=200)
    plt.close()

    # ---------------------------------------------------------
    # 3. Cache Size Sweep vs. Hit Rate & Bandwidth Saved
    # ---------------------------------------------------------
    print("\n--- Running Cache Size vs. Bandwidth Saved Sweep ---")
    cache_sizes = [32, 64, 96, 128, 160]
    
    # Pre-load sequence lists to speed up cache sweep simulation
    expert_sequences = defaultdict(list)
    for r in rows:
        key = (r[0], r[1])
        indices = json.loads(r[4])[:r[5]] # 50% energy set
        expert_sequences[key].append(indices)
        
    # Sub-sample expert-layers to ensure fast execution
    import random
    random.seed(42)
    sampled_keys = random.sample(list(expert_sequences.keys()), min(100, len(expert_sequences)))
    
    # We will simulate EMA and AAEC for each cache size
    ema_hit_rates = []
    aaec_hit_rates = []
    
    # Bandwidth measurement:
    # Full expert size: 768 neurons * 6144 parameters * 2 bytes = 9.437 MB per invocation
    # If a neuron is missed, we must load its column (6144 parameters * 2 bytes = 12.288 KB)
    # If we load the full expert on every miss, or just load the column-slice:
    # In sub-expert column-granular loading, the bytes transferred is:
    #   (number of misses) * 12.288 KB
    # Let's compare this to the baseline (loading the entire 9.437 MB expert on every token step where the expert is visited).
    # We report the Bandwidth Saved (%) = 1 - (Bytes Transferred / Baseline full expert bytes)
    ema_bw_saved = []
    aaec_bw_saved = []
    
    # Calculate baseline total bytes for the sampled runs:
    # baseline_bytes = sum of (len(seq) * 9.437 MB)
    total_invocations = sum(len(expert_sequences[k]) for k in sampled_keys)
    baseline_bytes = total_invocations * 9.437 * 1024 * 1024 # in bytes
    
    for size in cache_sizes:
        print(f"  Simulating cache size = {size}...")
        
        # AAEC config: 25% static head, 75% dynamic LRU
        static_size = size // 4
        dynamic_size = size - static_size
        
        ema_hits, ema_totals = 0, 0
        aaec_hits, aaec_totals = 0, 0
        
        ema_misses = 0
        aaec_misses = 0
        
        for key in sampled_keys:
            seq = expert_sequences[key]
            if len(seq) < 20:
                continue
                
            # Pre-compute Static Head for AAEC
            all_counts = defaultdict(int)
            for act in seq:
                for idx in act:
                    all_counts[idx] += 1
            sorted_neurons = sorted(all_counts.keys(), key=lambda x: all_counts[x], reverse=True)
            static_head = set(sorted_neurons[:static_size])
            
            # Cache states
            ema_weights = defaultdict(float)
            aaec_lru = []
            
            for act in seq:
                # 1. EMA cache hit check (before update)
                top_ema = set(sorted(ema_weights.keys(), key=lambda x: ema_weights[x], reverse=True)[:size])
                for neuron in act:
                    ema_totals += 1
                    if neuron in top_ema:
                        ema_hits += 1
                    else:
                        ema_misses += 1
                        
                # Update EMA weights
                for n_id in list(ema_weights.keys()):
                    ema_weights[n_id] *= 0.95
                for neuron in act:
                    ema_weights[neuron] += 0.05
                    
                # 2. AAEC cache hit check
                for neuron in act:
                    aaec_totals += 1
                    if neuron in static_head:
                        aaec_hits += 1
                    elif neuron in aaec_lru:
                        aaec_hits += 1
                        aaec_lru.remove(neuron)
                        aaec_lru.append(neuron)
                    else:
                        aaec_misses += 1
                        if len(aaec_lru) >= dynamic_size:
                            aaec_lru.pop(0)
                        aaec_lru.append(neuron)
                        
        hit_ema = ema_hits / ema_totals if ema_totals > 0 else 0
        hit_aaec = aaec_hits / aaec_totals if aaec_totals > 0 else 0
        
        # Bytes transferred
        ema_transferred = ema_misses * 12.288 * 1024 # bytes
        aaec_transferred = aaec_misses * 12.288 * 1024 # bytes
        
        bw_ema = 1.0 - (ema_transferred / baseline_bytes)
        bw_aaec = 1.0 - (aaec_transferred / baseline_bytes)
        
        ema_hit_rates.append(hit_ema * 100.0)
        aaec_hit_rates.append(hit_aaec * 100.0)
        
        ema_bw_saved.append(bw_ema * 100.0)
        aaec_bw_saved.append(bw_aaec * 100.0)
        
        print(f"    EMA  - Hit Rate: {hit_ema*100:6.2f}%, Bandwidth Saved: {bw_ema*100:6.2f}%")
        print(f"    AAEC - Hit Rate: {hit_aaec*100:6.2f}%, Bandwidth Saved: {bw_aaec*100:6.2f}%")

    # Generate Dual-Axis Plot (Cache Size vs Hit Rate and Bandwidth Saved)
    fig, ax1 = plt.subplots(figsize=(7.5, 5))
    
    color = '#1d3557'
    ax1.set_xlabel('Cache Size (Neurons)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Cache Hit Rate (%)', color=color, fontsize=11, fontweight='bold')
    line1 = ax1.plot(cache_sizes, ema_hit_rates, marker='o', ls='--', color='#457b9d', linewidth=2, label='EMA Hit Rate')
    line2 = ax1.plot(cache_sizes, aaec_hit_rates, marker='^', ls='-', color=color, linewidth=2.5, label='AAEC (Ours) Hit Rate')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 100)
    ax1.grid(True, ls='--', alpha=0.5)
    
    ax2 = ax1.twinx()  
    color_bw = '#e63946'
    ax2.set_ylabel('Bandwidth Savings (%)', color=color_bw, fontsize=11, fontweight='bold')
    line3 = ax2.plot(cache_sizes, ema_bw_saved, marker='s', ls='--', color='#f1a7a6', linewidth=2, label='EMA BW Savings')
    line4 = ax2.plot(cache_sizes, aaec_bw_saved, marker='D', ls='-', color=color_bw, linewidth=2.5, label='AAEC (Ours) BW Savings')
    ax2.tick_params(axis='y', labelcolor=color_bw)
    ax2.set_ylim(0, 100)
    
    # Combine legends
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right')
    
    plt.title('Cache Size vs. Hit Rate & Bandwidth Savings (Qwen3-30B)', fontsize=12, fontweight='bold', pad=12)
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, "cache_size_vs_metrics.png"), dpi=200)
    plt.close()
    
    print("\nAdvanced metrics run complete and plots saved successfully!")
    conn.close()

if __name__ == "__main__":
    main()
