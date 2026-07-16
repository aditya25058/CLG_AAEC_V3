import json
import sqlite3
import numpy as np
from collections import defaultdict

def main():
    db_path = "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/qwen3_30b_real_v2.db"
    print(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)
    
    # Query active_indices and energy_k_50 per layer and expert
    cursor = conn.execute("SELECT layer, expert_id, active_indices, energy_k_50 FROM activations")
    
    # Build expert-level neuron frequency counts
    # (layer, expert_id) -> neuron_id -> count
    expert_counts = defaultdict(lambda: defaultdict(int))
    expert_totals = defaultdict(int)
    
    for layer, exp_id, idx_json, k50 in cursor:
        indices = json.loads(idx_json)[:k50]
        key = (layer, exp_id)
        for idx in indices:
            expert_counts[key][idx] += 1
        expert_totals[key] += 1
        
    print(f"Loaded activation data for {len(expert_counts)} active experts.")
    
    # Fit Zipf for each expert
    alphas = []
    
    for key, counts in expert_counts.items():
        total = expert_totals[key]
        if total < 10:
            continue
            
        # Sort counts descending
        sorted_counts = sorted(counts.values(), reverse=True)
        ranks = np.arange(1, len(sorted_counts) + 1)
        freqs = np.array(sorted_counts) / total
        
        # Filter to avoid fit distortion from near-zero tails
        # We only fit on neurons that appear in at least 1% of the invocations
        mask = freqs > 0.01
        if np.sum(mask) < 3:
            # Fallback to top-30 hot neurons if too few exceed 1%
            mask = np.arange(len(freqs)) < min(len(freqs), 30)
            
        ranks_fit = ranks[mask]
        freqs_fit = freqs[mask]
        
        log_r = np.log(ranks_fit)
        log_f = np.log(freqs_fit)
        
        try:
            alpha, log_C = np.polyfit(log_r, log_f, 1)
            alphas.append(-alpha)
        except Exception as e:
            pass
            
    if alphas:
        alphas = np.array(alphas)
        print("\n--- Corrected Zipf Exponent (per expert) ---")
        print(f"  Zipf exponent alpha:")
        print(f"    Mean:   {np.mean(alphas):.4f}")
        print(f"    Std:    {np.std(alphas):.4f}")
        print(f"    Median: {np.median(alphas):.4f}")
        print(f"    P10:    {np.percentile(alphas, 10):.4f}")
        print(f"    P90:    {np.percentile(alphas, 90):.4f}")
    else:
        print("No fits succeeded.")

if __name__ == "__main__":
    main()
