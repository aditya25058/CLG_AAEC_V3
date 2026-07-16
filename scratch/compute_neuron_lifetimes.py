import json
import sqlite3
import numpy as np
from collections import defaultdict

def main():
    db_path = "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/qwen3_30b_real_v2.db"
    print(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)
    
    # --- Part 1: Layer-wise Active Neuron Counts ---
    print("\n--- 1. Layer-wise Active Neuron Counts (50% Energy) ---")
    cursor = conn.execute(
        "SELECT layer, energy_k_50 FROM activations ORDER BY layer"
    )
    layer_vals = defaultdict(list)
    for layer, k50 in cursor:
        layer_vals[layer].append(k50)
        
    print(f"Layer | Mean Neurons (50% Energy) | Std (Neurons)")
    print(f"------|--------------------------|--------------")
    for layer in sorted(layer_vals.keys()):
        if layer % 4 == 0 or layer == 47:
            arr = np.array(layer_vals[layer])
            print(f"{layer:5d} | {np.mean(arr):24.1f} | {np.std(arr):12.2f}")

    # --- Part 2: Working-Set Lifetime (Consecutive tokens active) ---
    print("\n--- 2. Neuron Active Lifetimes (Consecutive Tokens) ---")
    
    # Query layer, expert_id, prompt_id, token_pos, active_indices, energy_k_50
    # To trace consecutive runs, we order by layer, expert, prompt, and token_pos
    cursor = conn.execute(
        "SELECT layer, expert_id, prompt_id, token_pos, active_indices, energy_k_50 "
        "FROM activations ORDER BY layer, expert_id, prompt_id, token_pos"
    )
    
    # We trace runs for each unique (layer, expert_id, prompt_id)
    # inside this run, we trace which neurons are active at each token_pos.
    # To save memory, we can process them sequentially.
    current_key = None
    active_history = defaultdict(list)  # neuron_id -> list of token_pos where it is active
    
    run_lengths = []
    
    def process_history(history):
        # Compute consecutive run lengths for each neuron
        for neuron_id, token_positions in history.items():
            if not token_positions:
                continue
            # Sort positions
            sorted_pos = sorted(token_positions)
            
            # Find consecutive runs
            current_run = 1
            for idx in range(1, len(sorted_pos)):
                if sorted_pos[idx] == sorted_pos[idx - 1] + 1:
                    current_run += 1
                else:
                    run_lengths.append(current_run)
                    current_run = 1
            run_lengths.append(current_run)

    for layer, exp_id, prompt_id, token_pos, idx_json, k50 in cursor:
        key = (layer, exp_id, prompt_id)
        if key != current_key:
            if current_key is not None:
                process_history(active_history)
            current_key = key
            active_history = defaultdict(list)
            
        indices = json.loads(idx_json)[:k50]
        for idx in indices:
            active_history[idx].append(token_pos)
            
    # Process final key
    if current_key is not None:
        process_history(active_history)
        
    if run_lengths:
        run_lengths = np.array(run_lengths)
        print(f"Total neuron activation runs counted: {len(run_lengths)}")
        print(f"Neuron Active Lifetime (in consecutive tokens):")
        print(f"  Mean:    {np.mean(run_lengths):.2f} tokens")
        print(f"  Std:     {np.std(run_lengths):.2f} tokens")
        print(f"  Median:  {np.median(run_lengths):.1f} tokens")
        print(f"  P10:     {np.percentile(run_lengths, 10):.1f} tokens")
        print(f"  P90:     {np.percentile(run_lengths, 90):.1f} tokens")
        print(f"  P95:     {np.percentile(run_lengths, 95):.1f} tokens")
        print(f"  Maximum: {np.max(run_lengths)} tokens")
    else:
        print("No activation runs found.")

if __name__ == "__main__":
    main()
