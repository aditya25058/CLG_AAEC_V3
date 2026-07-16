#!/usr/bin/env python3
import os
import json
import sqlite3
import numpy as np

def main():
    db_path = "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/qwen3_30b_real_v2.db"
    if not os.path.exists(db_path):
        print(f"Error: DB not found at {db_path}")
        return

    print("Connecting to DB...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("Loading sequential activations...")
    # Load prompt_id, token_pos, layer, expert_id, active_indices, energy_k_50
    cursor.execute("""
        SELECT prompt_id, token_pos, layer, expert_id, active_indices, energy_k_50
        FROM activations
        ORDER BY prompt_id, token_pos, layer
    """)
    rows = cursor.fetchall()
    conn.close()

    print(f"Loaded {len(rows)} records. Processing...")
    
    # Organize data: data[prompt_id][token_pos][layer] = (expert_id, set_of_active_indices)
    data = {}
    for r in rows:
        p_id, t_pos, layer, exp_id, indices_str, k50 = r
        indices = json.loads(indices_str)[:k50] # 50% energy target
        active_set = set(indices)
        
        if p_id not in data:
            data[p_id] = {}
        if t_pos not in data[p_id]:
            data[p_id][t_pos] = {}
        data[p_id][t_pos][layer] = (exp_id, active_set)

    # Let's compute correlations
    # 1. Temporal overlap: Jaccard similarity of active set at layer L between token T and T-1
    temporal_overlaps = []
    # 2. Cross-layer expert correlation: matrix of size [128, 128] for P(E_{L+2} | E_L)
    cross_layer_exp_transitions = np.zeros((128, 128))
    # 3. Cross-layer column Jaccard overlap: Jaccard similarity of active columns between Layer L and Layer L+2 (for same token)
    cross_layer_col_overlaps = []
    
    # 4. Joint Predictor accuracy sweeps
    # Let's count how well we can predict C_{L+2, T} using:
    #   - Predictor A (Temporal): C_{L+2, T-1}
    #   - Predictor B (Cross-layer): Top-K overall columns of the active expert at Layer L+2 (assuming we predict the expert index)
    #   - Predictor C (Combined NAWP): Combine C_{L+2, T-1} and cross-layer predictors
    
    correct_temp = 0
    total_temp = 0
    
    correct_cross = 0
    total_cross = 0
    
    for p_id, tokens in data.items():
        t_positions = sorted(tokens.keys())
        for idx, t in enumerate(t_positions):
            # Check temporal correlation (t vs t-1)
            if idx > 0:
                prev_t = t_positions[idx - 1]
                for l in range(48):
                    if l in tokens[t] and l in tokens[prev_t]:
                        exp_curr, cols_curr = tokens[t][l]
                        exp_prev, cols_prev = tokens[prev_t][l]
                        if exp_curr == exp_prev:
                            intersection = len(cols_curr.intersection(cols_prev))
                            union = len(cols_curr.union(cols_prev))
                            if union > 0:
                                temporal_overlaps.append(intersection / union)

            # Check cross-layer correlation (L vs L+2)
            for l in range(46): # up to layer 45
                l_plus_2 = l + 2
                if l in tokens[t] and l_plus_2 in tokens[t]:
                    exp_l, cols_l = tokens[t][l]
                    exp_l2, cols_l2 = tokens[t][l_plus_2]
                    
                    cross_layer_exp_transitions[exp_l, exp_l2] += 1
                    
                    intersection = len(cols_l.intersection(cols_l2))
                    union = len(cols_l.union(cols_l2))
                    if union > 0:
                        cross_layer_col_overlaps.append(intersection / union)

    # Let's report results
    print("\n=== CORRELATION ANALYSIS RESULTS ===")
    if temporal_overlaps:
        print(f"Average Temporal Jaccard Overlap (same layer, adjacent tokens, same expert): {np.mean(temporal_overlaps)*100:.2f}%")
    if cross_layer_col_overlaps:
        print(f"Average Cross-layer Jaccard Overlap (Layer L vs L+2, same token): {np.mean(cross_layer_col_overlaps)*100:.2f}%")
        
    # Expert transitions: row-normalize to get transition probability P(E_{L+2} | E_L)
    row_sums = cross_layer_exp_transitions.sum(axis=1)
    # Avoid division by zero
    transitions_prob = np.zeros_like(cross_layer_exp_transitions)
    for i in range(128):
        if row_sums[i] > 0:
            transitions_prob[i] = cross_layer_exp_transitions[i] / row_sums[i]
            
    # Find max transition probability per expert
    max_probs = []
    for i in range(128):
        if row_sums[i] > 0:
            max_probs.append(np.max(transitions_prob[i]))
    if max_probs:
        print(f"Average Max Cross-layer Expert Transition Probability P(E_{{L+2}} | E_L): {np.mean(max_probs)*100:.2f}%")
        print(f"Peak Max Cross-layer Expert Transition Probability P(E_{{L+2}} | E_L): {np.max(max_probs)*100:.2f}%")

if __name__ == "__main__":
    main()
