import os
import json
import sqlite3
import numpy as np
from collections import defaultdict

DB_PATH = "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/qwen3_30b_real_v2.db"
OUTPUT_PATH = "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/scratch/adetr_permutations.json"

def compute_clusters(db_path, output_path, block_size=32):
    print(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("Fetching activation records from 'activations' table...")
    cursor.execute("SELECT layer, expert_id, active_indices FROM activations")
    rows = cursor.fetchall()
    print(f"Fetched {len(rows)} rows.")

    print("Grouping activations per (layer, expert)...")
    activations_by_expert = defaultdict(list)
    for layer, expert_id, active_indices_str in rows:
        try:
            indices = json.loads(active_indices_str)
            activations_by_expert[(layer, expert_id)].append(indices)
        except Exception as e:
            continue

    print(f"Total unique (layer, expert) pairs: {len(activations_by_expert)}")

    permutations = {}
    num_neurons = 768

    completed_count = 0
    total_pairs = len(activations_by_expert)

    print("Computing optimal column permutations using vectorized Jaccard matrices...")
    for (layer, expert_id), token_activations in activations_by_expert.items():
        T = len(token_activations)
        if T == 0:
            permutations[f"{layer}_{expert_id}"] = list(range(num_neurons))
            continue

        # Build binary activity matrix A of shape (T, num_neurons)
        A = np.zeros((T, num_neurons), dtype=np.float32)
        for t, indices in enumerate(token_activations):
            A[t, indices] = 1.0

        # Compute co-occurrence matrix C = A.T @ A
        C = A.T @ A

        # Frequencies are diagonal elements
        F = np.diag(C)

        # Compute pairwise Jaccard similarity: J(i, j) = C(i, j) / (F(i) + F(j) - C(i, j))
        # Add epsilon to prevent division by zero
        F_row = F[:, np.newaxis]
        F_col = F[np.newaxis, :]
        denom = F_row + F_col - C
        denom[denom <= 0] = 1.0 # prevent division by zero
        J = C / denom
        
        # Zero out diagonal
        np.fill_diagonal(J, 0.0)

        # Greedy clustering
        unassigned = set(range(num_neurons))
        permutation_order = []

        # Priority seed order: highest frequency first
        priority_neurons = np.argsort(F)[::-1]

        while unassigned:
            # Find the first priority neuron that is still unassigned
            seed = None
            for n in priority_neurons:
                if n in unassigned:
                    seed = int(n)
                    break
            
            if seed is None:
                break

            # Start a new cluster
            cluster = [seed]
            unassigned.remove(seed)

            # Track cumulative Jaccard similarity of candidates with the current cluster
            cum_sim = J[seed].copy()

            while len(cluster) < block_size and unassigned:
                # Find candidate with max average similarity to current cluster members
                # Average similarity is cum_sim / len(cluster)
                # Since len(cluster) is constant for all candidates, we just find argmax of cum_sim
                # over the unassigned set.
                best_neuron = None
                best_val = -1.0
                
                for cand in unassigned:
                    val = cum_sim[cand]
                    if val > best_val:
                        best_val = val
                        best_neuron = cand

                if best_neuron is not None and best_val > 0:
                    cluster.append(best_neuron)
                    unassigned.remove(best_neuron)
                    # Update cumulative similarities
                    cum_sim += J[best_neuron]
                else:
                    # No correlated candidate left
                    break

            permutation_order.extend(cluster)

        # Append leftovers if any
        if len(permutation_order) < num_neurons:
            leftovers = list(unassigned)
            permutation_order.extend(leftovers)

        permutations[f"{layer}_{expert_id}"] = permutation_order

        completed_count += 1
        if completed_count % 1000 == 0 or completed_count == total_pairs:
            print(f"  Processed {completed_count}/{total_pairs} pairs...")

    print("Filling missing layer-expert pairs with default identity...")
    for l in range(48):
        for e in range(128):
            key = f"{l}_{e}"
            if key not in permutations:
                permutations[key] = list(range(num_neurons))

    print(f"Saving permutations to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(permutations, f)
    print("Permutations computation successfully completed!")

    conn.close()

if __name__ == "__main__":
    compute_clusters(DB_PATH, OUTPUT_PATH)
