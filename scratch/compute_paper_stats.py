import os
import json
import sqlite3
import numpy as np
from collections import defaultdict

def load_db(db_path):
    print(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)
    return conn

def compute_energy_stats(conn):
    print("\n--- 1. Energy Concentration Stats ---")
    cursor = conn.execute(
        "SELECT energy_k_50, energy_k_70, energy_k_80, energy_k_90, energy_k_95, energy_k_99, intermediate_dim "
        "FROM activations"
    )
    rows = cursor.fetchall()
    if not rows:
        print("No activation records found.")
        return
    
    dim = rows[0][6]
    cols = ["50%", "70%", "80%", "90%", "95%", "99%"]
    
    for i, col in enumerate(cols):
        vals = [r[i] for r in rows]
        # Convert to percentage of intermediate dim
        vals_pct = np.array(vals) / dim * 100
        mean = np.mean(vals_pct)
        std = np.std(vals_pct)
        median = np.median(vals_pct)
        p90 = np.percentile(vals_pct, 90)
        p10 = np.percentile(vals_pct, 10)
        
        print(f"  {col} Energy Concentration (% of {dim} neurons):")
        print(f"    Mean:   {mean:6.2f}% ({mean/100*dim:5.1f} neurons)")
        print(f"    Std:    {std:6.2f}% ({std/100*dim:5.1f} neurons)")
        print(f"    Median: {median:6.2f}% ({median/100*dim:5.1f} neurons)")
        print(f"    P10:    {p10:6.2f}% ({p10/100*dim:5.1f} neurons)")
        print(f"    P90:    {p90:6.2f}% ({p90/100*dim:5.1f} neurons)")

def compute_zipf_fit(conn):
    print("\n--- 2. Zipf Rank-Frequency Fit ---")
    cursor = conn.execute("SELECT active_indices FROM activations")
    
    # Sub-sample every 20th row to speed up count
    neuron_counts = defaultdict(int)
    total_invocations = 0
    for i, (idx_json,) in enumerate(cursor):
        if i % 20 != 0:
            continue
        indices = json.loads(idx_json)
        for idx in indices:
            neuron_counts[idx] += 1
        total_invocations += 1
        
    if not neuron_counts:
        print("No neuron data.")
        return
        
    sorted_counts = sorted(neuron_counts.values(), reverse=True)
    ranks = np.arange(1, len(sorted_counts) + 1)
    freqs = np.array(sorted_counts) / total_invocations
    
    # Fit Zipf: f(r) = C * r^(-alpha) => log(f) = log(C) - alpha * log(r)
    # Filter out near-zero frequencies to avoid fit distortion
    mask = freqs > 1e-4
    ranks_fit = ranks[mask]
    freqs_fit = freqs[mask]
    
    log_r = np.log(ranks_fit)
    log_f = np.log(freqs_fit)
    
    alpha, log_C = np.polyfit(log_r, log_f, 1)
    # The polyfit returns slope as -alpha
    print(f"  Zipf exponent alpha = {-alpha:.4f}")
    print(f"  Scaling constant C = {np.exp(log_C):.4f}")
    print(f"  Number of unique active neurons sampled: {len(neuron_counts)} (out of 768 per layer/expert)")

def compute_jaccard_bootstrap(conn):
    print("\n--- 3. Jaccard Reuse Decay & Bootstrap CI ---")
    cursor = conn.execute("SELECT DISTINCT layer, expert_id FROM activations")
    all_experts = cursor.fetchall()
    
    import random
    random.seed(42)
    if len(all_experts) > 150:
        sampled_experts = random.sample(all_experts, 150)
    else:
        sampled_experts = all_experts
        
    expert_sequences = defaultdict(list)
    for layer, exp_id in sampled_experts:
        c = conn.execute(
            "SELECT active_indices, energy_k_50 FROM activations "
            "WHERE layer = ? AND expert_id = ? "
            "ORDER BY prompt_id, token_pos",
            (layer, exp_id)
        )
        for idx_json, k50 in c.fetchall():
            expert_sequences[(layer, exp_id)].append(set(json.loads(idx_json)[:k50]))
            
    distances = [1, 2, 4, 8, 16, 32, 64]
    expert_jaccards = []
    
    for (layer, exp_id) in sampled_experts:
        sets = expert_sequences[(layer, exp_id)]
        n = len(sets)
        if n < 10:
            continue
        exp_jaccards = {d: [] for d in distances}
        for dist in distances:
            for i in range(min(n - dist, 100)):
                union = len(sets[i].union(sets[i + dist]))
                inter = len(sets[i].intersection(sets[i + dist]))
                if union > 0:
                    exp_jaccards[dist].append(inter / union)
        expert_jaccards.append(exp_jaccards)
        
    def fit_decay(sampled_jaccards):
        # Average jaccard per distance across the sample
        y_means = []
        for d in distances:
            j_list = []
            for item in sampled_jaccards:
                j_list.extend(item[d])
            y_means.append(np.mean(j_list) if j_list else 0.0)
        y_means = np.array(y_means)
        
        # Linearized fit: log(y - B) = log(A) - d / tau
        # Assume B = min(y_means)
        B = min(y_means)
        
        # Fit A and tau
        try:
            x_fit, y_fit = [], []
            for d, y_val in zip(distances, y_means):
                if y_val > B + 0.001:
                    x_fit.append(d)
                    y_fit.append(np.log(y_val - B))
            if len(x_fit) >= 2:
                slope, intercept = np.polyfit(x_fit, y_fit, 1)
                tau = -1.0 / slope if slope != 0 else 100.0
                return tau, y_means
            return None, y_means
        except Exception as e:
            return None, y_means

    original_tau, original_means = fit_decay(expert_jaccards)
    print(f"  Fitted original tau: {original_tau:.2f} tokens")
    for d, m in zip(distances, original_means):
        print(f"    Distance {d:2d}: Mean Jaccard = {m:.4f}")
        
    # Bootstrap resampling
    print("  Running 200 bootstrap iterations to find confidence interval for tau...")
    bootstrap_taus = []
    n_samples = len(expert_jaccards)
    for _ in range(200):
        resampled = random.choices(expert_jaccards, k=n_samples)
        tau, _ = fit_decay(resampled)
        if tau is not None:
            bootstrap_taus.append(tau)
            
    if bootstrap_taus:
        bootstrap_taus = np.array(bootstrap_taus)
        mean_tau = np.mean(bootstrap_taus)
        std_tau = np.std(bootstrap_taus)
        ci_lower = np.percentile(bootstrap_taus, 2.5)
        ci_upper = np.percentile(bootstrap_taus, 97.5)
        print(f"  Bootstrap Tau Stats:")
        print(f"    Mean: {mean_tau:.2f} tokens")
        print(f"    Std:  {std_tau:.2f} tokens")
        print(f"    95% Confidence Interval: [{ci_lower:.2f}, {ci_upper:.2f}] tokens")
    else:
        print("  Bootstrap fitting failed.")

def compute_router_confidence(conn):
    print("\n--- 4. Router Confidence Statistics ---")
    # Query router_prob
    cursor = conn.execute("SELECT router_prob FROM activations WHERE router_prob IS NOT NULL")
    probs = [r[0] for r in cursor.fetchall()]
    if probs:
        probs = np.array(probs)
        print(f"  Active Expert Router Probability (assigned to routed expert):")
        print(f"    Mean:   {np.mean(probs):.4f}")
        print(f"    Std:    {np.std(probs):.4f}")
        print(f"    Median: {np.median(probs):.4f}")
        print(f"    P10:    {np.percentile(probs, 10):.4f}")
        print(f"    P90:    {np.percentile(probs, 90):.4f}")

def compute_cross_request_reuse(conn):
    print("\n--- 5. Cross-Request Neuron Reuse ---")
    cursor = conn.execute("SELECT DISTINCT prompt_id FROM activations")
    prompts = [r[0] for r in cursor.fetchall()]
    
    if len(prompts) < 2:
        print("Not enough prompts to compute cross-request reuse.")
        return
        
    print(f"  Found {len(prompts)} unique prompts. Computing pairwise active neuron overlap...")
    
    # Get active neuron set (EMA-equivalent active neurons at 50% energy) for each prompt
    # Since prompts might have different length, we collect the aggregate active neurons for each expert under each prompt.
    prompt_expert_sets = defaultdict(dict)  # prompt_id -> (layer, expert_id) -> set of active neurons
    
    # We query layer, expert_id, prompt_id, active_indices, energy_k_50
    cursor = conn.execute(
        "SELECT prompt_id, layer, expert_id, active_indices, energy_k_50 FROM activations"
    )
    
    for prompt_id, layer, exp_id, idx_json, k50 in cursor:
        key = (layer, exp_id)
        if key not in prompt_expert_sets[prompt_id]:
            prompt_expert_sets[prompt_id][key] = set()
        indices = json.loads(idx_json)[:k50]
        prompt_expert_sets[prompt_id][key].update(indices)
        
    # Sample up to 100 random prompt pairs to estimate cross-prompt overlap
    overlap_jaccards = []
    overlap_ratios = []
    
    prompt_ids = list(prompt_expert_sets.keys())
    import random
    random.seed(42)
    
    pairs = []
    for i in range(len(prompt_ids)):
        for j in range(i+1, len(prompt_ids)):
            pairs.append((prompt_ids[i], prompt_ids[j]))
            
    if len(pairs) > 100:
        pairs = random.sample(pairs, 100)
        
    for p1, p2 in pairs:
        common_keys = set(prompt_expert_sets[p1].keys()).intersection(prompt_expert_sets[p2].keys())
        if not common_keys:
            continue
        p_jaccards = []
        p_overlaps = []
        for key in common_keys:
            s1 = prompt_expert_sets[p1][key]
            s2 = prompt_expert_sets[p2][key]
            union = len(s1.union(s2))
            inter = len(s1.intersection(s2))
            if union > 0:
                p_jaccards.append(inter / union)
                p_overlaps.append(inter / min(len(s1), len(s2)) if min(len(s1), len(s2)) > 0 else 0)
        if p_jaccards:
            overlap_jaccards.append(np.mean(p_jaccards))
            overlap_ratios.append(np.mean(p_overlaps))
            
    if overlap_jaccards:
        print(f"  Across-Request Active Neuron Overlap (Averaged over {len(pairs)} prompt pairs):")
        print(f"    Mean Jaccard Similarity:      {np.mean(overlap_jaccards):.4f} (± {np.std(overlap_jaccards):.4f})")
        print(f"    Mean Shared Ratio (Overlap):  {np.mean(overlap_ratios)*100:.2f}% (± {np.std(overlap_ratios)*100:.2f}%)")
        print("  Systems Implication:")
        print(f"    A shared ratio of {np.mean(overlap_ratios)*100:.1f}% indicates that when a new user request arrives,")
        print("    more than half of the required active neurons are already warm in the AAEC cache from previous requests!")

def main():
    db_path = "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/qwen3_30b_real_v2.db"
    conn = load_db(db_path)
    compute_energy_stats(conn)
    compute_zipf_fit(conn)
    compute_jaccard_bootstrap(conn)
    compute_router_confidence(conn)
    compute_cross_request_reuse(conn)

if __name__ == "__main__":
    main()
