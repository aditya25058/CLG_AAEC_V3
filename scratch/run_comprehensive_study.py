import os
import json
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def save_heatmap(data, title, xlabel, ylabel, clabel, cmap, filename):
    plt.figure(figsize=(8, 6.5))
    im = plt.imshow(data, aspect='auto', cmap=cmap, interpolation='nearest')
    plt.colorbar(im, label=clabel)
    plt.xlabel(xlabel, fontsize=11, fontweight='bold')
    plt.ylabel(ylabel, fontsize=11, fontweight='bold')
    plt.title(title, fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

def main():
    db_path = "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/qwen3_30b_real_v2.db"
    out_dir = "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)
    
    # ---------------------------------------------------------
    # Helper: Load all activations
    # ---------------------------------------------------------
    print("Loading activations data...")
    cursor = conn.execute(
        "SELECT layer, expert_id, prompt_id, token_pos, active_indices, "
        "energy_k_50, energy_k_70, energy_k_80, energy_k_90, energy_k_95, energy_k_99, "
        "intermediate_dim FROM activations ORDER BY prompt_id, token_pos, layer, expert_id"
    )
    rows = cursor.fetchall()
    print(f"Loaded {len(rows)} activation records.")
    
    # ---------------------------------------------------------
    # Experiment 1: Spatial Footprint & Working-set Growth
    # ---------------------------------------------------------
    print("\n--- Running Experiment 1: Working-Set Growth ---")
    # Group by (layer, expert_id, prompt_id)
    seq_activations = defaultdict(list)
    for r in rows:
        key = (r[0], r[1], r[2])
        indices = json.loads(r[4])[:r[5]] # 50% energy set
        seq_activations[key].append((r[3], set(indices))) # (token_pos, active_set)
        
    n_values = [1, 2, 4, 8, 16, 32, 64, 128, 200]
    growth_stats = defaultdict(list)
    
    for key, tokens_data in seq_activations.items():
        sorted_tokens = sorted(tokens_data, key=lambda x: x[0])
        cumulative_set = set()
        for idx, (t_pos, active_set) in enumerate(sorted_tokens):
            cumulative_set.update(active_set)
            step = idx + 1
            if step in n_values:
                growth_stats[step].append(len(cumulative_set) / 768.0 * 100.0)
                
    # Filter to actual step values present in stats
    actual_n = [n for n in n_values if n in growth_stats and len(growth_stats[n]) > 0]
    
    print("Working-set growth W(n) (% of intermediate size 768):")
    for n in actual_n:
        arr = np.array(growth_stats[n])
        print(f"  n = {n:3d}: Mean = {np.mean(arr):6.2f}%, Median = {np.median(arr):6.2f}%, Std = {np.std(arr):6.2f}%, P90 = {np.percentile(arr, 90):6.2f}%, P95 = {np.percentile(arr, 95):6.2f}%")
        
    # Generate Working-Set Growth Plot
    plt.figure(figsize=(7, 4.5))
    x_steps = actual_n
    y_means = [np.mean(growth_stats[n]) for n in actual_n]
    y_p90 = [np.percentile(growth_stats[n], 90) for n in actual_n]
    y_p10 = [np.percentile(growth_stats[n], 10) for n in actual_n]
    plt.plot(x_steps, y_means, marker='o', linewidth=2.5, color='#1d3557', label='Mean Footprint')
    plt.fill_between(x_steps, y_p10, y_p90, color='#1d3557', alpha=0.15, label='P10 - P90 range')
    plt.xscale('log')
    plt.xticks(actual_n, [str(n) for n in actual_n])
    plt.xlabel('Sequence Length (Tokens)', fontsize=11, fontweight='bold')
    plt.ylabel('Unique Neuron Footprint (%)', fontsize=11, fontweight='bold')
    plt.title('Working-Set Growth W(n) (50% Energy Budget)', fontsize=12, fontweight='bold', pad=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "working_set_growth_n.png"), dpi=200)
    plt.close()
    
    # ---------------------------------------------------------
    # Experiment 2: Energy Distribution CDF
    # ---------------------------------------------------------
    print("\n--- Running Experiment 2: Energy Distribution CDF ---")
    thresholds = [50, 70, 80, 90, 95, 99]
    energy_means = []
    energy_stds = []
    
    for idx, th in enumerate(thresholds):
        vals = [r[idx + 5] for r in rows]
        vals_pct = np.array(vals) / 768.0 * 100.0
        energy_means.append(np.mean(vals_pct))
        energy_stds.append(np.std(vals_pct))
        
    print("Energy Concentration CDF values:")
    for th, mean, std in zip(thresholds, energy_means, energy_stds):
        print(f"  {th}% Energy: Mean = {mean:6.2f}% of FFN ({mean/100*768:5.1f} neurons), Std = {std:6.2f}%")
        
    plt.figure(figsize=(7, 4.5))
    plt.bar([f"{th}%" for th in thresholds], energy_means, yerr=energy_stds, color='#457b9d', capsize=5, edgecolor='black', alpha=0.85)
    plt.xlabel('Cumulative Absolute Energy Target', fontsize=11, fontweight='bold')
    plt.ylabel('Neuron Footprint Required (%)', fontsize=11, fontweight='bold')
    plt.title('Neuron Slicing vs. Captured Activation Energy', fontsize=12, fontweight='bold', pad=12)
    plt.grid(axis='y', ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "energy_cdf.png"), dpi=200)
    plt.close()
    
    # ---------------------------------------------------------
    # Experiment 3: Layer-wise Heterogeneity Heatmap
    # ---------------------------------------------------------
    print("\n--- Running Experiment 3: Layer-wise Heterogeneity Heatmap ---")
    layer_expert_density = np.zeros((48, 128))
    layer_expert_counts = np.zeros((48, 128))
    for r in rows:
        layer_expert_density[r[0], r[1]] += r[5] # energy_k_50
        layer_expert_counts[r[0], r[1]] += 1
        
    for l in range(48):
        for e in range(128):
            if layer_expert_counts[l, e] > 0:
                layer_expert_density[l, e] /= layer_expert_counts[l, e]
                
    save_heatmap(layer_expert_density, 'MoE Layer × Expert Neuron Density Heatmap', 'Expert Index', 'Layer Index', 'Active Neurons (50% Energy)', 'YlGnBu', os.path.join(out_dir, "layer_expert_heatmap.png"))
    
    # ---------------------------------------------------------
    # Experiment 4: Temporal Locality & Working-set Survival
    # ---------------------------------------------------------
    print("\n--- Running Experiment 4: Temporal Locality & Working-Set Survival ---")
    distances = [1, 2, 4, 8, 16, 32, 64]
    jaccards = {d: [] for d in distances}
    survivals = {d: [] for d in distances}
    
    for key, tokens_data in seq_activations.items():
        sorted_tokens = sorted(tokens_data, key=lambda x: x[0])
        n = len(sorted_tokens)
        if n < 10:
            continue
        for dist in distances:
            for i in range(min(n - dist, 100)):
                s1 = sorted_tokens[i][1]
                s2 = sorted_tokens[i + dist][1]
                
                union = len(s1.union(s2))
                inter = len(s1.intersection(s2))
                if union > 0:
                    jaccards[dist].append(inter / union)
                if len(s1) > 0:
                    survivals[dist].append(inter / len(s1))
                    
    actual_distances = [d for d in distances if len(jaccards[d]) > 0 and len(survivals[d]) > 0]
    
    print("Temporal locality metrics:")
    for d in actual_distances:
        j_arr = np.array(jaccards[d])
        s_arr = np.array(survivals[d])
        print(f"  dist = {d:2d}: Mean Jaccard = {np.mean(j_arr):.4f} | Mean Survival = {np.mean(s_arr):.4f}")
        
    plt.figure(figsize=(7, 4.5))
    y_j = [np.mean(jaccards[d]) for d in actual_distances]
    y_s = [np.mean(survivals[d]) for d in actual_distances]
    plt.plot(actual_distances, y_j, marker='o', linewidth=2.5, color='#d90429', label='Jaccard Similarity')
    plt.plot(actual_distances, y_s, marker='s', linewidth=2.5, color='#ef233c', label='Survival Probability')
    plt.xscale('log')
    plt.xticks(actual_distances, [str(d) for d in actual_distances])
    plt.xlabel('Token Distance (Visits)', fontsize=11, fontweight='bold')
    plt.ylabel('Score', fontsize=11, fontweight='bold')
    plt.title('Neuron Working-Set Temporal Locality & Survival Decay', fontsize=12, fontweight='bold', pad=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "working_set_survival.png"), dpi=200)
    plt.close()
    
    # ---------------------------------------------------------
    # Experiment 5: Active Neuron Stability
    # ---------------------------------------------------------
    print("\n--- Running Experiment 5: Active Neuron Stability ---")
    expert_invocations = defaultdict(int)
    neuron_active_counts = defaultdict(lambda: defaultdict(int))
    
    for r in rows:
        key = (r[0], r[1])
        expert_invocations[key] += 1
        indices = json.loads(r[4])[:r[5]] # 50% energy set
        for idx in indices:
            neuron_active_counts[key][idx] += 1
            
    always_hot = 0
    context_hot = 0
    rare = 0
    total_tracked = 0
    
    for key, counts in neuron_active_counts.items():
        total_inv = expert_invocations[key]
        for neuron_id in range(768):
            act_count = counts.get(neuron_id, 0)
            ratio = act_count / total_inv if total_inv > 0 else 0
            if ratio > 0.80:
                always_hot += 1
            elif ratio > 0.10:
                context_hot += 1
            else:
                rare += 1
            total_tracked += 1
            
    print(f"Neuron stability classification:")
    print(f"  Always Hot (freq > 80%):    {always_hot:6d} ({always_hot/total_tracked*100:5.2f}%)")
    print(f"  Context Hot (10% < freq <= 80%): {context_hot:6d} ({context_hot/total_tracked*100:5.2f}%)")
    print(f"  Rare (freq <= 10%):         {rare:6d} ({rare/total_tracked*100:5.2f}%)")
    
    # ---------------------------------------------------------
    # Experiment 6: Router Entropy & Confidence Margin
    # ---------------------------------------------------------
    print("\n--- Running Experiment 6: Router Entropy & Margin ---")
    layer_visits = defaultdict(lambda: defaultdict(int))
    layer_totals = defaultdict(int)
    for r in rows:
        layer_visits[r[0]][r[1]] += 1
        layer_totals[r[0]] += 1
        
    layer_entropies = {}
    for layer in sorted(layer_visits.keys()):
        total = layer_totals[layer]
        probs = [layer_visits[layer][exp] / total for exp in layer_visits[layer].keys()]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        layer_entropies[layer] = entropy
        
    print("Layer-wise routing entropy (out of max 6.88 bits):")
    for layer in [0, 8, 16, 24, 32, 40, 47]:
        print(f"  Layer {layer:2d}: Entropy = {layer_entropies[layer]:.4f} bits ({layer_entropies[layer]/6.882*100:.1f}% uniform)")
        
    # ---------------------------------------------------------
    # Experiment 7: Expert Popularity Zipf Ranking
    # ---------------------------------------------------------
    print("\n--- Running Experiment 7: Expert Popularity Zipf ---")
    expert_overall_visits = defaultdict(int)
    for r in rows:
        expert_overall_visits[r[1]] += 1
        
    sorted_visits = sorted(expert_overall_visits.values(), reverse=True)
    ranks = np.arange(1, len(sorted_visits) + 1)
    
    slope, intercept = np.polyfit(np.log(ranks), np.log(sorted_visits), 1)
    print(f"  Expert Zipf popularity exponent alpha = {-slope:.4f}")
    
    plt.figure(figsize=(7, 4.5))
    plt.loglog(ranks, sorted_visits, marker='o', linewidth=2.5, color='#e63946', label='Empirical Visit Counts')
    plt.loglog(ranks, np.exp(intercept) * (ranks ** slope), ls='--', color='black', label=f'Zipf Fit (alpha = {-slope:.2f})')
    plt.xlabel('Expert Rank', fontsize=11, fontweight='bold')
    plt.ylabel('Visit Count', fontsize=11, fontweight='bold')
    plt.title('Expert Popularity Zipf Distribution', fontsize=12, fontweight='bold', pad=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "expert_popularity_zipf.png"), dpi=200)
    plt.close()
    
    # ---------------------------------------------------------
    # Experiment 8: Transition Matrix
    # ---------------------------------------------------------
    print("\n--- Running Experiment 8: Transition Matrix ---")
    cursor = conn.execute(
        "SELECT prompt_id, layer, token_pos, expert_id FROM activations "
        "ORDER BY prompt_id, layer, token_pos"
    )
    routing_seq = defaultdict(list)
    for p_id, layer, t_pos, exp_id in cursor:
        routing_seq[(p_id, layer)].append(exp_id)
        
    transition_counts = np.zeros((128, 128))
    for key, seq in routing_seq.items():
        for i in range(len(seq) - 1):
            e_from = seq[i]
            e_to = seq[i + 1]
            transition_counts[e_from, e_to] += 1
            
    transition_prob = np.zeros((128, 128))
    for i in range(128):
        row_sum = np.sum(transition_counts[i])
        if row_sum > 0:
            transition_prob[i] = transition_counts[i] / row_sum
            
    print("Transition probabilities from sample experts:")
    for e in [10, 45, 92]:
        top_dest = np.argsort(transition_prob[e])[::-1][:3]
        print(f"  Expert {e:2d} -> Expert {top_dest[0]:2d} ({transition_prob[e, top_dest[0]]*100:.1f}%), Expert {top_dest[1]:2d} ({transition_prob[e, top_dest[1]]*100:.1f}%)")
        
    save_heatmap(transition_prob, 'Expert Routing Transition Probability Matrix', 'To Expert Index', 'From Expert Index', 'Transition Probability', 'Purples', os.path.join(out_dir, "expert_transition_matrix.png"))
    
    # ---------------------------------------------------------
    # Experiment 9: Co-occurrence Matrix
    # ---------------------------------------------------------
    print("\n--- Running Experiment 9: Co-occurrence Matrix ---")
    token_experts = defaultdict(set)
    for r in rows:
        key = (r[2], r[0], r[3])
        token_experts[key].add(r[1])
        
    co_counts = np.zeros((128, 128))
    for key, experts in token_experts.items():
        exp_list = list(experts)
        for i in range(len(exp_list)):
            for j in range(i + 1, len(exp_list)):
                e1 = exp_list[i]
                e2 = exp_list[j]
                co_counts[e1, e2] += 1
                co_counts[e2, e1] += 1
                
    co_occurrence = np.zeros((128, 128))
    for i in range(128):
        occur = expert_overall_visits[i]
        if occur > 0:
            co_occurrence[i] = co_counts[i] / occur
            
    save_heatmap(co_occurrence, 'Expert Co-occurrence Probability Matrix', 'Expert Index', 'Expert Index', 'Co-occurrence Frequency', 'Reds', os.path.join(out_dir, "expert_cooccurrence.png"))
    
    # ---------------------------------------------------------
    # Experiment 10: Cross-Request Locality
    # ---------------------------------------------------------
    print("\n--- Running Experiment 10: Cross-Request Locality ---")
    prompt_expert_sets = defaultdict(dict)
    for r in rows:
        key = (r[0], r[1])
        indices = json.loads(r[4])[:r[5]] # 50% energy set
        if key not in prompt_expert_sets[r[2]]:
            prompt_expert_sets[r[2]][key] = set()
        prompt_expert_sets[r[2]][key].update(indices)
        
    def compute_pair_overlap(p1, p2):
        common_keys = set(prompt_expert_sets[p1].keys()).intersection(prompt_expert_sets[p2].keys())
        if not common_keys:
            return 0.0, 0.0
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
        return np.mean(p_jaccards) if p_jaccards else 0.0, np.mean(p_overlaps) if p_overlaps else 0.0

    same_prompt_category = []
    diff_prompt_category = []
    
    import random
    random.seed(42)
    
    for cat in [range(0, 10), range(10, 20), range(20, 30)]:
        cat_list = list(cat)
        for i in range(len(cat_list)):
            for j in range(i+1, len(cat_list)):
                _, overlap = compute_pair_overlap(cat_list[i], cat_list[j])
                same_prompt_category.append(overlap)
                
    for p1 in range(0, 10):
        for p2 in range(35, 40):
            _, overlap = compute_pair_overlap(p1, p2)
            diff_prompt_category.append(overlap)
            
    print(f"Cross-Request active neuron overlaps:")
    print(f"  Same Category: Mean Overlap = {np.mean(same_prompt_category)*100:6.2f}% (± {np.std(same_prompt_category)*100:.2f}%)")
    print(f"  Diff Category: Mean Overlap = {np.mean(diff_prompt_category)*100:6.2f}% (± {np.std(diff_prompt_category)*100:.2f}%)")
    
    plt.figure(figsize=(6, 4))
    plt.boxplot([same_prompt_category, diff_prompt_category], labels=['Same Category', 'Different Category'], patch_artist=True,
                boxprops=dict(facecolor='#a8dadc', color='black'), medianprops=dict(color='red', linewidth=1.5))
    plt.ylabel('Active Neuron Overlap Ratio (%)', fontsize=11, fontweight='bold')
    plt.title('Cross-Request Active Neuron Overlap', fontsize=12, fontweight='bold', pad=12)
    plt.grid(axis='y', ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cross_request_locality.png"), dpi=200)
    plt.close()
    
    # ---------------------------------------------------------
    # Experiment 11: Cache Simulation
    # ---------------------------------------------------------
    print("\n--- Running Experiment 11: Cache Simulation ---")
    expert_sequences_rep = defaultdict(list)
    for r in rows:
        key = (r[0], r[1])
        indices = json.loads(r[4])[:r[5]] # 50% energy set
        expert_sequences_rep[key].append(indices)
        
    sampled_keys = random.sample(list(expert_sequences_rep.keys()), min(100, len(expert_sequences_rep)))
    
    lru_hits, lru_totals = 0, 0
    lfu_hits, lfu_totals = 0, 0
    ema_hits, ema_totals = 0, 0
    aaec_hits, aaec_totals = 0, 0
    
    for key in sampled_keys:
        seq = expert_sequences_rep[key]
        if len(seq) < 20:
            continue
            
        all_counts = defaultdict(int)
        for act in seq:
            for idx in act:
                all_counts[idx] += 1
        sorted_neurons = sorted(all_counts.keys(), key=lambda x: all_counts[x], reverse=True)
        static_head = set(sorted_neurons[:32])
        
        lru_cache = []
        lfu_freqs = defaultdict(int)
        ema_weights = defaultdict(float)
        aaec_lru = []
        
        for act in seq:
            # 1. LRU Cache
            for neuron in act:
                lru_totals += 1
                if neuron in lru_cache:
                    lru_hits += 1
                    lru_cache.remove(neuron)
                    lru_cache.append(neuron)
                else:
                    if len(lru_cache) >= 128:
                        lru_cache.pop(0)
                    lru_cache.append(neuron)
                    
            # 2. LFU Cache (Sort frequency keys once per token instead of once per neuron)
            top_lfu = set(sorted(lfu_freqs.keys(), key=lambda x: lfu_freqs[x], reverse=True)[:128])
            for neuron in act:
                lfu_totals += 1
                if neuron in top_lfu:
                    lfu_hits += 1
                lfu_freqs[neuron] += 1
                
            # 3. EMA Cache (Sort EMA weights once per token)
            top_ema = set(sorted(ema_weights.keys(), key=lambda x: ema_weights[x], reverse=True)[:128])
            for neuron in act:
                ema_totals += 1
                if neuron in top_ema:
                    ema_hits += 1
                    
            # Decay and update EMA weights
            for n_id in list(ema_weights.keys()):
                ema_weights[n_id] *= 0.95
            for neuron in act:
                ema_weights[neuron] += 0.05
                
            # 4. AAEC Cache (Static 32 + Dynamic 96)
            for neuron in act:
                aaec_totals += 1
                if neuron in static_head:
                    aaec_hits += 1
                elif neuron in aaec_lru:
                    aaec_hits += 1
                    aaec_lru.remove(neuron)
                    aaec_lru.append(neuron)
                else:
                    if len(aaec_lru) >= 96:
                        aaec_lru.pop(0)
                    aaec_lru.append(neuron)
                    
    hit_lru = lru_hits / lru_totals if lru_totals > 0 else 0
    hit_lfu = lfu_hits / lfu_totals if lfu_totals > 0 else 0
    hit_ema = ema_hits / ema_totals if ema_totals > 0 else 0
    hit_aaec = aaec_hits / aaec_totals if aaec_totals > 0 else 0
    
    print("Cache Simulation Results:")
    print(f"  LRU Cache Hit Rate  : {hit_lru*100:6.2f}%")
    print(f"  LFU Cache Hit Rate  : {hit_lfu*100:6.2f}%")
    print(f"  EMA Cache Hit Rate  : {hit_ema*100:6.2f}%")
    print(f"  AAEC Cache Hit Rate : {hit_aaec*100:6.2f}% (Static 32 + Dynamic 96)")
    
    plt.figure(figsize=(7, 4.5))
    policies = ['LRU', 'LFU', 'EMA', 'AAEC (Ours)']
    hits = [hit_lru * 100, hit_lfu * 100, hit_ema * 100, hit_aaec * 100]
    plt.bar(policies, hits, color=['#e63946', '#ffd166', '#06d6a0', '#118ab2'], edgecolor='black', width=0.55)
    plt.ylabel('Cache Hit Rate (%)', fontsize=11, fontweight='bold')
    plt.title('MoE Neuron Cache Policy Comparison (Cache Size = 128)', fontsize=12, fontweight='bold', pad=12)
    plt.ylim(0, 100)
    plt.grid(axis='y', ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cache_simulation_comparison.png"), dpi=200)
    plt.close()
    
    print("\nAll 9 publication plots saved successfully in artifacts root!")
    conn.close()

if __name__ == "__main__":
    main()
