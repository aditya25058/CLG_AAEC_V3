import json
import random
import numpy as np
import os

# =====================================================================
# ZIPF POOL & NEURON ACTIVATION SIMULATOR
# (Matches the logic of gate_function.py for consistency)
# =====================================================================
_ZIPF_POOL = []

def init_zipf_pool():
    global _ZIPF_POOL
    if _ZIPF_POOL:
        return
    rng = random.Random(42)
    # Generate 500 pre-sampled Zipf-weighted index lists of size 64 from [0..1023]
    for _ in range(500):
        scores = []
        for idx in range(1024):
            weight = 1.0 / (idx + 1.0)
            score = rng.random() ** (1.0 / weight)
            scores.append((score, idx))
        scores.sort(reverse=True)
        active_indices = [idx for _, idx in scores[:64]]
        _ZIPF_POOL.append(active_indices)

init_zipf_pool()
_CONTEXT_GROUPS = [_ZIPF_POOL[i*50:(i+1)*50] for i in range(10)]

_CANDIDATE_CACHE = {}

def get_candidates(layer, expert_id):
    """Deterministically generate and cache 1024 neuron IDs for this expert and layer."""
    key = (layer, expert_id)
    if key not in _CANDIDATE_CACHE:
        candidates = []
        for idx in range(1024):
            nid = hash((layer, expert_id, idx)) % 4096
            candidates.append(nid)
        _CANDIDATE_CACHE[key] = candidates
    return _CANDIDATE_CACHE[key]

def simulate_neuron_activations_for_routing(routing_map, num_experts, skew_intensity=0.8):
    """
    Simulates the sequence of active neurons for each token's routing decisions
    using a random-walk Zipfian drift model to capture realistic temporal locality.
    """
    rng = random.Random(42)
    expert_history = {}
    
    layers = sorted(routing_map.keys(), key=lambda x: int(x.split('_')[1]) if '_' in x else int(x))
    
    for layer in layers:
        token_routings = routing_map[layer]
        
        for token_idx, routed_experts in enumerate(token_routings):
            for expert_id in routed_experts:
                key = (layer, expert_id)
                if key not in expert_history:
                    # Initialize with a random Zipf pattern from the pool
                    group_idx = rng.randint(0, len(_CONTEXT_GROUPS) - 1)
                    active_indices = list(rng.choice(_CONTEXT_GROUPS[group_idx]))
                    expert_history[key] = [active_indices]
                else:
                    # Random walk mutation to simulate semantic correlation decay
                    prev_indices = expert_history[key][-1]
                    new_indices = []
                    # Mutation rate: 0.1 at d=1 to show ~0.75 Jaccard overlap
                    mutation_rate = 0.12
                    
                    for idx in prev_indices:
                        if rng.random() < mutation_rate:
                            # Draw a new index from the Zipf pool
                            group_idx = rng.randint(0, len(_CONTEXT_GROUPS) - 1)
                            new_idx = rng.choice(rng.choice(_CONTEXT_GROUPS[group_idx]))
                            new_indices.append(new_idx)
                        else:
                            new_indices.append(idx)
                    expert_history[key].append(new_indices)
                    
    # Map index sequences to actual candidate neuron IDs
    final_history = {}
    for (layer, expert_id), indices_list in expert_history.items():
        candidates = get_candidates(layer, expert_id)
        final_history[(layer, expert_id)] = [
            set(candidates[idx] for idx in indices) for indices in indices_list
        ]
        
    return final_history

# =====================================================================
# DATASET ANALYZER
# =====================================================================
def analyze_dataset(file_path):
    print(f"Analyzing dataset: {file_path}")
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return None
        
    results = {
        "active_neurons_per_req_exp": [], # Union of active neurons per request-expert pair
        "unique_experts_per_req": [],     # Number of unique experts touched per request
        "jaccard_vs_distance": {},        # distance -> list of Jaccard overlaps
        "neuron_global_freqs": {},        # neuron_id -> global activation count
        "expert_skew_counts": [],         # expert_id -> activation counts
    }
    
    rng = random.Random(42)
    with open(file_path, "r") as f:
        lines = f.readlines()
        
    for line_idx, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            print(f"Skipping line {line_idx} due to JSON error: {e}")
            continue
            
        if "moe_profile" not in req:
            continue
            
        profile = req["moe_profile"]
        routing_map = profile["routing_map"]
        num_experts = profile["num_experts"]
        
        # Flatten routing map structure to make it uniform: list of lists of expert IDs
        uniform_routing_map = {}
        for layer, val in routing_map.items():
            # If it's a 3D list (e.g., [[[...]]]), extract the first element (batch dimension)
            if val and isinstance(val[0], list) and isinstance(val[0][0], list):
                uniform_routing_map[layer] = val[0]
            else:
                uniform_routing_map[layer] = val
        
        # 1. Expert-level routing stats
        all_routed_experts = []
        expert_activation_counts = np.zeros(num_experts)
        
        for layer, tokens in uniform_routing_map.items():
            for t in tokens:
                for exp in t:
                    all_routed_experts.append(exp)
                    expert_activation_counts[exp] += 1
                    
        results["unique_experts_per_req"].append(len(set(all_routed_experts)))
        results["expert_skew_counts"].append(expert_activation_counts.tolist())
        
        # 2. Simulate neuron activations for this request
        expert_history = simulate_neuron_activations_for_routing(uniform_routing_map, num_experts)
        
        # 3. Analyze neuron-level stats
        for (layer, expert_id), token_neuron_sets in expert_history.items():
            if not token_neuron_sets:
                continue
                
            # Union of all active neurons for this request-expert pair
            union_neurons = set()
            for s in token_neuron_sets:
                union_neurons.update(s)
            results["active_neurons_per_req_exp"].append(len(union_neurons))
            
            # Global frequency of neuron activations
            for s in token_neuron_sets:
                for nid in s:
                    full_nid = (layer, expert_id, nid)
                    results["neuron_global_freqs"][full_nid] = results["neuron_global_freqs"].get(full_nid, 0) + 1
            
            # Temporal similarity (Jaccard similarity vs distance) - Downsampled starting points
            n_tokens = len(token_neuron_sets)
            if n_tokens > 1:
                sample_indices = rng.sample(range(n_tokens), min(10, n_tokens))
                for i in sample_indices:
                    for dist in range(1, min(15, n_tokens - i)):
                        set_a = token_neuron_sets[i]
                        set_b = token_neuron_sets[i + dist]
                        
                        intersection = len(set_a.intersection(set_b))
                        union = len(set_a.union(set_b))
                        jaccard = intersection / union if union > 0 else 0.0
                        
                        if dist not in results["jaccard_vs_distance"]:
                            results["jaccard_vs_distance"][dist] = []
                        results["jaccard_vs_distance"][dist].append(jaccard)
                    
        print(f"  Processed request {line_idx + 1}/{len(lines)}...")
            
    # Aggregate Jaccard vs distance
    jaccard_avg = {}
    for dist, values in results["jaccard_vs_distance"].items():
        jaccard_avg[dist] = float(np.mean(values))
    results["jaccard_vs_distance"] = jaccard_avg
    
    return results

if __name__ == "__main__":
    qwen3_res = analyze_dataset("/home/palakm/MoEServingSim/datasets/qwen3_10req.jsonl")
    deepseek_res = analyze_dataset("/home/palakm/MoEServingSim/datasets/deepseek_10req.jsonl")
    
    # Save the analysis results to a json file
    output_data = {
        "qwen3": qwen3_res,
        "deepseek": deepseek_res
    }
    
    # Convert tuple keys to strings for JSON serialization
    for model_name in ["qwen3", "deepseek"]:
        if output_data[model_name] is not None:
            freqs = output_data[model_name]["neuron_global_freqs"]
            # Convert (layer, expert_id, nid) to string "layer:expert:nid"
            new_freqs = {f"{k[0]}:{k[1]}:{k[2]}": v for k, v in freqs.items()}
            output_data[model_name]["neuron_global_freqs"] = new_freqs
            
    out_path = "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/scratch/activation_stats.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output_data, f)
    print(f"Analysis complete! Results saved to {out_path}")
