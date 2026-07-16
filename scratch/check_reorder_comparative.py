import json

def get_request_step_experts(req_data, chunk_size=1):
    moe_profile = req_data.get("moe_profile")
    if moe_profile is None or "routing_map" not in moe_profile:
        return set()

    is_prefill = req_data.get("num_computed_tokens", 0) < req_data.get("input_toks", 0)
    num_computed_tokens = req_data.get("num_computed_tokens", 0)
    original_input = req_data.get("input_toks", 0)

    experts = set()
    routing_map = moe_profile["routing_map"]
    for layer_key, layer_data in routing_map.items():
        if not layer_data:
            continue
        
        is_wrapped = False
        if len(layer_data) > 0 and isinstance(layer_data[0], list) and len(layer_data[0]) > 0 and isinstance(layer_data[0][0], list):
            is_wrapped = True

        if is_wrapped:
            if is_prefill:
                start_idx = num_computed_tokens
                end_idx = start_idx + chunk_size
                token_experts = layer_data[0][start_idx:end_idx]
                for tok_exps in token_experts:
                    for exp in tok_exps:
                        experts.add(exp)
            else:
                decode_idx = num_computed_tokens - original_input + 1
                if decode_idx < len(layer_data):
                    decode_exps = layer_data[decode_idx]
                    for tok_exps in decode_exps:
                        for exp in tok_exps:
                            experts.add(exp)
        else:
            if is_prefill:
                start_idx = num_computed_tokens
                end_idx = start_idx + chunk_size
                token_experts = layer_data[start_idx:end_idx]
                for item in token_experts:
                    if isinstance(item, list):
                        for exp in item:
                            experts.add(exp)
                    else:
                        experts.add(item)
            else:
                curr_idx = num_computed_tokens
                if curr_idx < len(layer_data):
                    item = layer_data[curr_idx]
                    if isinstance(item, list):
                        for exp in item:
                            experts.add(exp)
                    else:
                        experts.add(item)
    return experts

def get_request_step_experts_per_layer(req_data, chunk_size=1):
    moe_profile = req_data.get("moe_profile")
    if moe_profile is None or "routing_map" not in moe_profile:
        return {}

    is_prefill = req_data.get("num_computed_tokens", 0) < req_data.get("input_toks", 0)
    num_computed_tokens = req_data.get("num_computed_tokens", 0)
    original_input = req_data.get("input_toks", 0)

    experts_per_layer = {}
    routing_map = moe_profile["routing_map"]
    for layer_key, layer_data in routing_map.items():
        if not layer_data:
            continue
        
        is_wrapped = False
        if len(layer_data) > 0 and isinstance(layer_data[0], list) and len(layer_data[0]) > 0 and isinstance(layer_data[0][0], list):
            is_wrapped = True

        layer_exps = set()
        if is_wrapped:
            if is_prefill:
                start_idx = num_computed_tokens
                end_idx = start_idx + chunk_size
                token_experts = layer_data[0][start_idx:end_idx]
                for tok_exps in token_experts:
                    for exp in tok_exps:
                        layer_exps.add(exp)
            else:
                decode_idx = num_computed_tokens - original_input + 1
                if decode_idx < len(layer_data):
                    decode_exps = layer_data[decode_idx]
                    for tok_exps in decode_exps:
                        for exp in tok_exps:
                            layer_exps.add(exp)
        else:
            if is_prefill:
                start_idx = num_computed_tokens
                end_idx = start_idx + chunk_size
                token_experts = layer_data[start_idx:end_idx]
                for item in token_experts:
                    if isinstance(item, list):
                        for exp in item:
                            layer_exps.add(exp)
                    else:
                        layer_exps.add(item)
            else:
                curr_idx = num_computed_tokens
                if curr_idx < len(layer_data):
                    item = layer_data[curr_idx]
                    if isinstance(item, list):
                        for exp in item:
                            layer_exps.add(exp)
                    else:
                        layer_exps.add(item)
        experts_per_layer[layer_key] = layer_exps
    return experts_per_layer

def get_layer_by_layer_jaccard(exps_per_layer_A, exps_per_layer_B):
    common_layers = set(exps_per_layer_A.keys()).intersection(exps_per_layer_B.keys())
    if not common_layers:
        return 0.0
    
    total_jaccard = 0.0
    for layer_key in common_layers:
        setA = exps_per_layer_A[layer_key]
        setB = exps_per_layer_B[layer_key]
        if not setA and not setB:
            total_jaccard += 1.0
        else:
            intersection = len(setA.intersection(setB))
            union = len(setA.union(setB))
            total_jaccard += (intersection / union) if union > 0 else 0.0
            
    return total_jaccard / len(common_layers)

# Load dataset
reqs = []
with open("datasets/qwen3_remote_10req_concurrent.jsonl", "r") as f:
    for line in f:
        reqs.append(json.loads(line))

print("=== DECODE STEP 0 AFFINITIES ===")
for r in reqs:
    r["num_computed_tokens"] = r["input_toks"]

# Compare Jaccard for req 0 and req 1
r0 = reqs[0]
r1 = reqs[1]

# 1. Union method
exps_r0_union = get_request_step_experts(r0, chunk_size=1)
exps_r1_union = get_request_step_experts(r1, chunk_size=1)
jaccard_union = len(exps_r0_union.intersection(exps_r1_union)) / len(exps_r0_union.union(exps_r1_union))
print(f"Union Jaccard between Req 0 and Req 1: {jaccard_union:.4f}")

# 2. Layer-by-layer average method
exps_r0_layer = get_request_step_experts_per_layer(r0, chunk_size=1)
exps_r1_layer = get_request_step_experts_per_layer(r1, chunk_size=1)
jaccard_layer = get_layer_by_layer_jaccard(exps_r0_layer, exps_r1_layer)
print(f"Layer-by-Layer Avg Jaccard between Req 0 and Req 1: {jaccard_layer:.4f}")

print("\n--- ALL PAIRS DECODE AVG JACCARD (Layer-by-Layer vs Union) ---")
for i in range(5):
    for j in range(i+1, 5):
        ri = reqs[i]
        rj = reqs[j]
        
        # Union
        exps_ri_u = get_request_step_experts(ri, chunk_size=1)
        exps_rj_u = get_request_step_experts(rj, chunk_size=1)
        ju = len(exps_ri_u.intersection(exps_rj_u)) / len(exps_ri_u.union(exps_rj_u))
        
        # Layer-by-layer
        exps_ri_l = get_request_step_experts_per_layer(ri, chunk_size=1)
        exps_rj_l = get_request_step_experts_per_layer(rj, chunk_size=1)
        jl = get_layer_by_layer_jaccard(exps_ri_l, exps_rj_l)
        
        print(f"Req {i} vs Req {j} | Union Jaccard: {ju:.4f} | Layer-by-Layer Jaccard: {jl:.4f}")
