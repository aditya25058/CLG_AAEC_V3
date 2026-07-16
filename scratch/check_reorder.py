import json

def get_request_step_experts(req_data, chunk_size=1):
    moe_profile = req_data.get("moe_profile")
    if moe_profile is None or "routing_map" not in moe_profile:
        return set()

    is_prefill = req_data.get("num_computed_tokens", 0) < req_data.get("input", req_data.get("input_toks", 0))
    num_computed_tokens = req_data.get("num_computed_tokens", 0)
    original_input = req_data.get("input", req_data.get("input_toks", 0))

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

def reorder_by_affinity(batch_req, max_num_seqs=2):
    reordered = []
    remaining = list(batch_req)
    
    while remaining:
        seed = remaining.pop(0)
        reordered.append(seed)
        
        # We can assume chunk_size is input_toks for prefill, 1 for decode
        is_prefill = seed.get("num_computed_tokens", 0) < seed.get("input_toks", 0)
        chunk_size = seed.get("input_toks", 0) if is_prefill else 1
        
        active_experts = set(get_request_step_experts(seed, chunk_size))
        
        cluster_size = 1
        while cluster_size < max_num_seqs and remaining:
            best_idx = -1
            best_overlap = -1
            
            for idx, req in enumerate(remaining):
                is_req_prefill = req.get("num_computed_tokens", 0) < req.get("input_toks", 0)
                req_chunk_size = req.get("input_toks", 0) if is_req_prefill else 1
                req_exps = get_request_step_experts(req, req_chunk_size)
                
                if not req_exps and not active_experts:
                    jaccard = 0.0
                else:
                    intersection = len(active_experts.intersection(req_exps))
                    union = len(active_experts.union(req_exps))
                    jaccard = intersection / union if union > 0 else 0
                
                if jaccard > best_overlap:
                    best_overlap = jaccard
                    best_idx = idx
            
            if best_idx != -1:
                selected = remaining.pop(best_idx)
                reordered.append(selected)
                is_sel_prefill = selected.get("num_computed_tokens", 0) < selected.get("input_toks", 0)
                sel_chunk_size = selected.get("input_toks", 0) if is_sel_prefill else 1
                active_experts.update(get_request_step_experts(selected, sel_chunk_size))
                cluster_size += 1
            else:
                break
    
    return reordered

# Load dataset
reqs = []
with open("datasets/qwen3_remote_10req_concurrent.jsonl", "r") as f:
    for line in f:
        reqs.append(json.loads(line))

print("Original order (request IDs):")
print([r["request_id"] for r in reqs])


# Check for step 0 (prefill)
for r in reqs:
    # Set mock num_computed_tokens to 0 (prefill start)
    r["num_computed_tokens"] = 0
    exps = get_request_step_experts(r, chunk_size=r["input_toks"])
    print(f"Req {r['request_id']} prefill step: {len(exps)} active experts")

reordered_reqs = reorder_by_affinity(reqs, max_num_seqs=2)
print("Reordered prefill (request IDs):")
print([r["request_id"] for r in reordered_reqs])

# Check for decode step (say num_computed_tokens = input_toks)
print("\nDecode step (step 0 decode):")
for r in reqs:
    r["num_computed_tokens"] = r["input_toks"]
    exps = get_request_step_experts(r, chunk_size=1)
    print(f"Req {r['request_id']} decode step: {len(exps)} active experts")

reordered_reqs_dec = reorder_by_affinity(reqs, max_num_seqs=2)
print("Reordered decode (request IDs):")
print([r["request_id"] for r in reordered_reqs_dec])
