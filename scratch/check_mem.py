from serving.core.memory_model import get_config, full_cluster_kv_bytes_per_token, calculate_sizes

model_name = "Qwen/Qwen3-235B-A22B"
config = get_config(model_name)
print("Config:", config)

# Let's compute weight manually or via MemoryModel
from serving.core.memory_model import MemoryModel

# Let's instantiate a MemoryModel with dummy parameters matching configs/cluster/single_node_qwen3_a22b_h100_low_mem.json
# Memory: 54.825 GB, TP=8, EP=8, PP=1, num_npus=8
# Removed to avoid exception

# Let's inspect get_weight components
tp = 8
ep = 8
fp = 2
dense_block = 0
_, qkv_w, _ = calculate_sizes(model_name, 'qkv_proj', 1, parallel=tp, fp=fp, tp_size=tp)
_, o_w, _ = calculate_sizes(model_name, 'o_proj', 1, parallel=tp, fp=fp, tp_size=tp)
_, moe_w, _ = calculate_sizes(model_name, 'moe', 1, parallel=ep, fp=fp, tp_size=tp, replicated_experts=0)
_, embedding_w, _ = calculate_sizes(model_name, 'embedding', 1, parallel=tp, fp=fp, tp_size=tp)
_, sampler_w, _ = calculate_sizes(model_name, 'sampler', 1, parallel=tp, fp=fp, tp_size=tp)
_, lm_head_w, _ = calculate_sizes(model_name, 'lm_head', 1, parallel=tp, fp=fp, tp_size=tp)
_, ln_f_w, _ = calculate_sizes(model_name, 'final_layernorm', 1, parallel=tp, fp=fp, tp_size=tp)

print("embedding (MB):", embedding_w / (1024**2))
print("lm_head (MB):", lm_head_w / (1024**2))
print("final layernorm (MB):", ln_f_w / (1024**2))
print("dense layers per block (MB):", (qkv_w + o_w) / (1024**2))
print("moe layer per block (MB):", moe_w / (1024**2))
total_w = embedding_w + lm_head_w + ln_f_w + (qkv_w + o_w + moe_w) * 94
print("Total Weight per NPU (GB):", total_w / (1024**3))
print("Remaining KV cache capacity at 54.825 GB (MB):", (54.825 * 1024**3 - total_w) / (1024**2))
print("Remaining KV cache capacity at 54.820 GB (MB):", (54.820 * 1024**3 - total_w) / (1024**2))
print("Remaining KV cache capacity at 54.819 GB (MB):", (54.819 * 1024**3 - total_w) / (1024**2))


