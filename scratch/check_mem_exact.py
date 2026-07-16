from serving.core.memory_model import MemoryModel, Device

models = {
    "qwen3": {
        "name": "Qwen/Qwen3-235B-A22B",
        "npu_mem": 54.820,
        "num_npus": 8,
        "tp_size": 8,
        "ep_size": 8,
    },
    "deepseek": {
        "name": "deepseek/DeepSeek-R1",
        "npu_mem": 164.478,
        "num_npus": 8,
        "tp_size": 8,
        "ep_size": 8,
    },
    "llama4": {
        "name": "llama4/Llama-4-Maverick-17B-128E-Instruct",
        "npu_mem": 505.487,
        "num_npus": 4,
        "tp_size": 4,
        "ep_size": 4,
    }
}

for key, m in models.items():
    print(f"=== {key.upper()} ===")
    mem_model = MemoryModel(
        model=m["name"],
        instance_id=0,
        node_id=0,
        num_npus=m["num_npus"],
        tp_size=m["tp_size"],
        npu_mem=m["npu_mem"],
        cpu_mem=2048,
        block_size=16,
        fp=16,
        enable_prefix_caching=True,
        enable_prefix_sharing=False,
        prefix_pool=None,
        prefix_storage=None,
        ep_size=m["ep_size"],
    )
    print("Weight:", mem_model.weight)
    print("Weight (GB):", mem_model.weight / (1024**3))
    print("Available for KV (MB):", mem_model.mem_for_kv / (1024**2))
    print("KV block size (MB):", mem_model.get_kv(16) / (1024**2))
    print("Total blocks:", mem_model.mem_for_kv // mem_model.get_kv(16))
