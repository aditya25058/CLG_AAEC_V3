import subprocess

runs = [
    # Kimi-K2 (TP=8 is excluded/exception)
    {"model": "kimi", "name": "Kimi TP=1", "config": "single_node_kimi_k2_h100_tp1.json", "output": "kimi_k2_h100_tp1.csv", "dataset": "datasets/kimi_k2_10req.jsonl"},
    {"model": "kimi", "name": "Kimi TP=2 (NVLink)", "config": "single_node_kimi_k2_h100_tp2.json", "output": "kimi_k2_h100_tp2_nvlink.csv", "dataset": "datasets/kimi_k2_10req.jsonl"},
    {"model": "kimi", "name": "Kimi TP=2 (PCIe)", "config": "single_node_kimi_k2_h100_tp2_pcie.json", "output": "kimi_k2_h100_tp2_pcie.csv", "dataset": "datasets/kimi_k2_10req.jsonl"},
    {"model": "kimi", "name": "Kimi TP=4 (NVLink)", "config": "single_node_kimi_k2_h100_tp4.json", "output": "kimi_k2_h100_tp4_nvlink.csv", "dataset": "datasets/kimi_k2_10req.jsonl"},
    {"model": "kimi", "name": "Kimi TP=4 (PCIe)", "config": "single_node_kimi_k2_h100_tp4_pcie.json", "output": "kimi_k2_h100_tp4_pcie.csv", "dataset": "datasets/kimi_k2_10req.jsonl"}
]

print("Starting remaining ServingSim runs for Kimi-K2...")

for run in runs:
    cmd = [
        "docker", "exec", "servingsim_docker",
        "python3", "-m", "serving",
        "--cluster-config", f"configs/cluster/{run['config']}",
        "--dtype", "bfloat16",
        "--block-size", "16",
        "--dataset", run["dataset"],
        "--output", f"outputs/{run['output']}",
        "--num-reqs", "10"
    ]
    print(f"\n--- Running simulation for {run['name']} ---")
    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

print("\n--- All Kimi-K2 simulations completed successfully! ---")
