import subprocess
import csv
import numpy as np

# Define simulation runs
runs = [
    {"name": "TP=1", "config": "single_node_llama4_maverick_h100_tp1.json", "output": "llama4_maverick_h100_tp1.csv"},
    {"name": "TP=2 (NVLink)", "config": "single_node_llama4_maverick_h100_tp2.json", "output": "llama4_maverick_h100_tp2_nvlink.csv"},
    {"name": "TP=2 (PCIe)", "config": "single_node_llama4_maverick_h100_tp2_pcie.json", "output": "llama4_maverick_h100_tp2_pcie.csv"},
    {"name": "TP=4 (NVLink)", "config": "single_node_llama4_maverick_h100_tp4.json", "output": "llama4_maverick_h100_tp4_nvlink.csv"},
    {"name": "TP=4 (PCIe)", "config": "single_node_llama4_maverick_h100_tp4_pcie.json", "output": "llama4_maverick_h100_tp4_pcie.csv"},
    {"name": "TP=8 (NVLink)", "config": "single_node_llama4_maverick_h100_tp8.json", "output": "llama4_maverick_h100_tp8_nvlink.csv"},
    {"name": "TP=8 (PCIe)", "config": "single_node_llama4_maverick_h100_tp8_pcie.json", "output": "llama4_maverick_h100_tp8_pcie.csv"}
]

print("Starting LLMServingSim runs for Llama-4-Maverick-17B-128E-Instruct...")

for run in runs:
    cmd = [
        "docker", "exec", "servingsim_docker",
        "python3", "-m", "serving",
        "--cluster-config", f"configs/cluster/{run['config']}",
        "--dtype", "bfloat16",
        "--block-size", "16",
        "--dataset", "datasets/llama4_10req.jsonl",
        "--output", f"outputs/{run['output']}",
        "--num-reqs", "10"
    ]
    print(f"\n--- Running simulation for {run['name']} ---")
    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

print("\n--- All simulations finished successfully! parsing outputs... ---\n")

# Format output comparison
results = []
for run in runs:
    csv_path = f"outputs/{run['output']}"
    ttfts = []
    tpots = []
    latencies = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ttfts.append(float(row['TTFT']) / 1_000_000) # ns -> ms
            tpots.append(float(row['TPOT']) / 1_000_000) # ns -> ms
            latencies.append(float(row['latency']) / 1_000_000) # ns -> ms
            
    results.append({
        "name": run['name'],
        "mean_ttft": np.mean(ttfts),
        "p99_ttft": np.percentile(ttfts, 99),
        "mean_tpot": np.mean(tpots),
        "mean_latency": np.mean(latencies)
    })

# Print beautiful markdown table
print("| Configuration | Mean TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | Mean Latency (ms) |")
print("|---|---|---|---|---|")
for r in results:
    print(f"| {r['name']} | {r['mean_ttft']:.2f} | {r['p99_ttft']:.2f} | {r['mean_tpot']:.2f} | {r['mean_latency']:.2f} |")
