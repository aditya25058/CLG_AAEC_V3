import subprocess
import os

def main():
    venv_python = "venv/bin/python3"
    # We will run the serving module directly with python but redirect output and parse it
    cmd = [
        venv_python, "-m", "serving",
        "--cluster-config", "configs/cluster/single_node_qwen3_a22b_h100_low_mem.json",
        "--dtype", "bfloat16",
        "--dataset", "datasets/qwen3_remote_10req_concurrent.jsonl",
        "--expert-routing-policy", "DATASET",
        "--num-reqs", "10",
        "--max-num-seqs", "8",
        "--enable-affinity-batching",
        "--enable-affinity-eviction",
        "--output", "outputs/phase1/temp_diag.csv",
        "--log-level", "INFO"
    ]
    
    print("Running simulator with stdout trace...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    
    # Let's inspect the first 150 lines
    lines = res.stdout.split("\n")
    print("\n=== FIRST 150 LINES OF SIMULATOR OUTPUT ===")
    for line in lines[:150]:
        print(line)

if __name__ == "__main__":
    main()
