#!/usr/bin/env python3
import subprocess
import re
import os
import json

def parse_metrics(stdout):
    try:
        total_latency = float(re.search(r"Total latency \(s\):\s*([\d.]+)", stdout).group(1))
    except (AttributeError, ValueError):
        total_latency = 0.0

    try:
        prompt_thru = float(re.search(r"Average prompt throughput \(tok/s\):\s*([\d.]+)", stdout).group(1))
    except (AttributeError, ValueError):
        prompt_thru = 0.0

    try:
        gen_thru = float(re.search(r"Average generation throughput \(tok/s\):\s*([\d.]+)", stdout).group(1))
    except (AttributeError, ValueError):
        gen_thru = 0.0

    try:
        token_thru = float(re.search(r"Total token throughput \(tok/s\):\s*([\d.]+)", stdout).group(1))
    except (AttributeError, ValueError):
        token_thru = 0.0

    ttfts = [float(x) for x in re.findall(r"Mean TTFT \(ms\):\s*([\d.]+)", stdout)]
    tpots = [float(x) for x in re.findall(r"Mean TPOT \(ms\):\s*([\d.]+)", stdout)]
    itls = [float(x) for x in re.findall(r"Mean ITL \(ms\)\s*:\s*([\d.]+)", stdout)]

    avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0
    avg_tpot = sum(tpots) / len(tpots) if tpots else 0.0
    avg_itl = sum(itls) / len(itls) if itls else 0.0

    try:
        median_ttft = float(re.search(r"Median TTFT \(ms\):\s*([\d.]+)", stdout).group(1))
    except (AttributeError, ValueError):
        median_ttft = avg_ttft

    try:
        median_tpot = float(re.search(r"Median TPOT \(ms\):\s*([\d.]+)", stdout).group(1))
    except (AttributeError, ValueError):
        median_tpot = avg_tpot

    return {
        "total_latency_s": total_latency,
        "prompt_thru_tok_s": prompt_thru,
        "gen_thru_tok_s": gen_thru,
        "token_thru_tok_s": token_thru,
        "avg_ttft_ms": avg_ttft,
        "median_ttft_ms": median_ttft,
        "avg_tpot_ms": avg_tpot,
        "median_tpot_ms": median_tpot,
        "avg_itl_ms": avg_itl,
    }

def run_cmd(cmd, env=None):
    print(f"Running command: {' '.join(cmd)}")
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    res = subprocess.run(cmd, capture_output=True, text=True, env=run_env)
    if res.returncode != 0:
        print(f"Command failed with code {res.returncode}")
        print(res.stderr)
        return ""
    return res.stdout

def main():
    venv_python = "venv/bin/python3"
    os.makedirs("outputs/phase3", exist_ok=True)
    os.makedirs("configs/cluster", exist_ok=True)
    
    dataset_path = "datasets/qwen3_remote_10req_concurrent.jsonl"
    base_config_path = "configs/cluster/test_dual_node_tp2_ep4.json"
    
    with open(base_config_path, "r") as f:
        base_config = json.load(f)
        
    bandwidths = [1.0, 4.0, 16.0, 32.0]
    epeg_options = [False, True] # EPEG Disabled vs Enabled
    
    results = {str(bw): {} for bw in bandwidths}
    
    for bw in bandwidths:
        # Create temporary cluster config with adjusted link bandwidth
        temp_config = base_config.copy()
        temp_config["link_bw"] = bw
        temp_config_path = f"configs/cluster/temp_epeg_multinode_bw_{str(bw).replace('.', '_')}.json"
        with open(temp_config_path, "w") as f:
            json.dump(temp_config, f, indent=4)
            
        for enabled in epeg_options:
            enabled_str = "enabled" if enabled else "disabled"
            csv_out = f"outputs/phase3/epeg_multinode_{enabled_str}_bw_{str(bw).replace('.', '_')}.csv"
            
            cmd = [
                venv_python, "-m", "serving",
                "--cluster-config", temp_config_path,
                "--dataset", dataset_path,
                "--num-reqs", "2",
                "--expert-routing-policy", "DATASET",
                "--output", csv_out
            ]
            if enabled:
                cmd.append("--enable-epeg")
                
            stdout = run_cmd(cmd)
            metrics = parse_metrics(stdout)
            results[str(bw)][str(enabled)] = metrics
            print(f"Multi-Node EPEG Sweep: BW={bw} GB/s | EPEG={enabled_str.upper()} -> Latency: {metrics['total_latency_s']}s, TTFT: {metrics['avg_ttft_ms']}ms, TPOT: {metrics['avg_tpot_ms']}ms")
            
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

    # Save results to JSON
    with open("outputs/phase3/epeg_multinode_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("Successfully completed Multi-Node EPEG sweeps and wrote outputs/phase3/epeg_multinode_results.json")

if __name__ == "__main__":
    main()
