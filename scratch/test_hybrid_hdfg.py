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

def run_scenario(name, args):
    venv_python = "venv/bin/python3"
    cmd = [
        venv_python, "-m", "serving",
        "--cluster-config", "configs/cluster/single_node_qwen3_a22b_h100_low_mem.json",
        "--dtype", "bfloat16",
        "--dataset", "datasets/qwen3_remote_10req_concurrent.jsonl",
        "--expert-routing-policy", "DATASET",
        "--num-reqs", "10",
        "--max-num-seqs", "8",
        "--output", f"outputs/phase1/temp_diag_{name.lower()}.csv"
    ] + args
    
    print(f"Running: {name}...")
    env = os.environ.copy()
    env["SIM_RUN_ID"] = f"diag_{name.lower()}"
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        print(f"FAILED: {name}")
        print(res.stderr)
        return None
    return parse_metrics(res.stdout)

def main():
    scenarios = {
        "FIFO": [],
        "SAB": ["--enable-affinity-batching"],
        "SAB_HYBRID": ["--enable-affinity-batching", "--enable-hybrid-sab"],
        "SAB_HDFG": ["--enable-affinity-batching", "--enable-hdfg"]
    }
    
    results = {}
    for name, args in scenarios.items():
        metrics = run_scenario(name, args)
        if metrics:
            results[name] = metrics
            
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPARISON (10 Requests Concurrent Qwen3)")
    print("="*80)
    print(f"{'Policy':<15} | {'Makespan (s)':<12} | {'Avg TTFT (ms)':<14} | {'Med TTFT (ms)':<14} | {'Avg TPOT (ms)':<14}")
    print("-"*80)
    for name, m in results.items():
        print(f"{name:<15} | {m['total_latency_s']:<12.3f} | {m['avg_ttft_ms']:<14.2f} | {m['median_ttft_ms']:<14.2f} | {m['avg_tpot_ms']:<14.2f}")
    print("="*80)

if __name__ == "__main__":
    main()
