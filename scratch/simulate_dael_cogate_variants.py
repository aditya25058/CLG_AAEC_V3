#!/usr/bin/env python3
"""Run a skew=0.7 workload under 4 configurations (DAEL, DAEL+INDWM, Co-Gate, Co-Gate+INDWM) under two workload scales and output detailed redirection distributions."""
import subprocess
import re
from collections import Counter

def run_simulation(args, dataset, num_reqs, redirect_fraction):
    cmd = [
        "venv/bin/python3", "-m", "serving",
        "--cluster-config", "configs/cluster/test_dual_node_tp2_ep4.json",
        "--dataset", dataset,
        "--num-reqs", str(num_reqs),
        "--gpus-per-node", "2",
        "--expert-routing-policy", "BALANCED",
        "--expert-skew-intensity", "0.7",
        "--dael-saturation-threshold", "0.15",
        "--dael-redirect-fraction", str(redirect_fraction)
    ] + args
    
    print(f"Running command: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    stdout = res.stdout
    stderr = res.stderr
    
    # Parse latency
    try:
        latency = float(re.search(r"Total latency \(s\):\s*([\d.]+)", stdout).group(1))
    except:
        latency = 0.0
        print(f"Failed to parse latency! stderr: {stderr}")
        
    # Parse all layer 0 metrics to build a redirection count histogram
    all_dael_lines = re.findall(r"\[DAEL_METRICS\] layer=0 cov=([\d.]+) max_to_mean=([\d.]+) redirected=(\d+) local_redirects=(\d+) remote_redirects=(\d+)", stdout)
    
    redir_counts = []
    covs = []
    max_to_means = []
    
    for cov_str, m2m_str, redir_str, loc_str, rem_str in all_dael_lines:
        redir_counts.append(int(redir_str))
        covs.append(float(cov_str))
        max_to_means.append(float(m2m_str))
        
    redir_counter = Counter(redir_counts)
    
    # Representative/first metrics for summary table
    cov = covs[0] if covs else 0.0
    max_to_mean = max_to_means[0] if max_to_means else 1.0
    
    return {
        "latency": latency,
        "cov": cov,
        "max_to_mean": max_to_mean,
        "redir_counter": redir_counter,
        "total_steps": len(redir_counts)
    }

def print_table(results):
    print(f"{'Configuration':<22} | {'Latency (s)':<12} | {'First CoV':<10} | {'First M/M':<10} | {'Redirection Distribution (Count: Steps)'}")
    print("-"*110)
    for name, res in results.items():
        dist_str = ", ".join(f"{k} tok: {v} steps" for k, v in sorted(res['redir_counter'].items()))
        print(f"{name:<22} | {res['latency']:<12.4f} | {res['cov']:<10.4f} | {res['max_to_mean']:<10.4f} | {dist_str}")

def main():
    print("="*80)
    print("SIMULATING DAEL vs. CO-GATE VARIANTS WITH DETAILED REDIRECTION DISTRIBUTION")
    print("="*80)
    
    dataset_2 = "datasets/qwen3_remote_30req_concurrent.jsonl"
    num_reqs_2 = 30
    frac_2 = 0.30
    
    print("\n[1/4] Running DAEL Alone...")
    dael_alone_2 = run_simulation(["--enable-dael"], dataset_2, num_reqs_2, frac_2)
    
    print("\n[2/4] Running DAEL + INDWM...")
    dael_indwm_2 = run_simulation(["--enable-dael", "--enable-indwm"], dataset_2, num_reqs_2, frac_2)
    
    print("\n[3/4] Running Co-Gate...")
    cogate_2 = run_simulation(["--enable-cogate"], dataset_2, num_reqs_2, frac_2)
    
    print("\n[4/4] Running Co-Gate + INDWM...")
    cogate_indwm_2 = run_simulation(["--enable-cogate", "--enable-indwm"], dataset_2, num_reqs_2, frac_2)
    
    print("\n" + "="*110)
    print("EXPERIMENT RESULTS (High Workload - 30 Requests)")
    print("="*110)
    print_table({
        "DAEL Alone": dael_alone_2,
        "DAEL + INDWM": dael_indwm_2,
        "Co-Gate": cogate_2,
        "Co-Gate + INDWM": cogate_indwm_2
    })
    print("="*110)
    
    print("\nDETAILED ANALYSIS:")
    print("The simulator computes steering overhead at each layer for a step:")
    print("1. Without INDWM: charge = redirected_tokens * 28 us (activation transfer over RDMA).")
    print("2. With INDWM: charge = 222.22 us (flat NVLink weight streaming) + remote_tokens * 28 us.")
    print("The crossover point is exactly: 222.22 us / 28 us = 7.93 tokens.")
    print("- Steps with < 8 redirected tokens are faster WITHOUT INDWM.")
    print("- Steps with >= 8 redirected tokens are faster WITH INDWM.")
    
    print("\nLet's check the step redirection counts:")
    for name, res in [("DAEL Alone", dael_alone_2), ("Co-Gate", cogate_2)]:
        total_below = sum(v for k, v in res['redir_counter'].items() if k < 8)
        total_above = sum(v for k, v in res['redir_counter'].items() if k >= 8)
        pct_below = (total_below / res['total_steps']) * 100 if res['total_steps'] > 0 else 0.0
        print(f"\n{name}:")
        print(f"  - Total evaluated steps: {res['total_steps']}")
        print(f"  - Steps below crossover (<8 tokens): {total_below} ({pct_below:.1f}%)")
        print(f"  - Steps above crossover (>=8 tokens): {total_above} ({100 - pct_below:.1f}%)")

if __name__ == "__main__":
    main()
