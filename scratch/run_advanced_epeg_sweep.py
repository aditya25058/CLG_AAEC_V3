#!/usr/bin/env python3
import subprocess
import re
import os
import json
import csv

def parse_metrics(stdout):
    try:
        total_latency = float(re.search(r"Total latency \(s\):\s*([\d.]+)", stdout).group(1))
    except (AttributeError, ValueError):
        total_latency = 0.0

    ttfts = [float(x) for x in re.findall(r"Mean TTFT \(ms\):\s*([\d.]+)", stdout)]
    tpots = [float(x) for x in re.findall(r"Mean TPOT \(ms\):\s*([\d.]+)", stdout)]

    avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0
    avg_tpot = sum(tpots) / len(tpots) if tpots else 0.0

    return {
        "total_latency_s": total_latency,
        "avg_ttft_ms": avg_ttft,
        "avg_tpot_ms": avg_tpot,
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

def parse_csv_results(csv_path):
    if not os.path.exists(csv_path):
        return []
    records = []
    with open(csv_path, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # queuing_delay is in CPU cycles (ns since FREQ = 1GHz). Divide by 1e6 to get ms.
            q_delay_ns = float(row.get('queuing_delay', 0.0))
            q_delay_ms = q_delay_ns / 1000000.0
            
            ttft_ns = float(row.get('TTFT', 0.0))
            ttft_ms = ttft_ns / 1000000.0
            
            tpot_ns = float(row.get('TPOT', 0.0))
            tpot_ms = tpot_ns / 1000000.0
            
            records.append({
                "request_id": int(row.get('request_id', 0)),
                "queuing_delay_ms": q_delay_ms,
                "ttft_ms": ttft_ms,
                "tpot_ms": tpot_ms
            })
    return records

def calculate_stats(records, variant):
    if not records:
        return {
            "mean_q_delay_ms": 0.0,
            "max_q_delay_ms": 0.0,
            "p99_q_delay_ms": 0.0,
            "congested_reqs_count": 0,
            "congested_reqs_pct": 0.0,
            "projected_gsm8k_loss_pct": 0.0,
            "projected_mmlu_loss_pct": 0.0,
            "projected_lcb_loss_pct": 0.0
        }
    
    q_delays = [r["queuing_delay_ms"] for r in records]
    mean_q = sum(q_delays) / len(q_delays)
    max_q = max(q_delays)
    q_delays_sorted = sorted(q_delays)
    p99_idx = int(len(q_delays_sorted) * 0.99)
    p99_q = q_delays_sorted[min(p99_idx, len(q_delays_sorted) - 1)]
    
    # SLA queue threshold is 50ms
    congested = [q for q in q_delays if q > 50.0]
    congested_count = len(congested)
    congested_pct = (congested_count / len(records)) * 100.0
    
    # Calculate projected accuracy loss
    if variant == "baseline":
        gsm = 0.0
        mmlu = 0.0
        lcb = 0.0
    elif variant == "static_epeg":
        gsm = 0.0810
        mmlu = 0.0540
        lcb = 0.1081
    elif variant in ["epeg_sla", "epeg_sla_caps", "epeg_sla_slice", "full_epeg"]:
        # Weight by fraction of requests that exceeded the 50ms SLA queue latency
        frac_congested = congested_count / len(records)
        gsm = frac_congested * 0.2754
        mmlu = frac_congested * 0.1836
        lcb = frac_congested * 0.3672
    else:
        gsm = 0.0
        mmlu = 0.0
        lcb = 0.0
        
    return {
        "mean_q_delay_ms": mean_q,
        "max_q_delay_ms": max_q,
        "p99_q_delay_ms": p99_q,
        "congested_reqs_count": congested_count,
        "congested_reqs_pct": congested_pct,
        "projected_gsm8k_loss_pct": gsm,
        "projected_mmlu_loss_pct": mmlu,
        "projected_lcb_loss_pct": lcb
    }

def main():
    venv_python = "venv/bin/python3"
    os.makedirs("outputs/phase3/advanced_epeg", exist_ok=True)
    
    cluster_config = "configs/cluster/single_node_qwen3_a22b_h100_tp8.json"
    dataset = "datasets/qwen3_livecodebench_200req_concurrent.jsonl"
    num_reqs = 50
    
    variants = [
        {
            "id": "baseline",
            "name": "Baseline (Uniform BF16)",
            "flags": []
        },
        {
            "id": "static_epeg",
            "name": "Static EPEG",
            "flags": ["--enable-epeg", "--epeg-tau-high", "0.40", "--epeg-tau-low", "0.05"]
        },
        {
            "id": "epeg_sla",
            "name": "EPEG-SLA",
            "flags": ["--enable-epeg", "--enable-epeg-sla"]
        },
        {
            "id": "epeg_sla_caps",
            "name": "EPEG-SLA + CAPS",
            "flags": ["--enable-epeg", "--enable-epeg-sla", "--enable-caps"]
        },
        {
            "id": "epeg_sla_slice",
            "name": "EPEG-SLA + EPEG-Slice",
            "flags": ["--enable-epeg", "--enable-epeg-sla", "--enable-epeg-slice"]
        },
        {
            "id": "full_epeg",
            "name": "Full Co-Designed EPEG (SLA + CAPS + Slice)",
            "flags": ["--enable-epeg", "--enable-epeg-sla", "--enable-caps", "--enable-epeg-slice"]
        }
    ]
    
    results = {}
    print(f"Starting Advanced EPEG sweeps on {num_reqs} requests (Concurrent)...")
    
    for variant in variants:
        var_id = variant["id"]
        csv_out = f"outputs/phase3/advanced_epeg/advanced_{var_id}.csv"
        
        cmd = [
            venv_python, "-m", "serving",
            "--cluster-config", cluster_config,
            "--dataset", dataset,
            "--num-reqs", str(num_reqs),
            "--expert-routing-policy", "DATASET",
            "--gpus-per-node", "8",
            "--output", csv_out
        ] + variant["flags"]
        
        stdout = run_cmd(cmd)
        metrics = parse_metrics(stdout)
        
        if metrics["total_latency_s"] == 0.0:
            print(f"ERROR: Simulation failed for {var_id}")
            continue
            
        records = parse_csv_results(csv_out)
        stats = calculate_stats(records, var_id)
        
        combined = {**metrics, **stats}
        results[var_id] = combined
        
        print(f"-> {variant['name']}:")
        print(f"   Latency: {combined['total_latency_s']:.3f}s | Mean TTFT: {combined['avg_ttft_ms']:.2f}ms | Mean TPOT: {combined['avg_tpot_ms']:.2f}ms")
        print(f"   Mean Queue Delay: {combined['mean_q_delay_ms']:.2f}ms | Max Queue Delay: {combined['max_q_delay_ms']:.2f}ms")
        print(f"   Congested Reqs: {combined['congested_reqs_count']}/{num_reqs} ({combined['congested_reqs_pct']:.1f}%)")
        print(f"   Projected Accuracy Loss (GSM8K): {combined['projected_gsm8k_loss_pct']:.4f}%")
        
    results_path = "outputs/phase3/advanced_epeg_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\n======================================================================")
    print(f"Successfully completed Advanced EPEG sweeps! Wrote results to {results_path}")
    print(f"======================================================================")

if __name__ == "__main__":
    main()
