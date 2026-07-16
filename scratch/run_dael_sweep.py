#!/usr/bin/env python3
import subprocess
import re
import os
import json
import csv
import math

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

def parse_dael_metrics(stdout):
    covs = []
    max_to_means = []
    redirected = 0
    
    matches = re.findall(r"\[DAEL_METRICS\] layer=\d+ cov=([\d.]+) max_to_mean=([\d.]+) redirected=(\d+)", stdout)
    for cov_str, mtm_str, redir_str in matches:
        covs.append(float(cov_str))
        max_to_means.append(float(mtm_str))
        redirected += int(redir_str)
        
    avg_cov = sum(covs) / len(covs) if covs else 0.0
    avg_max_to_mean = sum(max_to_means) / len(max_to_means) if max_to_means else 1.0
    routing_overhead_ms = (redirected * 50) / 1000000.0  # 50 ns per redirected token -> ms
    
    return {
        "avg_link_saturation_cov": avg_cov,
        "avg_expert_queue_max_to_mean": avg_max_to_mean,
        "total_redirected_tokens": redirected,
        "routing_overhead_ms": routing_overhead_ms
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

def make_temp_config(original_path, temp_path, new_bw):
    with open(original_path, 'r') as f:
        data = json.load(f)
    data["link_bw"] = new_bw
    with open(temp_path, 'w') as f:
        json.dump(data, f, indent=4)

def parse_csv_results(csv_path):
    if not os.path.exists(csv_path):
        return []
    records = []
    with open(csv_path, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
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

def main():
    venv_python = "venv/bin/python3"
    os.makedirs("outputs/phase4", exist_ok=True)
    
    # We will test both models: Qwen3-235B and DeepSeek-R1
    models = {
        "qwen3": {
            "name": "Qwen3-235B",
            "config": "configs/cluster/single_node_qwen3_a22b_h100_tp8.json",
            "datasets": {
                "low_load": ("datasets/qwen3_remote_10req_concurrent.jsonl", 10),
                "high_load": ("datasets/qwen3_remote_30req_concurrent.jsonl", 30)
            }
        },
        "deepseek": {
            "name": "DeepSeek-R1",
            "config": "configs/cluster/single_node_deepseek_r1_h100_tp8.json",
            "datasets": {
                "low_load": ("datasets/deepseek_remote_10req_concurrent.jsonl", 10),
                "high_load": ("datasets/deepseek_remote_30req_concurrent.jsonl", 30)
            }
        }
    }
    
    # Sweep settings
    bandwidths = [2.0, 16.0, 32.0]  # GB/s link bandwidths
    
    variants = [
        {
            "id": "baseline",
            "name": "Baseline (Uniform BF16)",
            "flags": []
        },
        {
            "id": "epeg",
            "name": "EPEG",
            "flags": ["--enable-epeg", "--epeg-tau-high", "0.40", "--epeg-tau-low", "0.05"]
        },
        {
            "id": "dael",
            "name": "DAEL",
            "flags": ["--enable-dael", "--dael-saturation-threshold", "0.80", "--dael-redirect-fraction", "0.30"]
        },
        {
            "id": "epeg_dael",
            "name": "EPEG+DAEL",
            "flags": ["--enable-epeg", "--epeg-tau-high", "0.40", "--epeg-tau-low", "0.05",
                      "--enable-dael", "--dael-saturation-threshold", "0.80", "--dael-redirect-fraction", "0.30"]
        }
    ]
    
    results = {}
    
    for model_id, model_info in models.items():
        results[model_id] = {}
        print(f"\n======================================================================")
        print(f"Running Sweeps for Model: {model_info['name']}")
        print(f"======================================================================")
        
        for bw in bandwidths:
            bw_str = f"{int(bw)}GBs"
            results[model_id][bw_str] = {}
            
            # Create a temporary config file with the specific link bandwidth
            temp_config = f"configs/cluster/temp_dael_bw_{model_id}_{bw_str}.json"
            make_temp_config(model_info["config"], temp_config, bw)
            
            for load_level, (dataset_path, num_reqs) in model_info["datasets"].items():
                results[model_id][bw_str][load_level] = {}
                print(f"\n--- Bandwidth: {bw} GB/s | Load: {load_level} ({num_reqs} reqs) ---")
                
                for var in variants:
                    var_id = var["id"]
                    csv_out = f"outputs/phase4/dael_{model_id}_{bw_str}_{load_level}_{var_id}.csv"
                    
                    cmd = [
                        venv_python, "-m", "serving",
                        "--cluster-config", temp_config,
                        "--dataset", dataset_path,
                        "--num-reqs", str(num_reqs),
                        "--expert-routing-policy", "DATASET",
                        "--gpus-per-node", "8",
                        "--output", csv_out
                    ] + var["flags"]
                    
                    stdout = run_cmd(cmd)
                    metrics = parse_metrics(stdout)
                    dael_metrics = parse_dael_metrics(stdout)
                    
                    if metrics["total_latency_s"] == 0.0:
                        print(f"ERROR: Simulation failed for {model_id} / {bw_str} / {load_level} / {var_id}")
                        continue
                        
                    records = parse_csv_results(csv_out)
                    q_delays = [r["queuing_delay_ms"] for r in records]
                    mean_q = sum(q_delays) / len(q_delays) if q_delays else 0.0
                    max_q = max(q_delays) if q_delays else 0.0
                    
                    combined = {
                        **metrics,
                        **dael_metrics,
                        "mean_queuing_delay_ms": mean_q,
                        "max_queuing_delay_ms": max_q
                    }
                    results[model_id][bw_str][load_level][var_id] = combined
                    
                    print(f"-> {var['name']}:")
                    print(f"   Latency: {combined['total_latency_s']:.3f}s | Mean TTFT: {combined['avg_ttft_ms']:.2f}ms | Mean TPOT: {combined['avg_tpot_ms']:.2f}ms")
                    print(f"   Link Saturation CoV: {combined['avg_link_saturation_cov']:.4f} | Expert Queue Max/Mean: {combined['avg_expert_queue_max_to_mean']:.4f}")
                    print(f"   Redirected Tokens: {combined['total_redirected_tokens']} | Routing Overhead: {combined['routing_overhead_ms']:.4f}ms")
            
            # Clean up temporary config file
            if os.path.exists(temp_config):
                os.remove(temp_config)
                
    # Save final results
    results_path = "outputs/phase4/dael_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\n======================================================================")
    print(f"Successfully completed DAEL sweeps! Wrote results to {results_path}")
    print(f"======================================================================")

if __name__ == "__main__":
    main()
