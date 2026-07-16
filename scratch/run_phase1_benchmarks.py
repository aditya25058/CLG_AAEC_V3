#!/usr/bin/env python3
import subprocess
import re
import os
import json
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

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

def compute_metrics_from_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
        total_latency = (df["end_time"].max() - df["arrival"].min()) / 1e9
        total_prompt = df["input"].sum()
        total_gen = df["output"].sum()
        
        prompt_thru = total_prompt / total_latency if total_latency > 0 else 0.0
        gen_thru = total_gen / total_latency if total_latency > 0 else 0.0
        token_thru = (total_prompt + total_gen) / total_latency if total_latency > 0 else 0.0
        
        avg_ttft = df["TTFT"].mean() / 1e6
        median_ttft = df["TTFT"].median() / 1e6
        avg_tpot = df["TPOT"].mean() / 1e6
        median_tpot = df["TPOT"].median() / 1e6
        
        all_itls = []
        for itl_str in df["ITL"]:
            itls = json.loads(itl_str)
            all_itls.extend(itls)
        avg_itl = np.mean(all_itls) / 1e6 if all_itls else 0.0
        
        return {
            "total_latency_s": round(total_latency, 3),
            "prompt_thru_tok_s": round(prompt_thru, 2),
            "gen_thru_tok_s": round(gen_thru, 2),
            "token_thru_tok_s": round(token_thru, 2),
            "avg_ttft_ms": round(avg_ttft, 2),
            "median_ttft_ms": round(median_ttft, 2),
            "avg_tpot_ms": round(avg_tpot, 2),
            "median_tpot_ms": round(median_tpot, 2),
            "avg_itl_ms": round(avg_itl, 2),
        }
    except Exception as e:
        print(f"Error reading metrics from {csv_path}: {e}")
        return None

def run_scenario(task_info):
    model, pol_name, cmd, csv_out = task_info
    
    # Check if CSV output exists and has 51 lines (50 requests + 1 header)
    if os.path.exists(csv_out):
        try:
            with open(csv_out, "r") as f:
                lines = f.readlines()
            if len(lines) == 51:
                metrics = compute_metrics_from_csv(csv_out)
                if metrics is not None:
                    print(f"SKIPPED: {model.upper()} | {pol_name.upper()} (using existing results) -> TTFT: {metrics['avg_ttft_ms']:.2f}ms (Med: {metrics['median_ttft_ms']:.2f}ms), TPOT: {metrics['avg_tpot_ms']:.2f}ms, Latency: {metrics['total_latency_s']:.2f}s")
                    return model, pol_name, metrics
        except Exception as e:
            print(f"Could not verify existing CSV {csv_out}: {e}")

    print(f"Starting: {model.upper()} | {pol_name.upper()}...")
    env = os.environ.copy()
    env["SIM_RUN_ID"] = f"{model}_{pol_name}"
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        print(f"FAILED: {model.upper()} | {pol_name.upper()} with return code {res.returncode}")
        print(res.stderr)
        metrics = parse_metrics("")
    else:
        metrics = parse_metrics(res.stdout)
        print(f"COMPLETED: {model.upper()} | {pol_name.upper()} -> TTFT: {metrics['avg_ttft_ms']:.2f}ms (Med: {metrics['median_ttft_ms']:.2f}ms), TPOT: {metrics['avg_tpot_ms']:.2f}ms, Latency: {metrics['total_latency_s']:.2f}s")
    return model, pol_name, metrics


def main():
    venv_python = "venv/bin/python3"
    os.makedirs("outputs/phase1", exist_ok=True)

    scenarios = {
        "llama4": {
            "config": "configs/cluster/single_node_llama4_maverick_h100_tp4_calibrated_eviction.json",
            "dataset": "datasets/llama4_livecodebench_200req_concurrent.jsonl",
        },
        "deepseek": {
            "config": "configs/cluster/single_node_deepseek_r1_h100_calibrated_eviction.json",
            "dataset": "datasets/deepseek_livecodebench_200req_concurrent.jsonl",
        },
        "qwen3": {
            "config": "configs/cluster/single_node_qwen3_a22b_h100_calibrated_eviction.json",
            "dataset": "datasets/qwen3_livecodebench_200req_concurrent.jsonl",
        }
    }

    policies = {
        "fifo": [],
        "sab": ["--enable-affinity-batching"],
        "sab_aae": ["--enable-affinity-batching", "--enable-affinity-eviction"],
        "sab_cooldown": ["--enable-affinity-batching", "--enable-affinity-eviction", "--preemption-cooldown", "16"],
        "sab_thresh": ["--enable-affinity-batching", "--enable-affinity-eviction", "--affinity-threshold", "0.15"],
        "sab_hybrid": ["--enable-affinity-batching", "--enable-hybrid-sab"]
    }

    tasks = []
    for model, paths in scenarios.items():
        for pol_name, extra_args in policies.items():
            csv_out = f"outputs/phase1/{model}_{pol_name}.csv"
            cmd = [
                venv_python, "-m", "serving",
                "--cluster-config", paths["config"],
                "--dtype", "bfloat16",
                "--dataset", paths["dataset"],
                "--expert-routing-policy", "DATASET",
                "--num-reqs", "50",
                "--max-num-seqs", "8",
                "--no-enable-prefix-caching",
                "--output", csv_out
            ] + extra_args
            tasks.append((model, pol_name, cmd, csv_out))

    print(f"Submitting {len(tasks)} scenarios to run sequentially...")
    results = {m: {} for m in scenarios}
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(run_scenario, task) for task in tasks]
        for f in futures:
            model, pol_name, metrics = f.result()
            results[model][pol_name] = metrics

    with open("outputs/phase1/summary.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Phase 1 parallel sweep completed successfully. Summary saved to outputs/phase1/summary.json.")

if __name__ == "__main__":
    main()
