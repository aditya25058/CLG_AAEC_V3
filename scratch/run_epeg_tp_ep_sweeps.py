#!/usr/bin/env python3
import subprocess
import re
import os
import json
import math

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

    try:
        total_prompt = int(re.search(r"Total input tokens:\s*(\d+)", stdout).group(1))
    except (AttributeError, ValueError):
        total_prompt = 0

    try:
        total_gen = int(re.search(r"Total generated tokens:\s*(\d+)", stdout).group(1))
    except (AttributeError, ValueError):
        total_gen = 0

    ttfts = [float(x) for x in re.findall(r"Mean TTFT \(ms\):\s*([\d.]+)", stdout)]
    tpots = [float(x) for x in re.findall(r"Mean TPOT \(ms\):\s*([\d.]+)", stdout)]

    avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0
    avg_tpot = sum(tpots) / len(tpots) if tpots else 0.0

    return {
        "total_latency_s": total_latency,
        "prompt_thru_tok_s": prompt_thru,
        "gen_thru_tok_s": gen_thru,
        "token_thru_tok_s": token_thru,
        "avg_ttft_ms": avg_ttft,
        "avg_tpot_ms": avg_tpot,
        "total_prompt_tokens": total_prompt,
        "total_gen_tokens": total_gen
    }

def run_cmd(cmd, env_vars=None):
    print(f"Running command: {' '.join(cmd)}")
    run_env = os.environ.copy()
    if env_vars:
        run_env.update(env_vars)
    res = subprocess.run(cmd, capture_output=True, text=True, env=run_env)
    if res.returncode != 0:
        print(f"Command failed with code {res.returncode}")
        print(res.stderr)
        return ""
    return res.stdout

def create_single_node_config(tp, ep):
    config = {
        "num_nodes": 1,
        "link_bw": 900,
        "link_latency": 1500,
        "nodes": [
            {
                "num_instances": 1,
                "cpu_mem": {
                    "mem_size": 2048,
                    "mem_bw": 256,
                    "mem_latency": 0
                },
                "instances": [
                    {
                        "model_name": "Qwen/Qwen3-235B-A22B",
                        "hardware": "H100",
                        "npu_mem": {
                            "mem_size": 2000,
                            "mem_bw": 1597,
                            "mem_latency": 0
                        },
                        "num_npus": tp,
                        "tp_size": tp,
                        "ep_size": ep,
                        "pd_type": None
                    }
                ]
            }
        ]
    }
    path = f"configs/cluster/temp_single_node_tp{tp}_ep{ep}.json"
    with open(path, "w") as f:
        json.dump(config, f, indent=4)
    return path

def create_multi_node_config(tp, ep, bw=1.0):
    config = {
        "num_nodes": 2,
        "link_bw": bw,
        "link_latency": 20000,
        "nodes": [
            {
                "num_instances": 1,
                "cpu_mem": {
                    "mem_size": 512,
                    "mem_bw": 256,
                    "mem_latency": 0
                },
                "instances": [
                    {
                        "model_name": "Qwen/Qwen3-235B-A22B",
                        "hardware": "H100",
                        "npu_mem": {
                            "mem_size": 1000,
                            "mem_bw": 1597,
                            "mem_latency": 0
                        },
                        "num_npus": tp,
                        "tp_size": tp,
                        "ep_size": ep,
                        "dp_group": "A",
                        "pd_type": None
                    }
                ]
            },
            {
                "num_instances": 1,
                "cpu_mem": {
                    "mem_size": 512,
                    "mem_bw": 256,
                    "mem_latency": 0
                },
                "instances": [
                    {
                        "model_name": "Qwen/Qwen3-235B-A22B",
                        "hardware": "H100",
                        "npu_mem": {
                            "mem_size": 1000,
                            "mem_bw": 1597,
                            "mem_latency": 0
                        },
                        "num_npus": tp,
                        "tp_size": tp,
                        "ep_size": ep,
                        "dp_group": "A",
                        "pd_type": None
                    }
                ]
            }
        ]
    }
    path = f"configs/cluster/temp_multi_node_tp{tp}_ep{ep}_bw_{str(bw).replace('.', '_')}.json"
    with open(path, "w") as f:
        json.dump(config, f, indent=4)
    return path

def calculate_a2a_bytes(ep, total_tokens, enable_epeg, exclude_comm, k=8):
    if ep <= 1:
        return 0.0, 0.0
    
    # Model details
    n_embd = 4096
    num_experts = 128
    num_hidden_layers = 94
    fp_size = 2 # BF16
    
    dispatch_per_token = (n_embd + num_experts) * fp_size
    combine_per_token = n_embd * fp_size
    base_per_token = dispatch_per_token + combine_per_token
    
    # EPEG scaling
    precision_scale = 1.0
    if enable_epeg and not exclude_comm:
        # Calculate precision_scale analytically
        alpha = 0.5
        weights = [math.exp(-alpha * idx) for idx in range(1, k + 1)]
        s = sum(weights)
        gate_weights = [w / s for w in weights]
        ps = 0.0
        for w in gate_weights:
            if w >= 0.40:
                ps += 1.0
            elif w >= 0.05:
                ps += 0.5
            else:
                ps += 0.25
        precision_scale = ps / k
        
    bytes_per_token = base_per_token * num_hidden_layers * precision_scale
    total_bytes = bytes_per_token * total_tokens
    
    return total_bytes, bytes_per_token

def main():
    venv_python = "venv/bin/python3"
    os.makedirs("outputs/phase3", exist_ok=True)
    os.makedirs("configs/cluster", exist_ok=True)
    
    default_dataset = "datasets/qwen3_remote_10req_concurrent.jsonl"
    concurrency_dataset = "datasets/qwen3_livecodebench_200req_concurrent.jsonl"
    
    # Result maps
    ablation_results = {}
    ep_tp_results = {}
    bw_sensitivity_results = {}
    topk_results = {}
    concurrency_results = {}
    
    # ----------------------------------------------------
    # Helper executor
    # ----------------------------------------------------
    def run_sim(tp, ep, is_multi, variant, bw=1.0, k=8, dataset=default_dataset, num_reqs=10):
        # Create config
        if is_multi:
            config_path = create_multi_node_config(tp, ep, bw)
        else:
            config_path = create_single_node_config(tp, ep)
            
        cmd = [
            venv_python, "-m", "serving",
            "--cluster-config", config_path,
            "--dataset", dataset,
            "--num-reqs", str(num_reqs),
            "--expert-routing-policy", "DATASET",
            "--gpus-per-node", str(tp),
            "--output", f"outputs/phase3/temp_sweep_out.csv"
        ]
        
        env_vars = {}
        if k != 8:
            env_vars["OVERRIDE_TOPK"] = str(k)
            
        enable_epeg = False
        exclude_comm = False
        exclude_compute = False
        
        if variant == "Comm Only":
            cmd.extend(["--enable-epeg", "--epeg-exclude-compute"])
            enable_epeg = True
            exclude_compute = True
        elif variant == "Compute Only":
            cmd.extend(["--enable-epeg", "--epeg-exclude-comm"])
            enable_epeg = True
            exclude_comm = True
        elif variant == "Full EPEG":
            cmd.append("--enable-epeg")
            enable_epeg = True
            
        stdout = run_cmd(cmd, env_vars)
        metrics = parse_metrics(stdout)
        
        # Calculate A2A metrics
        total_tokens = metrics["total_prompt_tokens"] + metrics["total_gen_tokens"]
        total_a2a_bytes, a2a_bytes_per_token = calculate_a2a_bytes(ep, total_tokens, enable_epeg, exclude_comm, k)
        
        metrics["total_a2a_bytes"] = total_a2a_bytes
        metrics["a2a_bytes_per_token"] = a2a_bytes_per_token
        metrics["a2a_bytes_per_gen_token"] = total_a2a_bytes / max(1, metrics["total_gen_tokens"])
        
        # Cleanup config
        if os.path.exists(config_path):
            os.remove(config_path)
            
        return metrics

    # ====================================================
    # 1. Ablation Sweep
    # ====================================================
    print("\n=== RUNNING SWEEP 1: EPEG Ablations ===")
    ablation_configs = [
        # (tp, ep, is_multi)
        (8, 8, False),
        (4, 4, False),
        (2, 4, True),
        (4, 8, True)
    ]
    variants = ["Baseline", "Comm Only", "Compute Only", "Full EPEG"]
    
    for tp, ep, is_multi in ablation_configs:
        cfg_key = f"tp{tp}_ep{ep}_multi" if is_multi else f"tp{tp}_ep{ep}"
        ablation_results[cfg_key] = {}
        for var in variants:
            print(f"\n--- config: {cfg_key} | variant: {var} ---")
            metrics = run_sim(tp, ep, is_multi, var)
            ablation_results[cfg_key][var] = metrics

    # ====================================================
    # 2. EP & TP Scaling Separation
    # ====================================================
    print("\n=== RUNNING SWEEP 2: EP & TP Scaling Separation ===")
    # Fixed TP=8, scale EP
    for ep in [1, 2, 4, 8]:
        cfg_key = f"fixed_tp8_ep{ep}"
        ep_tp_results[cfg_key] = {}
        for var in ["Baseline", "Full EPEG"]:
            # Check if we already evaluated this in ablation
            if ep == 8 and var in ablation_results.get("tp8_ep8", {}):
                metrics = ablation_results["tp8_ep8"][var]
            else:
                metrics = run_sim(8, ep, False, var)
            ep_tp_results[cfg_key][var] = metrics
            
    # Fixed EP=1, scale TP
    for tp in [1, 2, 4, 8]:
        cfg_key = f"fixed_ep1_tp{tp}"
        ep_tp_results[cfg_key] = {}
        for var in ["Baseline", "Full EPEG"]:
            if tp == 8 and var in ep_tp_results.get("fixed_tp8_ep1", {}):
                metrics = ep_tp_results["fixed_tp8_ep1"][var]
            else:
                metrics = run_sim(tp, 1, False, var)
            ep_tp_results[cfg_key][var] = metrics

    # Extra EP scaling points for TP=2 and TP=4
    # TP=2, EP=2
    ep_tp_results["fixed_tp2_ep2"] = {}
    for var in ["Baseline", "Full EPEG"]:
        ep_tp_results["fixed_tp2_ep2"][var] = run_sim(2, 2, False, var)
    # TP=4, EP=2
    ep_tp_results["fixed_tp4_ep2"] = {}
    for var in ["Baseline", "Full EPEG"]:
        ep_tp_results["fixed_tp4_ep2"][var] = run_sim(4, 2, False, var)
    # TP=4, EP=4
    ep_tp_results["fixed_tp4_ep4"] = {}
    for var in ["Baseline", "Full EPEG"]:
        if var in ablation_results.get("tp4_ep4", {}):
            ep_tp_results["fixed_tp4_ep4"][var] = ablation_results["tp4_ep4"][var]
        else:
            ep_tp_results["fixed_tp4_ep4"][var] = run_sim(4, 4, False, var)

    # ====================================================
    # 3. Interconnect Speed Sensitivity
    # ====================================================
    print("\n=== RUNNING SWEEP 3: Interconnect Speed Sensitivity ===")
    speeds = [0.25, 1.0, 4.0]
    multi_configs = [
        # (tp, ep)
        (2, 4),
        (4, 8)
    ]
    for tp, ep in multi_configs:
        cfg_key = f"tp{tp}_ep{ep}_dp2"
        bw_sensitivity_results[cfg_key] = {}
        for bw in speeds:
            bw_key = f"bw_{str(bw).replace('.', '_')}"
            bw_sensitivity_results[cfg_key][bw_key] = {}
            for var in variants:
                # Reuse if already run in ablation (bw=1.0)
                ab_key = f"tp{tp}_ep{ep}_multi"
                if bw == 1.0 and var in ablation_results.get(ab_key, {}):
                    metrics = ablation_results[ab_key][var]
                else:
                    metrics = run_sim(tp, ep, True, var, bw=bw)
                bw_sensitivity_results[cfg_key][bw_key][var] = metrics

    # ====================================================
    # 4. Top-k Scaling Study
    # ====================================================
    print("\n=== RUNNING SWEEP 4: Top-k Scaling ===")
    for k in [2, 4, 8]:
        k_key = f"k_{k}"
        topk_results[k_key] = {}
        for var in ["Baseline", "Full EPEG"]:
            if k == 8 and var in ablation_results.get("tp8_ep8", {}):
                metrics = ablation_results["tp8_ep8"][var]
            else:
                metrics = run_sim(8, 8, False, var, k=k)
            topk_results[k_key][var] = metrics

    # ====================================================
    # 5. Concurrency Scaling Study
    # ====================================================
    print("\n=== RUNNING SWEEP 5: Concurrency Scaling ===")
    for reqs in [10, 50, 200]:
        req_key = f"reqs_{reqs}"
        concurrency_results[req_key] = {}
        for var in ["Baseline", "Full EPEG"]:
            metrics = run_sim(8, 8, False, var, dataset=concurrency_dataset, num_reqs=reqs)
            concurrency_results[req_key][var] = metrics

    # ====================================================
    # Dump Consolidated Results
    # ====================================================
    consolidated_results = {
        "ablation": ablation_results,
        "ep_tp_scaling": ep_tp_results,
        "bw_sensitivity": bw_sensitivity_results,
        "topk": topk_results,
        "concurrency": concurrency_results
    }
    
    results_path = "outputs/phase3/epeg_tp_ep_sweep_results.json"
    with open(results_path, "w") as f:
        json.dump(consolidated_results, f, indent=4)
        
    print(f"\nAll sweeps completed successfully! Wrote consolidated JSON to {results_path}")

if __name__ == "__main__":
    main()
