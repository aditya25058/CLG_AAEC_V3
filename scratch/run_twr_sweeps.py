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

    avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0
    avg_tpot = sum(tpots) / len(tpots) if tpots else 0.0

    # Parse original load imbalance metrics before local load-balancing
    covs = []
    max_to_means = []
    matches = re.findall(r"\[DAEL_METRICS\] layer=\d+ cov=([\d.]+) max_to_mean=([\d.]+)", stdout)
    for cov_str, mtm_str in matches:
        covs.append(float(cov_str))
        max_to_means.append(float(mtm_str))
        
    avg_cov = sum(covs) / len(covs) if covs else 0.0
    avg_max_to_mean = sum(max_to_means) / len(max_to_means) if max_to_means else 1.0

    return {
        "total_latency_s": total_latency,
        "prompt_thru_tok_s": prompt_thru,
        "gen_thru_tok_s": gen_thru,
        "token_thru_tok_s": token_thru,
        "avg_ttft_ms": avg_ttft,
        "avg_tpot_ms": avg_tpot,
        "original_cov": avg_cov,
        "original_max_to_mean": avg_max_to_mean,
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

def make_temp_config(original_path, temp_path, new_bw, num_nodes=2):
    with open(original_path, 'r') as f:
        data = json.load(f)
    
    data["num_nodes"] = num_nodes
    data["link_bw"] = new_bw
    
    # Scale ep_size in instances list based on num_nodes
    single_node = data["nodes"][0]
    single_node["instances"][0]["ep_size"] = num_nodes * 2
    
    # Generate list of nodes
    data["nodes"] = [json.loads(json.dumps(single_node)) for _ in range(num_nodes)]
    
    with open(temp_path, 'w') as f:
        json.dump(data, f, indent=4)

def main():
    venv_python = "venv/bin/python3"
    os.makedirs("outputs/phase5", exist_ok=True)
    os.makedirs("configs/cluster", exist_ok=True)
    
    dataset_path = "datasets/qwen3_remote_10req_concurrent_fast.jsonl"
    base_config_path = "configs/cluster/test_dual_node_tp2_ep4.json"
    
    variants = {
        "baseline": [],
        "dael": ["--enable-dael", "--dael-saturation-threshold", "0.80", "--dael-redirect-fraction", "0.30"],
        "epeg": ["--enable-epeg", "--epeg-tau-high", "0.40", "--epeg-tau-low", "0.05"],
        "epeg_dael": ["--enable-epeg", "--epeg-tau-high", "0.40", "--epeg-tau-low", "0.05",
                      "--enable-dael", "--dael-saturation-threshold", "0.80", "--dael-redirect-fraction", "0.30"],
        "epeg_twr": ["--enable-epeg", "--epeg-tau-high", "0.40", "--epeg-tau-low", "0.05",
                     "--enable-twr", "--twr-alpha", "20.0"],
        "twr": ["--enable-twr", "--twr-alpha", "20.0"]
    }
    
    results = {
        "sweep1_bandwidth": {},
        "sweep2_scaling": {},
        "sweep3_skew": {},
        "sweep4_alpha": {}
    }
    
    # -------------------------------------------------------------------------
    # Sweep 1: Bandwidth Sensitivity (Across link bandwidths 1.0, 4.0, 16.0, 32.0)
    # -------------------------------------------------------------------------
    bandwidths = [1.0, 4.0, 16.0, 32.0]
    print("\n=== STARTING SWEEP 1: BANDWIDTH SENSITIVITY ===")
    for bw in bandwidths:
        bw_str = str(bw)
        results["sweep1_bandwidth"][bw_str] = {}
        temp_config = f"configs/cluster/temp_twr_sweep1_bw_{bw_str.replace('.', '_')}.json"
        make_temp_config(base_config_path, temp_config, bw, num_nodes=2)
        
        for var_id, var_flags in variants.items():
            csv_out = f"outputs/phase5/twr_sweep1_{var_id}_bw_{bw_str.replace('.', '_')}.csv"
            cmd = [
                venv_python, "-m", "serving",
                "--cluster-config", temp_config,
                "--dataset", dataset_path,
                "--num-reqs", "2",
                "--gpus-per-node", "2",
                "--expert-routing-policy", "BALANCED",
                "--output", csv_out
            ] + var_flags
            
            stdout = run_cmd(cmd)
            metrics = parse_metrics(stdout)
            results["sweep1_bandwidth"][bw_str][var_id] = metrics
            print(f"Sweep 1: BW={bw} GB/s | Var={var_id} -> Latency={metrics['total_latency_s']}s")
            
        if os.path.exists(temp_config):
            os.remove(temp_config)
            
    # -------------------------------------------------------------------------
    # Sweep 2: Multi-Node Scaling (2, 4, 8 nodes at BW = 16.0 GB/s)
    # -------------------------------------------------------------------------
    node_counts = [2]
    scaling_bw = 16.0
    scaling_variants = ["baseline", "epeg_dael", "epeg_twr", "twr"]
    print("\n=== STARTING SWEEP 2: MULTI-NODE SCALING ===")
    for nc in node_counts:
        nc_str = str(nc)
        results["sweep2_scaling"][nc_str] = {}
        temp_config = f"configs/cluster/temp_twr_sweep2_nc_{nc}.json"
        make_temp_config(base_config_path, temp_config, scaling_bw, num_nodes=nc)
        
        for var_id in scaling_variants:
            # We can reuse results from Sweep 1 for 2 nodes, but let's re-run or assign for completeness
            csv_out = f"outputs/phase5/twr_sweep2_{var_id}_nodes_{nc}.csv"
            cmd = [
                venv_python, "-m", "serving",
                "--cluster-config", temp_config,
                "--dataset", dataset_path,
                "--num-reqs", "2",
                "--gpus-per-node", "2",
                "--expert-routing-policy", "BALANCED",
                "--output", csv_out
            ] + variants[var_id]
            
            stdout = run_cmd(cmd)
            metrics = parse_metrics(stdout)
            results["sweep2_scaling"][nc_str][var_id] = metrics
            print(f"Sweep 2: Nodes={nc} | Var={var_id} -> Latency={metrics['total_latency_s']}s")
            
        if os.path.exists(temp_config):
            os.remove(temp_config)

    # -------------------------------------------------------------------------
    # Sweep 3: Hotspot Stress Test (Skew intensities 0.1, 0.3, 0.5, 0.7 at BW = 16.0 GB/s)
    # -------------------------------------------------------------------------
    skew_intensities = [0.1, 0.3, 0.5, 0.7]
    skew_variants = ["epeg", "epeg_dael", "epeg_twr", "twr"]
    print("\n=== STARTING SWEEP 3: HOTSPOT STRESS TEST ===")
    temp_config = f"configs/cluster/temp_twr_sweep3.json"
    make_temp_config(base_config_path, temp_config, 16.0, num_nodes=2)
    for skew in skew_intensities:
        skew_str = str(skew)
        results["sweep3_skew"][skew_str] = {}
        for var_id in skew_variants:
            csv_out = f"outputs/phase5/twr_sweep3_{var_id}_skew_{skew_str.replace('.', '_')}.csv"
            cmd = [
                venv_python, "-m", "serving",
                "--cluster-config", temp_config,
                "--dataset", dataset_path,
                "--num-reqs", "2",
                "--gpus-per-node", "2",
                "--expert-routing-policy", "BALANCED",
                "--expert-skew-intensity", skew_str,
                "--output", csv_out
            ] + variants[var_id]
            
            stdout = run_cmd(cmd)
            metrics = parse_metrics(stdout)
            results["sweep3_skew"][skew_str][var_id] = metrics
            print(f"Sweep 3: Skew={skew} | Var={var_id} -> Latency={metrics['total_latency_s']}s | CoV={metrics['original_cov']}")
            
    if os.path.exists(temp_config):
        os.remove(temp_config)

    # -------------------------------------------------------------------------
    # Sweep 4: NVLink Sensitivity (twr-alpha 10.0, 20.0, 50.0 at BW = 4.0 GB/s)
    # -------------------------------------------------------------------------
    alpha_values = [10.0, 20.0, 50.0]
    alpha_bw = 4.0
    print("\n=== STARTING SWEEP 4: ALPHA SENSITIVITY ===")
    temp_config = f"configs/cluster/temp_twr_sweep4.json"
    make_temp_config(base_config_path, temp_config, alpha_bw, num_nodes=2)
    for alpha in alpha_values:
        alpha_str = str(alpha)
        csv_out = f"outputs/phase5/twr_sweep4_alpha_{alpha_str.replace('.', '_')}.csv"
        # Run EPEG+TWR with the specific twr-alpha
        var_flags = ["--enable-epeg", "--epeg-tau-high", "0.40", "--epeg-tau-low", "0.05",
                     "--enable-twr", "--twr-alpha", alpha_str]
        cmd = [
            venv_python, "-m", "serving",
            "--cluster-config", temp_config,
            "--dataset", dataset_path,
            "--num-reqs", "2",
            "--gpus-per-node", "2",
            "--expert-routing-policy", "BALANCED",
            "--output", csv_out
        ] + var_flags
        
        stdout = run_cmd(cmd)
        metrics = parse_metrics(stdout)
        results["sweep4_alpha"][alpha_str] = metrics
        print(f"Sweep 4: Alpha={alpha} | EPEG+TWR -> Latency={metrics['total_latency_s']}s")
        
    if os.path.exists(temp_config):
        os.remove(temp_config)

    # Save results to JSON
    with open("outputs/phase5/twr_sweep_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\n=== SWEEPS COMPLETED SUCCESSFULLY ===")
    print("Wrote results to outputs/phase5/twr_sweep_results.json")

if __name__ == "__main__":
    main()
