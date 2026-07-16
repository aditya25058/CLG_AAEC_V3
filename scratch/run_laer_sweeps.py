#!/usr/bin/env python3
"""LAER (Locality-Aware Expert Routing) evaluation sweeps.

Runs 5 experiments:
1. Baseline vs LAER: Pure LAER benefit at different bandwidths
2. Beta/Gamma Pareto Sweep: Quality vs latency trade-off across LAER parameters
3. Skew Robustness: LAER + DAEL under expert popularity skew
4. LAER + TWR Synergy: Combined optimization at different bandwidths
5. Multi-Node Scaling: LAER benefit across cluster sizes
"""
import subprocess
import re
import os
import json
import sys

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

    # Parse LAER metrics
    remote_fracs = [float(x) for x in re.findall(r"\[LAER_METRICS\] .* remote_frac=([\d.]+)", stdout)]
    quality_deltas = [float(x) for x in re.findall(r"\[LAER_METRICS\] .* quality_delta=([\d.]+)", stdout)]
    inter_nodes = [int(x) for x in re.findall(r"\[LAER_METRICS\] .* inter_node=(\d+)", stdout)]
    laer_redirected = [int(x) for x in re.findall(r"\[LAER_METRICS\] .* redirected=(\d+)", stdout)]

    avg_remote_frac = sum(remote_fracs) / len(remote_fracs) if remote_fracs else 0.0
    avg_quality_delta = sum(quality_deltas) / len(quality_deltas) if quality_deltas else 0.0
    total_inter_node = sum(inter_nodes) if inter_nodes else 0
    total_laer_redirected = sum(laer_redirected) if laer_redirected else 0

    # Parse DAEL metrics
    covs = [float(x) for x in re.findall(r"\[DAEL_METRICS\] .* cov=([\d.]+)", stdout)]
    max_to_means = [float(x) for x in re.findall(r"\[DAEL_METRICS\] .* max_to_mean=([\d.]+)", stdout)]
    avg_cov = sum(covs) / len(covs) if covs else 0.0
    avg_max_to_mean = sum(max_to_means) / len(max_to_means) if max_to_means else 1.0

    return {
        "total_latency_s": total_latency,
        "prompt_thru_tok_s": prompt_thru,
        "gen_thru_tok_s": gen_thru,
        "token_thru_tok_s": token_thru,
        "avg_ttft_ms": avg_ttft,
        "avg_tpot_ms": avg_tpot,
        "avg_remote_fraction": avg_remote_frac,
        "avg_quality_delta": avg_quality_delta,
        "total_inter_node_tokens": total_inter_node,
        "total_laer_redirected": total_laer_redirected,
        "original_cov": avg_cov,
        "original_max_to_mean": avg_max_to_mean,
    }

def run_cmd(cmd, env=None):
    print(f"  CMD: {' '.join(cmd[-8:])}")
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    res = subprocess.run(cmd, capture_output=True, text=True, env=run_env, timeout=300)
    if res.returncode != 0:
        print(f"  FAILED (rc={res.returncode}): {res.stderr[:200]}")
        return ""
    return res.stdout

def make_temp_config(original_path, temp_path, new_bw, num_nodes=2):
    with open(original_path, 'r') as f:
        data = json.load(f)
    data["num_nodes"] = num_nodes
    data["link_bw"] = new_bw
    single_node = data["nodes"][0]
    single_node["instances"][0]["ep_size"] = num_nodes * 2
    data["nodes"] = [json.loads(json.dumps(single_node)) for _ in range(num_nodes)]
    with open(temp_path, 'w') as f:
        json.dump(data, f, indent=4)


def main():
    venv_python = "venv/bin/python3"
    out_dir = "outputs/phase4/LAER"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("configs/cluster", exist_ok=True)

    dataset_path = "datasets/qwen3_remote_10req_concurrent_fast.jsonl"
    base_config_path = "configs/cluster/test_dual_node_tp2_ep4.json"

    results = {}

    # ============================================================================
    # SWEEP 1: Baseline vs LAER across bandwidths
    # ============================================================================
    print("\n" + "="*70)
    print("SWEEP 1: BASELINE vs LAER — Bandwidth Sensitivity")
    print("="*70)

    bandwidths = [1.0, 4.0, 16.0, 32.0]
    variants = {
        "baseline": [],
        "laer": ["--enable-laer", "--laer-beta", "0.95", "--laer-gamma", "0.70"],
        "laer_aggressive": ["--enable-laer", "--laer-beta", "0.90", "--laer-gamma", "0.50"],
    }
    results["sweep1_bandwidth"] = {}

    for bw in bandwidths:
        bw_str = str(bw)
        results["sweep1_bandwidth"][bw_str] = {}
        temp_config = f"configs/cluster/temp_laer_sweep1_bw_{bw_str.replace('.', '_')}.json"
        make_temp_config(base_config_path, temp_config, bw, num_nodes=2)

        for var_id, var_flags in variants.items():
            csv_out = f"{out_dir}/sweep1_{var_id}_bw_{bw_str.replace('.', '_')}.csv"
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
            print(f"  BW={bw} | {var_id}: Latency={metrics['total_latency_s']:.4f}s | Remote={metrics['avg_remote_fraction']:.4f} | Q-Delta={metrics['avg_quality_delta']:.6f}")

        if os.path.exists(temp_config):
            os.remove(temp_config)

    # ============================================================================
    # SWEEP 2: Beta/Gamma Pareto Sweep (Quality vs Latency trade-off)
    # ============================================================================
    print("\n" + "="*70)
    print("SWEEP 2: BETA/GAMMA PARETO SWEEP — Quality vs Latency")
    print("="*70)

    pareto_bw = 4.0
    temp_config = f"configs/cluster/temp_laer_sweep2.json"
    make_temp_config(base_config_path, temp_config, pareto_bw, num_nodes=2)

    beta_gamma_pairs = [
        (1.0, 1.0),   # No LAER (effectively disabled)
        (0.98, 0.90),  # Mild preference for local
        (0.95, 0.70),  # Default LAER
        (0.90, 0.50),  # Aggressive LAER
        (0.80, 0.30),  # Very aggressive LAER
        (0.70, 0.10),  # Extreme LAER (almost always local)
    ]
    results["sweep2_pareto"] = {}

    for beta, gamma in beta_gamma_pairs:
        key = f"b{beta}_g{gamma}"
        csv_out = f"{out_dir}/sweep2_pareto_{key.replace('.', '_')}.csv"
        cmd = [
            venv_python, "-m", "serving",
            "--cluster-config", temp_config,
            "--dataset", dataset_path,
            "--num-reqs", "2",
            "--gpus-per-node", "2",
            "--expert-routing-policy", "BALANCED",
            "--enable-laer", "--laer-beta", str(beta), "--laer-gamma", str(gamma),
            "--output", csv_out
        ]
        stdout = run_cmd(cmd)
        metrics = parse_metrics(stdout)
        metrics["beta"] = beta
        metrics["gamma"] = gamma
        results["sweep2_pareto"][key] = metrics
        print(f"  β={beta}, γ={gamma}: Latency={metrics['total_latency_s']:.4f}s | Remote={metrics['avg_remote_fraction']:.4f} | Q-Delta={metrics['avg_quality_delta']:.6f}")

    if os.path.exists(temp_config):
        os.remove(temp_config)

    # ============================================================================
    # SWEEP 3: Skew Robustness (LAER + DAEL under expert popularity skew)
    # ============================================================================
    print("\n" + "="*70)
    print("SWEEP 3: SKEW ROBUSTNESS — LAER + DAEL under Hotspots")
    print("="*70)

    skew_intensities = [0.0, 0.1, 0.3, 0.5, 0.7]
    skew_variants = {
        "baseline": [],
        "laer": ["--enable-laer", "--laer-beta", "0.95", "--laer-gamma", "0.70"],
        "dael": ["--enable-dael", "--dael-saturation-threshold", "0.80", "--dael-redirect-fraction", "0.30"],
        "laer_dael": ["--enable-laer", "--laer-beta", "0.95", "--laer-gamma", "0.70",
                       "--enable-dael", "--dael-saturation-threshold", "0.80", "--dael-redirect-fraction", "0.30"],
    }
    temp_config = f"configs/cluster/temp_laer_sweep3.json"
    make_temp_config(base_config_path, temp_config, 16.0, num_nodes=2)
    results["sweep3_skew"] = {}

    for skew in skew_intensities:
        skew_str = str(skew)
        results["sweep3_skew"][skew_str] = {}
        for var_id, var_flags in skew_variants.items():
            csv_out = f"{out_dir}/sweep3_{var_id}_skew_{skew_str.replace('.', '_')}.csv"
            cmd = [
                venv_python, "-m", "serving",
                "--cluster-config", temp_config,
                "--dataset", dataset_path,
                "--num-reqs", "2",
                "--gpus-per-node", "2",
                "--expert-routing-policy", "BALANCED",
                "--expert-skew-intensity", skew_str,
                "--output", csv_out
            ] + var_flags

            stdout = run_cmd(cmd)
            metrics = parse_metrics(stdout)
            results["sweep3_skew"][skew_str][var_id] = metrics
            print(f"  Skew={skew} | {var_id}: Latency={metrics['total_latency_s']:.4f}s | Remote={metrics['avg_remote_fraction']:.4f} | CoV={metrics['original_cov']:.4f}")

    if os.path.exists(temp_config):
        os.remove(temp_config)

    # ============================================================================
    # SWEEP 4: LAER + TWR + EPEG Synergy (Combined optimization)
    # ============================================================================
    print("\n" + "="*70)
    print("SWEEP 4: LAER + TWR + EPEG SYNERGY")
    print("="*70)

    synergy_bw = [1.0, 4.0, 16.0]
    synergy_variants = {
        "baseline": [],
        "laer_only": ["--enable-laer", "--laer-beta", "0.95", "--laer-gamma", "0.70"],
        "twr_only": ["--enable-twr", "--twr-alpha", "20.0"],
        "epeg_only": ["--enable-epeg", "--epeg-tau-high", "0.40", "--epeg-tau-low", "0.05"],
        "laer_twr": ["--enable-laer", "--laer-beta", "0.95", "--laer-gamma", "0.70",
                      "--enable-twr", "--twr-alpha", "20.0"],
        "laer_epeg": ["--enable-laer", "--laer-beta", "0.95", "--laer-gamma", "0.70",
                       "--enable-epeg", "--epeg-tau-high", "0.40", "--epeg-tau-low", "0.05"],
        "full_stack": ["--enable-laer", "--laer-beta", "0.95", "--laer-gamma", "0.70",
                        "--enable-twr", "--twr-alpha", "20.0",
                        "--enable-epeg", "--epeg-tau-high", "0.40", "--epeg-tau-low", "0.05"],
    }
    results["sweep4_synergy"] = {}

    for bw in synergy_bw:
        bw_str = str(bw)
        results["sweep4_synergy"][bw_str] = {}
        temp_config = f"configs/cluster/temp_laer_sweep4_bw_{bw_str.replace('.', '_')}.json"
        make_temp_config(base_config_path, temp_config, bw, num_nodes=2)

        for var_id, var_flags in synergy_variants.items():
            csv_out = f"{out_dir}/sweep4_{var_id}_bw_{bw_str.replace('.', '_')}.csv"
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
            results["sweep4_synergy"][bw_str][var_id] = metrics
            print(f"  BW={bw} | {var_id}: Latency={metrics['total_latency_s']:.4f}s | Remote={metrics['avg_remote_fraction']:.4f}")

        if os.path.exists(temp_config):
            os.remove(temp_config)

    # ============================================================================
    # SWEEP 5: Remote Token Fraction Deep Dive (the key LAER metric)
    # ============================================================================
    print("\n" + "="*70)
    print("SWEEP 5: REMOTE TOKEN FRACTION ANALYSIS")
    print("="*70)

    rtf_variants = {
        "no_laer": [],
        "laer_mild": ["--enable-laer", "--laer-beta", "0.98", "--laer-gamma", "0.90"],
        "laer_default": ["--enable-laer", "--laer-beta", "0.95", "--laer-gamma", "0.70"],
        "laer_strong": ["--enable-laer", "--laer-beta", "0.90", "--laer-gamma", "0.50"],
        "laer_extreme": ["--enable-laer", "--laer-beta", "0.80", "--laer-gamma", "0.30"],
    }
    temp_config = f"configs/cluster/temp_laer_sweep5.json"
    make_temp_config(base_config_path, temp_config, 4.0, num_nodes=2)
    results["sweep5_rtf"] = {}

    for var_id, var_flags in rtf_variants.items():
        csv_out = f"{out_dir}/sweep5_{var_id}.csv"
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
        results["sweep5_rtf"][var_id] = metrics
        print(f"  {var_id}: Latency={metrics['total_latency_s']:.4f}s | Remote Frac={metrics['avg_remote_fraction']:.4f} | Inter-node={metrics['total_inter_node_tokens']} | Q-Delta={metrics['avg_quality_delta']:.6f}")

    if os.path.exists(temp_config):
        os.remove(temp_config)

    # Save all results
    results_path = f"{out_dir}/laer_sweep_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n{'='*70}")
    print(f"ALL LAER SWEEPS COMPLETED!")
    print(f"Results: {results_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
