#!/usr/bin/env python3
import subprocess
import re
import os
import json
import matplotlib.pyplot as plt

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

    ttfts = [float(x) for x in re.findall(r"Mean TTFT \(ms\):\s*([\d.]+)", stdout)]
    tpots = [float(x) for x in re.findall(r"Mean TPOT \(ms\):\s*([\d.]+)", stdout)]
    avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0
    avg_tpot = sum(tpots) / len(tpots) if tpots else 0.0

    # Parse AAEC metrics
    speculated_list = [int(x) for x in re.findall(r"\[AAEC_METRICS\] .* speculated=(\d+)", stdout)]
    bypassed_list = [int(x) for x in re.findall(r"\[AAEC_METRICS\] .* bypassed=(\d+)", stdout)]
    q_deltas = [float(x) for x in re.findall(r"\[AAEC_METRICS\] .* quality_delta=([\d.]+)", stdout)]
    bg_bytes_list = [int(x) for x in re.findall(r"\[AAEC_METRICS\] .* background_bytes=(\d+)", stdout)]

    total_speculated = sum(speculated_list)
    total_bypassed = sum(bypassed_list)
    avg_quality_delta = sum(q_deltas) / len(q_deltas) if q_deltas else 0.0
    total_bg_bytes = sum(bg_bytes_list)

    total_decisions = total_speculated + total_bypassed
    hit_rate = total_speculated / max(1, total_decisions)

    return {
        "total_latency_s": total_latency,
        "prompt_thru_tok_s": prompt_thru,
        "gen_thru_tok_s": gen_thru,
        "avg_ttft_ms": avg_ttft,
        "avg_tpot_ms": avg_tpot,
        "hit_rate": hit_rate,
        "quality_delta": avg_quality_delta,
        "background_bytes": total_bg_bytes,
        "total_speculated": total_speculated,
        "total_bypassed": total_bypassed
    }

def run_cmd(cmd, env=None):
    print(f"Running command: {' '.join(cmd[-8:])}")
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    res = subprocess.run(cmd, capture_output=True, text=True, env=run_env, timeout=600)
    if res.returncode != 0:
        print(f"Command failed: {res.stderr[:150]}")
        return ""
    return res.stdout

def make_temp_config(original_path, temp_path, new_bw, num_nodes=2):
    with open(original_path, 'r') as f:
        data = json.load(f)
    data["num_nodes"] = num_nodes
    data["link_bw"] = new_bw
    single_node = data["nodes"][0]
    single_node["instances"][0]["ep_size"] = num_nodes * 2
    # For single-node configs, remove dp_group to avoid DP-group ASTRA-Sim SIGABRT
    if num_nodes == 1:
        single_node["instances"][0].pop("dp_group", None)
    data["nodes"] = [json.loads(json.dumps(single_node)) for _ in range(num_nodes)]
    with open(temp_path, 'w') as f:
        json.dump(data, f, indent=4)

def main():
    venv_python = "venv/bin/python3"
    out_dir = "outputs/phase4"
    os.makedirs(out_dir, exist_ok=True)

    dataset_path = "datasets/aaec_sweep_dataset.jsonl"
    base_config_path = "configs/cluster/test_dual_node_tp2_ep4.json"

    results = {}

    # ============================================================================
    # SWEEP 1: SOTA Comparison across bandwidths
    # ============================================================================
    print("\n" + "="*70)
    print("EXP 1: SOTA COMPARISON vs Bandwidth Sensitivity")
    print("="*70)
    bandwidths = [2.0, 16.0]
    variants = {
        "Baseline": [],
        "LAER": ["--enable-laer", "--laer-beta", "0.95", "--laer-gamma", "0.70"],
        "DAEL": ["--enable-dael", "--dael-saturation-threshold", "0.80", "--dael-redirect-fraction", "0.30"],
        "DeepEP (Uniform FP8)": ["--enable-epeg", "--epeg-tau-high", "2.0", "--epeg-tau-low", "0.0"],
        "AAEC (Ours)": ["--enable-aaec", "--aaec-cache-size", "128", "--aaec-dma-batch-layers", "4", "--slsr-speculation-threshold", "0.20"]
    }
    results["bandwidth_sweep"] = {}

    for bw in bandwidths:
        bw_str = str(bw)
        results["bandwidth_sweep"][bw_str] = {}
        temp_config = f"configs/cluster/temp_aaec_bw_{bw_str.replace('.', '_')}.json"
        make_temp_config(base_config_path, temp_config, bw, num_nodes=1)

        for var_name, var_flags in variants.items():
            cmd = [
                venv_python, "-m", "serving",
                "--cluster-config", temp_config,
                "--dataset", dataset_path,
                "--num-reqs", "1",
                "--gpus-per-node", "2",
                "--expert-routing-policy", "BALANCED",
                "--output", f"{out_dir}/sweep_bw_{bw_str.replace('.', '_')}_{var_name.replace(' ', '_').replace('(', '').replace(')', '')}.csv"
            ] + var_flags

            stdout = run_cmd(cmd)
            metrics = parse_metrics(stdout)
            results["bandwidth_sweep"][bw_str][var_name] = metrics
            print(f"  BW={bw} | {var_name}: Latency={metrics['total_latency_s']:.4f}s | HitRate={metrics['hit_rate']:.4f}")

        if os.path.exists(temp_config):
            os.remove(temp_config)

    # ============================================================================
    # SWEEP 2: Cache Size Tradeoff
    # ============================================================================
    print("\n" + "="*70)
    print("EXP 2: CACHE SIZE TRADEOFF")
    print("="*70)
    cache_sizes = [64, 128]
    results["cache_size_sweep"] = {}
    temp_config = "configs/cluster/temp_aaec_size.json"
    make_temp_config(base_config_path, temp_config, 4.0, num_nodes=1)

    for cs in cache_sizes:
        cmd = [
            venv_python, "-m", "serving",
            "--cluster-config", temp_config,
            "--dataset", dataset_path,
            "--num-reqs", "1",
            "--gpus-per-node", "2",
            "--expert-routing-policy", "BALANCED",
            "--enable-aaec",
            "--aaec-cache-size", str(cs),
            "--aaec-dma-batch-layers", "4",
            "--slsr-speculation-threshold", "0.20",
            "--output", f"{out_dir}/sweep_size_{cs}.csv"
        ]
        stdout = run_cmd(cmd)
        metrics = parse_metrics(stdout)
        results["cache_size_sweep"][str(cs)] = metrics
        print(f"  CacheSize={cs} | Latency={metrics['total_latency_s']:.4f}s | HitRate={metrics['hit_rate']:.4f} | BGBytes={metrics['background_bytes']}")

    # ============================================================================
    # SWEEP 3: Caching Policies comparison
    # ============================================================================
    print("\n" + "="*70)
    print("EXP 3: CACHING POLICIES (LRU vs LFU vs AAEC)")
    print("="*70)
    policies = ["LRU", "AAEC"]
    results["policy_sweep"] = {}

    for pol in policies:
        cmd = [
            venv_python, "-m", "serving",
            "--cluster-config", temp_config,
            "--dataset", dataset_path,
            "--num-reqs", "1",
            "--gpus-per-node", "2",
            "--expert-routing-policy", "BALANCED",
            "--enable-aaec",
            "--aaec-cache-size", "128",
            "--aaec-dma-batch-layers", "4",
            "--aaec-policy", pol,
            "--slsr-speculation-threshold", "0.20",
            "--output", f"{out_dir}/sweep_policy_{pol}.csv"
        ]
        stdout = run_cmd(cmd)
        metrics = parse_metrics(stdout)
        results["policy_sweep"][pol] = metrics
        print(f"  Policy={pol} | Latency={metrics['total_latency_s']:.4f}s | HitRate={metrics['hit_rate']:.4f}")

    # ============================================================================
    # SWEEP 4: Hit Threshold sensitivity
    # ============================================================================
    print("\n" + "="*70)
    print("EXP 4: HIT THRESHOLD SENSITIVITY")
    print("="*70)
    thresholds = [0.20, 0.50]
    results["threshold_sweep"] = {}

    for th in thresholds:
        cmd = [
            venv_python, "-m", "serving",
            "--cluster-config", temp_config,
            "--dataset", dataset_path,
            "--num-reqs", "1",
            "--gpus-per-node", "2",
            "--expert-routing-policy", "BALANCED",
            "--enable-aaec",
            "--aaec-cache-size", "128",
            "--aaec-dma-batch-layers", "4",
            "--slsr-speculation-threshold", str(th),
            "--output", f"{out_dir}/sweep_th_{str(th).replace('.', '_')}.csv"
        ]
        stdout = run_cmd(cmd)
        metrics = parse_metrics(stdout)
        results["threshold_sweep"][str(th)] = metrics
        print(f"  Threshold={th} | Latency={metrics['total_latency_s']:.4f}s | HitRate={metrics['hit_rate']:.4f} | QualityDelta={metrics['quality_delta']:.6f}")

    if os.path.exists(temp_config):
        os.remove(temp_config)

    # Save to JSON
    with open(f"{out_dir}/aaec_results.json", "w") as f:
        json.dump(results, f, indent=4)

    # ============================================================================
    # PLOTTING
    # ============================================================================
    print("\nPlotting results...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Color definitions
    colors = {
        "Baseline": "#94A3B8",
        "LAER": "#F59E0B",
        "DAEL": "#10B981",
        "DeepEP (Uniform FP8)": "#3B82F6",
        "AAEC (Ours)": "#EF4444"
    }

    # Plot 1: SOTA Comparison vs Bandwidth
    ax = axs[0, 0]
    for var_name in variants.keys():
        latencies = [results["bandwidth_sweep"][str(bw)][var_name]["total_latency_s"] for bw in bandwidths]
        ax.plot(bandwidths, latencies, marker='o', linewidth=2, label=var_name, color=colors[var_name])
    ax.set_title("Serving Latency vs Network Bandwidth", fontsize=12, fontweight='bold')
    ax.set_xlabel("Interconnect Bandwidth (GB/s)", fontsize=10)
    ax.set_ylabel("Latency (seconds)", fontsize=10)
    ax.legend(frameon=True, facecolor='white', edgecolor='#E2E8F0')

    # Plot 2: Hit Rate vs Caching Policy
    ax = axs[0, 1]
    hr_policies = [results["policy_sweep"][p]["hit_rate"] * 100 for p in policies]
    ax.bar(policies, hr_policies, color=["#3B82F6", "#10B981", "#EF4444"], width=0.5, edgecolor="#E2E8F0", alpha=0.9)
    ax.set_title("Neuron Cache Hit Rate vs Eviction Policy", fontsize=12, fontweight='bold')
    ax.set_xlabel("Policy", fontsize=10)
    ax.set_ylabel("Hit Rate (%)", fontsize=10)
    ax.set_ylim(0, 100)

    # Plot 3: Latency & DMA Bytes vs Cache Size
    ax = axs[1, 0]
    cs_lat = [results["cache_size_sweep"][str(cs)]["total_latency_s"] for cs in cache_sizes]
    ax.plot(cache_sizes, cs_lat, marker='s', color="#EF4444", linewidth=2, label="Latency (s)")
    ax.set_xlabel("Cache Size (Neurons)", fontsize=10)
    ax.set_ylabel("Latency (seconds)", color="#EF4444", fontsize=10)
    ax.tick_params(axis='y', labelcolor="#EF4444")
    
    ax2 = ax.twinx()
    cs_bytes = [results["cache_size_sweep"][str(cs)]["background_bytes"] / (1024 * 1024) for cs in cache_sizes]
    ax2.plot(cache_sizes, cs_bytes, marker='^', color="#3B82F6", linewidth=2, linestyle='--', label="DMA Traffic (MB)")
    ax2.set_ylabel("Asynchronous DMA Volume (MB)", color="#3B82F6", fontsize=10)
    ax2.tick_params(axis='y', labelcolor="#3B82F6")
    ax.set_title("Memory-Latency Tradeoff (AAEC)", fontsize=12, fontweight='bold')

    # Plot 4: Hit Threshold Sweep
    ax = axs[1, 1]
    th_lat = [results["threshold_sweep"][str(th)]["total_latency_s"] for th in thresholds]
    th_qd = [results["threshold_sweep"][str(th)]["quality_delta"] * 1000 for th in thresholds]
    
    ax.plot(thresholds, th_lat, marker='o', color="#10B981", linewidth=2)
    ax.set_xlabel("Hit Threshold (\u03b8_filter)", fontsize=10)
    ax.set_ylabel("Latency (seconds)", color="#10B981", fontsize=10)
    ax.tick_params(axis='y', labelcolor="#10B981")
    
    ax3 = ax.twinx()
    ax3.plot(thresholds, th_qd, marker='x', color="#F59E0B", linewidth=2, linestyle=':')
    ax3.set_ylabel("Quality Penalty (x10^-3)", color="#F59E0B", fontsize=10)
    ax3.tick_params(axis='y', labelcolor="#F59E0B")
    ax.set_title("Accuracy-Latency Tradeoff Frontier", fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{out_dir}/aaec_results_plot.png", dpi=150)
    print(f"Results chart saved to {out_dir}/aaec_results_plot.png")

if __name__ == "__main__":
    main()
