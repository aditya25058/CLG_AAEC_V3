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

    return {
        "total_latency_s": total_latency,
        "prompt_thru_tok_s": prompt_thru,
        "gen_thru_tok_s": gen_thru,
        "token_thru_tok_s": token_thru,
        "avg_ttft_ms": avg_ttft,
        "avg_tpot_ms": avg_tpot,
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
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("configs/cluster", exist_ok=True)
    
    dataset_path = "datasets/qwen3_remote_10req_concurrent.jsonl"
    
    # ----------------------------------------------------
    # SWEEP 1: Routing Policies (BALANCED, RAND, DATASET)
    # ----------------------------------------------------
    print("\n=== Running Sweep 1: Expert Routing Policies Scaling ===")
    sweep1_tps = [1, 2, 4, 8]
    sweep1_policies = ["BALANCED", "RAND", "DATASET"]
    sweep1_results = {tp: {} for tp in sweep1_tps}
    
    for tp in sweep1_tps:
        config_path = f"configs/cluster/single_node_qwen3_a22b_h100_tp{tp}.json"
        # Fallback if specific tp config doesn't exist
        if not os.path.exists(config_path):
            config_path = "configs/cluster/single_node_moe_single_instance.json"
            
        for policy in sweep1_policies:
            csv_out = f"outputs/sweep1_tp_{tp}_policy_{policy}.csv"
            cmd = [
                venv_python, "-m", "serving",
                "--cluster-config", config_path,
                "--dataset", dataset_path,
                "--num-reqs", "10",
                "--expert-routing-policy", policy,
                "--gpus-per-node", str(tp),
                "--output", csv_out
            ]
            stdout = run_cmd(cmd)
            metrics = parse_metrics(stdout)
            sweep1_results[tp][policy] = metrics
            print(f"Sweep 1: TP={tp} | Policy={policy} -> Latency: {metrics['total_latency_s']}s, TTFT: {metrics['avg_ttft_ms']}ms, TPOT: {metrics['avg_tpot_ms']}ms")

    # ----------------------------------------------------
    # SWEEP 2: Interconnect-Aware Gate Pruning (lambda_c)
    # ----------------------------------------------------
    print("\n=== Running Sweep 2: Interconnect-Aware Gate Pruning Trend ===")
    base_config_path = "configs/cluster/single_node_moe_single_instance.json"
    with open(base_config_path, "r") as f:
        base_config = json.load(f)
        
    sweep2_lambdas = [0.0, 0.2, 0.5, 0.8]
    sweep2_bandwidths = [1.0, 32.0]
    
    sweep2_results = {bw: {} for bw in sweep2_bandwidths}
    
    for bw in sweep2_bandwidths:
        temp_config = base_config.copy()
        temp_config["link_bw"] = bw
        temp_config_path = f"configs/cluster/temp_sweep2_bw_{str(bw).replace('.', '_')}.json"
        with open(temp_config_path, "w") as f:
            json.dump(temp_config, f, indent=4)
            
        for l in sweep2_lambdas:
            csv_out = f"outputs/sweep2_bw_{str(bw).replace('.', '_')}_lambda_{str(l).replace('.', '_')}.csv"
            cmd = [
                venv_python, "-m", "serving",
                "--cluster-config", temp_config_path,
                "--dataset", dataset_path,
                "--num-reqs", "10",
                "--expert-routing-policy", "DATASET",
                "--lambda-c", str(l),
                "--gpus-per-node", "1",
                "--output", csv_out
            ]
            stdout = run_cmd(cmd)
            metrics = parse_metrics(stdout)
            sweep2_results[bw][l] = metrics
            print(f"Sweep 2: BW={bw} GB/s | lambda_c={l} -> Latency: {metrics['total_latency_s']}s, TTFT: {metrics['avg_ttft_ms']}ms, TPOT: {metrics['avg_tpot_ms']}ms")
            
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

    # ----------------------------------------------------
    # SWEEP 3: HDFG Weight Fetching (Enabled vs Disabled)
    # ----------------------------------------------------
    print("\n=== Running Sweep 3: HDFG Weight Fetching Crossover Trend ===")
    hdfg_bandwidths = [1.0, 4.0, 16.0, 32.0]
    hdfg_options = [True, False] # Enabled vs Disabled
    
    sweep3_results = {bw: {} for bw in hdfg_bandwidths}
    
    for bw in hdfg_bandwidths:
        temp_config = base_config.copy()
        temp_config["link_bw"] = bw
        temp_config_path = f"configs/cluster/temp_sweep3_bw_{str(bw).replace('.', '_')}.json"
        with open(temp_config_path, "w") as f:
            json.dump(temp_config, f, indent=4)
            
        for enabled in hdfg_options:
            enabled_str = "enabled" if enabled else "disabled"
            csv_out = f"outputs/sweep3_bw_{str(bw).replace('.', '_')}_hdfg_{enabled_str}.csv"
            
            cmd = [
                venv_python, "-m", "serving",
                "--cluster-config", temp_config_path,
                "--dataset", dataset_path,
                "--num-reqs", "10",
                "--expert-routing-policy", "DATASET",
                "--gpus-per-node", "2",
                "--output", csv_out
            ]
            if enabled:
                cmd.append("--enable-hdfg")
            else:
                cmd.append("--no-enable-hdfg")
                
            stdout = run_cmd(cmd)
            metrics = parse_metrics(stdout)
            sweep3_results[bw][enabled] = metrics
            print(f"Sweep 3: BW={bw} GB/s | HDFG={enabled_str} -> Latency: {metrics['total_latency_s']}s, TTFT: {metrics['avg_ttft_ms']}ms, TPOT: {metrics['avg_tpot_ms']}ms")
            
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

    # ----------------------------------------------------
    # PHASE 2 SWEEP: Decomposed vs Flat collective (OMITTED)
    # ----------------------------------------------------
    print("\n=== Phase 2 Sweep: Flat vs Decomposed All-to-All (Omitted as requested) ===")

    # ----------------------------------------------------
    # Compile the final comprehensive Markdown report
    # ----------------------------------------------------
    markdown_lines = []
    markdown_lines.append("# Unified Co-Design Evaluation Report: MoE Serving (RCM Framework)")
    markdown_lines.append("\nThis report details the execution and results of Sweeps 1, 2, and 3 benchmarks, analyzing expert routing policy scaling, interconnect-aware gate pruning, and Hierarchical Dispatch-Fetch Gating (HDFG).")
    markdown_lines.append("\nDetailed execution logs and per-request trace CSVs are saved in the `outputs/` folder.")
    markdown_lines.append("\n---")
    
    # Sweep 1 section (Dynamic H100)
    markdown_lines.append("\n## 1. Sweep 1: Expert Routing Policies Scaling Trend")
    markdown_lines.append("\n| TP/EP Scale | Routing Policy | Total Latency (s) | Prompt Throughput (tok/s) | Gen Throughput (tok/s) | Mean TTFT (ms) | Mean TPOT (ms) |")
    markdown_lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: |")
    for tp in sweep1_tps:
        for policy in sweep1_policies:
            m = sweep1_results[tp][policy]
            markdown_lines.append(f"| **TP={tp} / EP={tp}** | **{policy}** | {m['total_latency_s']:.3f} | {m['prompt_thru_tok_s']:.2f} | {m['gen_thru_tok_s']:.2f} | {m['avg_ttft_ms']:.2f} | {m['avg_tpot_ms']:.2f} |")
            
    markdown_lines.append("\n### Graphical Results (Sweep 1)")
    markdown_lines.append("![Sweep 1 Results Table Grid](/home/palakm/MoEServingSim/outputs/sweep1_results_table.png)")
    markdown_lines.append("![Sweep 1 Latency Scaling Trend Plot](/home/palakm/MoEServingSim/outputs/sweep1_scaling_plot.png)")
    
    # Sweep 2 section
    markdown_lines.append("\n---")
    markdown_lines.append("\n## 2. Sweep 2: Interconnect-Aware Gate Pruning Trend")
    markdown_lines.append("\nEvaluating pruning strength parameter $\\lambda_c \\in [0.0, 0.2, 0.5, 0.8]$ across slow (1.0 GB/s) and fast (32.0 GB/s) inter-node interconnect settings.")
    markdown_lines.append("\n| Interconnect Speed | Pruning Factor $\\lambda_c$ | Total Latency (s) | Prompt Throughput (tok/s) | Gen Throughput (tok/s) | Mean TTFT (ms) | Mean TPOT (ms) |")
    markdown_lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for bw in sweep2_bandwidths:
        for l in sweep2_lambdas:
            speed_str = "Slow" if bw == 1.0 else "Fast"
            m = sweep2_results[bw][l]
            markdown_lines.append(f"| **{bw} GB/s ({speed_str})** | **{l}** | {m['total_latency_s']:.3f} | {m['prompt_thru_tok_s']:.2f} | {m['gen_thru_tok_s']:.2f} | {m['avg_ttft_ms']:.2f} | {m['avg_tpot_ms']:.2f} |")
 
    markdown_lines.append("\n### Graphical Results (Sweep 2)")
    markdown_lines.append("![Sweep 2 Results Table Grid](/home/palakm/MoEServingSim/outputs/sweep2_results_table.png)")
    markdown_lines.append("![Sweep 2 Latency Scaling Trend Plot](/home/palakm/MoEServingSim/outputs/sweep2_pruning_plot.png)")
 
    # Sweep 3 section (HDFG)
    markdown_lines.append("\n---")
    markdown_lines.append("\n## 3. Sweep 3: Hierarchical Dispatch-Fetch Gating (HDFG) Crossover Trend")
    markdown_lines.append("\nEvaluating HDFG (Enabled vs. Disabled) across varying link bandwidths to identify the crossover speed where weight prefetching outweighs token routing.")
    markdown_lines.append("\n| Interconnect Speed | HDFG Status | Total Latency (s) | Prompt Throughput (tok/s) | Gen Throughput (tok/s) | Mean TTFT (ms) | Mean TPOT (ms) |")
    markdown_lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for bw in hdfg_bandwidths:
        for enabled in hdfg_options:
            status_str = "Enabled" if enabled else "Disabled"
            m = sweep3_results[bw][enabled]
            markdown_lines.append(f"| **{bw} GB/s** | **{status_str}** | {m['total_latency_s']:.3f} | {m['prompt_thru_tok_s']:.2f} | {m['gen_thru_tok_s']:.2f} | {m['avg_ttft_ms']:.2f} | {m['avg_tpot_ms']:.2f} |")
            
    markdown_lines.append("\n### Graphical Results (Sweep 3)")
    markdown_lines.append("![Sweep 3 Results Table Grid](/home/palakm/MoEServingSim/outputs/sweep3_results_table.png)")
    markdown_lines.append("![Sweep 3 Crossover Trend Plot](/home/palakm/MoEServingSim/outputs/sweep3_crossover_plot.png)")

    # Dump results to a JSON file for plotting scripts to load dynamically
    with open("outputs/sweep_results.json", "w") as f:
        json.dump({
            "sweep1": {str(tp): res for tp, res in sweep1_results.items()},
            "sweep2": {str(bw): {str(l): m for l, m in res.items()} for bw, res in sweep2_results.items()},
            "sweep3": {str(bw): {str(enabled): m for enabled, m in res.items()} for bw, res in sweep3_results.items()}
        }, f, indent=4)

    report_content = "\n".join(markdown_lines) + "\n"
    
    os.makedirs("docs", exist_ok=True)
    with open("docs/phase1_report.md", "w") as f:
        f.write(report_content)
    with open("outputs/phase1_report.md", "w") as f:
        f.write(report_content)
        
    print("Successfully wrote docs/phase1_report.md and outputs/phase1_report.md")

if __name__ == "__main__":
    main()
