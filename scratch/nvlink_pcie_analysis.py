#!/usr/bin/env python3
"""
NVLink vs PCIe analysis for all three models:
  - DeepSeek-R1 (256 experts, k=8, 61 layers, hidden=7168)
  - Llama-4 Maverick (128 experts, k=1, 48 layers, hidden=4096)
  - Qwen3-235B-A22B (128 experts, k=8, 94 layers, hidden=4096)

Extracts TTFT, TPOT, total latency, throughput from existing CSVs.
Produces comparison tables (PNG) and bar/line plots (PNG).
Produces a JSON with all extracted metrics.
"""

import csv
import json
import os
import sys
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')

# ── File mapping ──────────────────────────────────────────────────────────
# For each model we have TP=1 (baseline, no interconnect), TP=2/4/8 NVLink, TP=2/4/8 PCIe
FILES = {
    'DeepSeek-R1': {
        'tp1':       'deepseek_r1_h100_tp1_results.csv',        # TP=1 baseline
        'tp2_nvlink': 'deepseek_r1_h100_tp2_nvlink_results.csv',
        'tp2_pcie':   'deepseek_r1_h100_tp2_pcie_results.csv',
        'tp4_nvlink': 'deepseek_r1_h100_tp4_nvlink_results.csv',
        'tp4_pcie':   'deepseek_r1_h100_tp4_pcie_results.csv',
        'tp8_nvlink': 'deepseek_r1_h100_tp8_nvlink_results.csv',
        'tp8_pcie':   'deepseek_r1_h100_tp8_pcie_results.csv',
    },
    'Llama-4 Maverick': {
        'tp1':        'llama4_maverick_h100_tp1.csv',
        'tp2_nvlink': 'llama4_maverick_h100_tp2_nvlink.csv',
        'tp2_pcie':   'llama4_maverick_h100_tp2_pcie.csv',
        'tp4_nvlink': 'llama4_maverick_h100_tp4_nvlink.csv',
        'tp4_pcie':   'llama4_maverick_h100_tp4_pcie.csv',
        'tp8_nvlink': 'llama4_maverick_h100_tp8_nvlink.csv',
        'tp8_pcie':   'llama4_maverick_h100_tp8_pcie.csv',
    },
    'Qwen3-235B-A22B': {
        'tp1':        'qwen3_a22b_h100_tp1.csv',
        'tp2_nvlink': 'qwen3_a22b_h100_tp2_nvlink.csv',
        'tp2_pcie':   'qwen3_a22b_h100_tp2_pcie.csv',
        'tp4_nvlink': 'qwen3_a22b_h100_tp4_nvlink.csv',
        'tp4_pcie':   'qwen3_a22b_h100_tp4_pcie.csv',
        'tp8_nvlink': 'qwen3_a22b_h100_tp8_nvlink.csv',
        'tp8_pcie':   'qwen3_a22b_h100_tp8_pcie.csv',
    },
}


def parse_csv(filepath):
    """Parse a simulator output CSV and return aggregate metrics."""
    rows = []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return None

    # Values are in nanoseconds
    ttfts = [int(row['TTFT']) / 1e6 for row in rows]          # → ms
    tpots = [float(row['TPOT']) / 1e6 for row in rows]        # → ms
    latencies = [int(row['latency']) / 1e9 for row in rows]    # → seconds
    end_times = [int(row['end_time']) / 1e9 for row in rows]
    arrivals = [int(row['arrival']) / 1e9 for row in rows]
    inputs = [int(row['input']) for row in rows]
    outputs = [int(row['output']) for row in rows]

    total_time = max(end_times) - min(arrivals)
    total_prompt_tokens = sum(inputs)
    total_gen_tokens = sum(outputs)
    total_tokens = total_prompt_tokens + total_gen_tokens

    return {
        'avg_ttft_ms': round(statistics.mean(ttfts), 2),
        'avg_tpot_ms': round(statistics.mean(tpots), 2),
        'total_latency_s': round(total_time, 3),
        'prompt_thru_tok_s': round(total_prompt_tokens / total_time, 2),
        'gen_thru_tok_s': round(total_gen_tokens / total_time, 2),
        'token_thru_tok_s': round(total_tokens / total_time, 2),
        'num_requests': len(rows),
    }


def extract_all_metrics():
    """Extract metrics for every model × interconnect × TP combination."""
    all_metrics = {}
    for model, files in FILES.items():
        all_metrics[model] = {}
        for key, filename in files.items():
            filepath = os.path.join(OUTPUT_DIR, filename)
            if os.path.exists(filepath):
                m = parse_csv(filepath)
                if m:
                    all_metrics[model][key] = m
                    print(f"  ✓ {model} / {key}: latency={m['total_latency_s']}s, TTFT={m['avg_ttft_ms']}ms, TPOT={m['avg_tpot_ms']}ms")
                else:
                    print(f"  ✗ {model} / {key}: empty CSV")
            else:
                print(f"  ✗ {model} / {key}: file not found ({filepath})")
    return all_metrics


# ── Plotting ──────────────────────────────────────────────────────────────

COLORS = {
    'NVLink': '#00b4d8',
    'PCIe': '#e63946',
    'TP=1': '#8d99ae',
}

MODEL_COLORS = {
    'DeepSeek-R1': ('#2d6a4f', '#40916c', '#52b788'),
    'Llama-4 Maverick': ('#023e8a', '#0077b6', '#00b4d8'),
    'Qwen3-235B-A22B': ('#6a040f', '#9d0208', '#dc2f02'),
}


def plot_nvlink_vs_pcie_latency(all_metrics):
    """Grouped bar chart: total latency for NVLink vs PCIe, per model, per TP."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    fig.suptitle('NVLink vs PCIe: Total Serving Latency (lower is better)',
                 fontsize=16, fontweight='bold', y=1.02)

    tp_scales = [2, 4, 8]
    bar_width = 0.3

    for idx, model in enumerate(FILES.keys()):
        ax = axes[idx]
        metrics = all_metrics.get(model, {})

        nvlink_vals = []
        pcie_vals = []
        tp1_val = metrics.get('tp1', {}).get('total_latency_s', None)

        for tp in tp_scales:
            nv = metrics.get(f'tp{tp}_nvlink', {}).get('total_latency_s', 0)
            pc = metrics.get(f'tp{tp}_pcie', {}).get('total_latency_s', 0)
            nvlink_vals.append(nv)
            pcie_vals.append(pc)

        x = np.arange(len(tp_scales))
        bars_nv = ax.bar(x - bar_width/2, nvlink_vals, bar_width,
                         label='NVLink (900 GB/s)', color=COLORS['NVLink'],
                         edgecolor='white', linewidth=0.5, zorder=3)
        bars_pc = ax.bar(x + bar_width/2, pcie_vals, bar_width,
                         label='PCIe (16 GB/s)', color=COLORS['PCIe'],
                         edgecolor='white', linewidth=0.5, zorder=3)

        # TP=1 baseline line
        if tp1_val:
            ax.axhline(y=tp1_val, color=COLORS['TP=1'], linestyle='--',
                       linewidth=1.5, label=f'TP=1 baseline ({tp1_val:.2f}s)', zorder=2)

        # Value labels
        for bar in bars_nv:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{bar.get_height():.2f}s', ha='center', va='bottom', fontsize=8)
        for bar in bars_pc:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{bar.get_height():.2f}s', ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Tensor Parallelism Degree', fontsize=11)
        ax.set_ylabel('Total Latency (seconds)', fontsize=11)
        ax.set_title(model, fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'TP={t}' for t in tp_scales])
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(axis='y', alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'nvlink_pcie_latency_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✓ Saved: {path}")
    return path


def plot_nvlink_vs_pcie_ttft(all_metrics):
    """Grouped bar chart: TTFT for NVLink vs PCIe, per model, per TP."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    fig.suptitle('NVLink vs PCIe: Average Time-to-First-Token (lower is better)',
                 fontsize=16, fontweight='bold', y=1.02)

    tp_scales = [2, 4, 8]
    bar_width = 0.3

    for idx, model in enumerate(FILES.keys()):
        ax = axes[idx]
        metrics = all_metrics.get(model, {})

        nvlink_vals = []
        pcie_vals = []
        tp1_val = metrics.get('tp1', {}).get('avg_ttft_ms', None)

        for tp in tp_scales:
            nv = metrics.get(f'tp{tp}_nvlink', {}).get('avg_ttft_ms', 0)
            pc = metrics.get(f'tp{tp}_pcie', {}).get('avg_ttft_ms', 0)
            nvlink_vals.append(nv)
            pcie_vals.append(pc)

        x = np.arange(len(tp_scales))
        bars_nv = ax.bar(x - bar_width/2, nvlink_vals, bar_width,
                         label='NVLink (900 GB/s)', color=COLORS['NVLink'],
                         edgecolor='white', linewidth=0.5, zorder=3)
        bars_pc = ax.bar(x + bar_width/2, pcie_vals, bar_width,
                         label='PCIe (16 GB/s)', color=COLORS['PCIe'],
                         edgecolor='white', linewidth=0.5, zorder=3)

        if tp1_val:
            ax.axhline(y=tp1_val, color=COLORS['TP=1'], linestyle='--',
                       linewidth=1.5, label=f'TP=1 baseline ({tp1_val:.1f} ms)', zorder=2)

        for bar in bars_nv:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=7)
        for bar in bars_pc:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=7)

        ax.set_xlabel('Tensor Parallelism Degree', fontsize=11)
        ax.set_ylabel('Avg TTFT (ms)', fontsize=11)
        ax.set_title(model, fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'TP={t}' for t in tp_scales])
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(axis='y', alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'nvlink_pcie_ttft_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved: {path}")
    return path


def plot_nvlink_vs_pcie_tpot(all_metrics):
    """Grouped bar chart: TPOT for NVLink vs PCIe, per model, per TP."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    fig.suptitle('NVLink vs PCIe: Average Time-per-Output-Token (lower is better)',
                 fontsize=16, fontweight='bold', y=1.02)

    tp_scales = [2, 4, 8]
    bar_width = 0.3

    for idx, model in enumerate(FILES.keys()):
        ax = axes[idx]
        metrics = all_metrics.get(model, {})

        nvlink_vals = []
        pcie_vals = []
        tp1_val = metrics.get('tp1', {}).get('avg_tpot_ms', None)

        for tp in tp_scales:
            nv = metrics.get(f'tp{tp}_nvlink', {}).get('avg_tpot_ms', 0)
            pc = metrics.get(f'tp{tp}_pcie', {}).get('avg_tpot_ms', 0)
            nvlink_vals.append(nv)
            pcie_vals.append(pc)

        x = np.arange(len(tp_scales))
        bars_nv = ax.bar(x - bar_width/2, nvlink_vals, bar_width,
                         label='NVLink (900 GB/s)', color=COLORS['NVLink'],
                         edgecolor='white', linewidth=0.5, zorder=3)
        bars_pc = ax.bar(x + bar_width/2, pcie_vals, bar_width,
                         label='PCIe (16 GB/s)', color=COLORS['PCIe'],
                         edgecolor='white', linewidth=0.5, zorder=3)

        if tp1_val:
            ax.axhline(y=tp1_val, color=COLORS['TP=1'], linestyle='--',
                       linewidth=1.5, label=f'TP=1 baseline ({tp1_val:.1f} ms)', zorder=2)

        for bar in bars_nv:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=7)
        for bar in bars_pc:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=7)

        ax.set_xlabel('Tensor Parallelism Degree', fontsize=11)
        ax.set_ylabel('Avg TPOT (ms)', fontsize=11)
        ax.set_title(model, fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'TP={t}' for t in tp_scales])
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(axis='y', alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'nvlink_pcie_tpot_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved: {path}")
    return path


def plot_throughput_comparison(all_metrics):
    """Line plot: token throughput scaling for NVLink vs PCIe, all 3 models."""
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle('Token Throughput Scaling: NVLink vs PCIe (All Models)',
                 fontsize=16, fontweight='bold', y=0.98)

    tp_scales = [1, 2, 4, 8]
    markers = {'DeepSeek-R1': 'o', 'Llama-4 Maverick': 's', 'Qwen3-235B-A22B': '^'}

    for model in FILES.keys():
        metrics = all_metrics.get(model, {})
        colors = MODEL_COLORS[model]

        # NVLink line
        nv_vals = []
        nv_tps = []
        for tp in tp_scales:
            key = 'tp1' if tp == 1 else f'tp{tp}_nvlink'
            v = metrics.get(key, {}).get('token_thru_tok_s', None)
            if v:
                nv_vals.append(v)
                nv_tps.append(tp)
        if nv_vals:
            ax.plot(nv_tps, nv_vals, marker=markers[model], linewidth=2.5,
                    markersize=9, color=colors[0], label=f'{model} (NVLink)',
                    linestyle='-')

        # PCIe line
        pc_vals = []
        pc_tps = []
        for tp in tp_scales:
            if tp == 1:
                # same as NVLink for TP=1
                v = metrics.get('tp1', {}).get('token_thru_tok_s', None)
            else:
                v = metrics.get(f'tp{tp}_pcie', {}).get('token_thru_tok_s', None)
            if v:
                pc_vals.append(v)
                pc_tps.append(tp)
        if pc_vals:
            ax.plot(pc_tps, pc_vals, marker=markers[model], linewidth=2.5,
                    markersize=9, color=colors[2], label=f'{model} (PCIe)',
                    linestyle='--')

    ax.set_xlabel('Tensor Parallelism Degree', fontsize=13)
    ax.set_ylabel('Total Token Throughput (tok/s)', fontsize=13)
    ax.set_xticks(tp_scales)
    ax.set_xticklabels([f'TP={t}' for t in tp_scales])
    ax.legend(fontsize=9, loc='upper left', ncol=2)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'nvlink_pcie_throughput_scaling.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved: {path}")
    return path


def plot_pcie_penalty_heatmap(all_metrics):
    """Heatmap showing the PCIe slowdown percentage vs NVLink."""
    models = list(FILES.keys())
    tp_scales = [2, 4, 8]
    metrics_names = ['total_latency_s', 'avg_ttft_ms', 'avg_tpot_ms']
    metric_labels = ['Total Latency', 'TTFT', 'TPOT']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('PCIe Penalty vs NVLink (% slowdown, higher = worse for PCIe)',
                 fontsize=16, fontweight='bold', y=1.02)

    for midx, (mname, mlabel) in enumerate(zip(metrics_names, metric_labels)):
        ax = axes[midx]
        data = np.zeros((len(models), len(tp_scales)))

        for ridx, model in enumerate(models):
            metrics = all_metrics.get(model, {})
            for cidx, tp in enumerate(tp_scales):
                nv = metrics.get(f'tp{tp}_nvlink', {}).get(mname, None)
                pc = metrics.get(f'tp{tp}_pcie', {}).get(mname, None)
                if nv and pc and nv > 0:
                    pct = ((pc - nv) / nv) * 100
                    data[ridx, cidx] = round(pct, 1)

        im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0,
                       vmax=max(data.max(), 1))
        ax.set_xticks(range(len(tp_scales)))
        ax.set_xticklabels([f'TP={t}' for t in tp_scales])
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=9)
        ax.set_title(mlabel, fontsize=13, fontweight='bold')

        for i in range(len(models)):
            for j in range(len(tp_scales)):
                val = data[i, j]
                color = 'white' if val > data.max() * 0.6 else 'black'
                ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                        fontsize=11, fontweight='bold', color=color)

        plt.colorbar(im, ax=ax, shrink=0.8, label='% Slowdown')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'nvlink_pcie_penalty_heatmap.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved: {path}")
    return path


def create_results_table_png(all_metrics):
    """Create a formatted results table as PNG."""
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.axis('off')
    ax.set_title('NVLink vs PCIe — Full Results Table (All 3 Models)',
                 fontsize=16, fontweight='bold', pad=20)

    headers = ['Model', 'Config', 'BW', 'Total Lat (s)', 'Prompt (tok/s)',
               'Gen (tok/s)', 'Total (tok/s)', 'TTFT (ms)', 'TPOT (ms)']

    cell_data = []
    cell_colors = []
    nvlink_color = '#e0f7fa'
    pcie_color = '#fce4ec'
    baseline_color = '#f5f5f5'

    for model in FILES.keys():
        metrics = all_metrics.get(model, {})
        # TP=1 baseline
        m = metrics.get('tp1')
        if m:
            cell_data.append([model, 'TP=1', '—',
                              f"{m['total_latency_s']:.3f}",
                              f"{m['prompt_thru_tok_s']:.2f}",
                              f"{m['gen_thru_tok_s']:.2f}",
                              f"{m['token_thru_tok_s']:.2f}",
                              f"{m['avg_ttft_ms']:.2f}",
                              f"{m['avg_tpot_ms']:.2f}"])
            cell_colors.append([baseline_color]*9)

        for tp in [2, 4, 8]:
            for ic_type, ic_label, color in [('nvlink', 'NVLink', nvlink_color),
                                              ('pcie', 'PCIe', pcie_color)]:
                key = f'tp{tp}_{ic_type}'
                m = metrics.get(key)
                if m:
                    bw = '900 GB/s' if ic_type == 'nvlink' else '16 GB/s'
                    cell_data.append([model, f'TP={tp}', bw,
                                      f"{m['total_latency_s']:.3f}",
                                      f"{m['prompt_thru_tok_s']:.2f}",
                                      f"{m['gen_thru_tok_s']:.2f}",
                                      f"{m['token_thru_tok_s']:.2f}",
                                      f"{m['avg_ttft_ms']:.2f}",
                                      f"{m['avg_tpot_ms']:.2f}"])
                    cell_colors.append([color]*9)

    table = ax.table(cellText=cell_data, colLabels=headers,
                     cellColours=cell_colors,
                     colColours=['#263238']*9,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    # Header text color
    for j in range(len(headers)):
        table[0, j].set_text_props(color='white', fontweight='bold')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'nvlink_pcie_results_table.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved: {path}")
    return path


if __name__ == '__main__':
    print("=" * 60)
    print("NVLink vs PCIe Analysis — All Three Models")
    print("=" * 60)

    print("\n[1/6] Extracting metrics from CSVs...")
    all_metrics = extract_all_metrics()

    print("\n[2/6] Plotting latency comparison...")
    plot_nvlink_vs_pcie_latency(all_metrics)

    print("[3/6] Plotting TTFT comparison...")
    plot_nvlink_vs_pcie_ttft(all_metrics)

    print("[4/6] Plotting TPOT comparison...")
    plot_nvlink_vs_pcie_tpot(all_metrics)

    print("[5/6] Plotting throughput scaling...")
    plot_throughput_comparison(all_metrics)

    print("[6/6] Creating results table & heatmap...")
    create_results_table_png(all_metrics)
    plot_pcie_penalty_heatmap(all_metrics)

    # Save metrics JSON
    json_path = os.path.join(OUTPUT_DIR, 'nvlink_pcie_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    print(f"\n✓ Saved metrics JSON: {json_path}")
    print("\n✓ All done!")
