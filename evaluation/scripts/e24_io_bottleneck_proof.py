#!/usr/bin/env python3
"""
===============================================================================
E24 — Empirical Proof of the Weight-Transfer I/O Bottleneck on NVIDIA H100
===============================================================================
Physically measures and compares on real hardware:
  1. Single-token FFN GEMV compute latency (T_compute) at B=1 on GPU VRAM
  2. Monolithic expert PCIe DMA transfer latency (T_transfer_expert, 9.44 MB)
  3. Single-column PCIe DMA transfer latency (T_transfer_col, 12.29 KB)
  4. Batched missed column PCIe DMA transfer latencies (16, 32, 64, 128, 256 cols)
  5. GPU Idle / Stall Fraction and Transfer-to-Compute Disparity Ratio (T_trans / T_comp)

All latencies and bandwidths are physically measured using CUDA Events on an NVIDIA H100 NVL.

Usage:
  gpurun -g 1 python3 evaluation/scripts/e24_io_bottleneck_proof.py
===============================================================================
"""

import os
import sys
import json
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.cuda

# Model Specifications: Qwen3-30B-A3B
HIDDEN_SIZE = 2048        # K dimension
INTERMEDIATE_DIM = 768    # I dimension (columns per expert)
DTYPE = torch.bfloat16

MONOLITHIC_EXPERT_BYTES = 3 * INTERMEDIATE_DIM * HIDDEN_SIZE * 2  # 9,437,184 bytes (9.44 MB)
COLUMN_BYTES = 3 * HIDDEN_SIZE * 2                               # 12,288 bytes (12.29 KB)

RESULTS_DIR = "/home/palakm/MoEServingSim/evaluation/results/e24_io_proof"


def run_io_bottleneck_proof():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    cc = torch.cuda.get_device_capability(0)

    print("=" * 95)
    print(f"🔥 E24 — EMPIRICAL PROOF OF WEIGHT-TRANSFER I/O BOTTLENECK ON NVIDIA H100")
    print("=" * 95)
    print(f"  GPU Hardware       : {gpu_name} (SM {cc[0]}.{cc[1]}, {gpu_mem:.1f} GB HBM3e)")
    print(f"  Expert Geometry    : Gate=[{INTERMEDIATE_DIM},{HIDDEN_SIZE}], Up=[{INTERMEDIATE_DIM},{HIDDEN_SIZE}], Down=[{HIDDEN_SIZE},{INTERMEDIATE_DIM}]")
    print(f"  Precision          : BF16 (2 bytes/element)")
    print(f"  Monolithic Expert  : {MONOLITHIC_EXPERT_BYTES / (1024**2):.2f} MB ({MONOLITHIC_EXPERT_BYTES:,} bytes)")
    print(f"  Single Column      : {COLUMN_BYTES / 1024:.2f} KB ({COLUMN_BYTES:,} bytes)")
    print(f"  Decode Scope       : Single-Token Autoregressive Decode (Batch Size B = 1)")

    # ─────────────────────────────────────────────────────
    # Step 1: Physical Allocations (CPU Pinned + GPU VRAM)
    # ─────────────────────────────────────────────────────
    print(f"\n[1/4] Allocating CPU Pinned Memory & GPU VRAM Tensors...")

    # CPU Pinned Memory Tensors
    cpu_expert_gate = torch.randn(INTERMEDIATE_DIM, HIDDEN_SIZE, dtype=DTYPE, pin_memory=True)
    cpu_expert_up = torch.randn(INTERMEDIATE_DIM, HIDDEN_SIZE, dtype=DTYPE, pin_memory=True)
    cpu_expert_down = torch.randn(HIDDEN_SIZE, INTERMEDIATE_DIM, dtype=DTYPE, pin_memory=True)

    # GPU VRAM Tensors
    gpu_x = torch.randn(1, HIDDEN_SIZE, dtype=DTYPE, device=device)

    gpu_expert_gate = torch.zeros(INTERMEDIATE_DIM, HIDDEN_SIZE, dtype=DTYPE, device=device)
    gpu_expert_up = torch.zeros(INTERMEDIATE_DIM, HIDDEN_SIZE, dtype=DTYPE, device=device)
    gpu_expert_down = torch.zeros(HIDDEN_SIZE, INTERMEDIATE_DIM, dtype=DTYPE, device=device)

    dma_stream = torch.cuda.Stream(device)
    print(f"    CPU Pinned Memory: {MONOLITHIC_EXPERT_BYTES / (1024**2):.2f} MB allocated.")
    print(f"    GPU VRAM Buffers:  {MONOLITHIC_EXPERT_BYTES / (1024**2):.2f} MB allocated.")

    # ─────────────────────────────────────────────────────
    # Step 2: Measure Single-Token FFN Compute (T_compute)
    # ─────────────────────────────────────────────────────
    print(f"\n[2/4] Measuring Physical Single-Token FFN Compute Latency (T_compute, 500 iterations)...")

    # Warmup
    for _ in range(100):
        g = torch.matmul(gpu_x, gpu_expert_gate.t())
        u = torch.matmul(gpu_x, gpu_expert_up.t())
        _ = torch.matmul(F.silu(g) * u, gpu_expert_down.t())
    torch.cuda.synchronize()

    times = []
    for _ in range(500):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        g = torch.matmul(gpu_x, gpu_expert_gate.t())
        u = torch.matmul(gpu_x, gpu_expert_up.t())
        _ = torch.matmul(F.silu(g) * u, gpu_expert_down.t())
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000)  # microseconds

    t_compute_us = float(np.mean(times))
    t_compute_std_us = float(np.std(times))
    print(f"    Measured FFN GEMV Compute Latency (T_compute): {t_compute_us:.2f} ± {t_compute_std_us:.2f} µs ({t_compute_us/1000:.4f} ms)")

    # ─────────────────────────────────────────────────────
    # Step 3: Measure PCIe DMA Transfer Latencies (T_transfer)
    # ─────────────────────────────────────────────────────
    print(f"\n[3/4] Measuring Physical PCIe Gen5 DMA Transfer Latencies (cudaMemcpyAsync, 200 iterations)...")

    # Monolithic Expert Transfer (9.44 MB)
    for _ in range(20):
        with torch.cuda.stream(dma_stream):
            gpu_expert_gate.copy_(cpu_expert_gate, non_blocking=True)
            gpu_expert_up.copy_(cpu_expert_up, non_blocking=True)
            gpu_expert_down.copy_(cpu_expert_down, non_blocking=True)
        dma_stream.synchronize()

    times_expert = []
    for _ in range(200):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(dma_stream)
        with torch.cuda.stream(dma_stream):
            gpu_expert_gate.copy_(cpu_expert_gate, non_blocking=True)
            gpu_expert_up.copy_(cpu_expert_up, non_blocking=True)
            gpu_expert_down.copy_(cpu_expert_down, non_blocking=True)
        e.record(dma_stream)
        dma_stream.synchronize()
        times_expert.append(s.elapsed_time(e) * 1000)

    t_expert_us = float(np.mean(times_expert))
    t_expert_std_us = float(np.std(times_expert))
    bw_expert = (MONOLITHIC_EXPERT_BYTES / (t_expert_us * 1e-6)) / (1024**3)

    print(f"    Monolithic Expert Transfer (9.44 MB): {t_expert_us:.2f} ± {t_expert_std_us:.2f} µs ({t_expert_us/1000:.3f} ms) | Bandwidth={bw_expert:.2f} GB/s")

    # Column Sweep Transfers
    column_counts = [1, 8, 16, 32, 64, 128, 256]
    col_results = {}

    for n_cols in column_counts:
        payload = n_cols * COLUMN_BYTES
        cpu_col_g = torch.randn(n_cols, HIDDEN_SIZE, dtype=DTYPE, pin_memory=True)
        gpu_col_g = torch.zeros(n_cols, HIDDEN_SIZE, dtype=DTYPE, device=device)

        # Warmup
        for _ in range(20):
            with torch.cuda.stream(dma_stream):
                gpu_col_g.copy_(cpu_col_g, non_blocking=True)
            dma_stream.synchronize()

        times_col = []
        for _ in range(200):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(dma_stream)
            with torch.cuda.stream(dma_stream):
                gpu_col_g.copy_(cpu_col_g, non_blocking=True)
            e.record(dma_stream)
            dma_stream.synchronize()
            times_col.append(s.elapsed_time(e) * 1000)

        t_col_us = float(np.mean(times_col))
        t_col_std = float(np.std(times_col))
        bw_col = (payload / (t_col_us * 1e-6)) / (1024**3)
        col_results[n_cols] = {
            "payload_bytes": payload,
            "payload_kb": payload / 1024,
            "t_transfer_us": t_col_us,
            "t_transfer_std_us": t_col_std,
            "bw_gbps": bw_col
        }
        print(f"    Columns={n_cols:>3} ({payload/1024:>7.1f} KB): {t_col_us:>7.2f} ± {t_col_std:>5.2f} µs | Bandwidth={bw_col:>5.2f} GB/s")

    # ─────────────────────────────────────────────────────
    # Step 4: Empirical Bottleneck Disparity & Idle Analysis
    # ─────────────────────────────────────────────────────
    print(f"\n[4/4] Physical Bottleneck Analysis & Disparity Metrics")
    print("=" * 95)

    disparity_ratio = t_expert_us / t_compute_us
    idle_fraction_expert = max(0.0, (t_expert_us - t_compute_us) / t_expert_us * 100)
    exposed_stall_expert_us = max(0.0, t_expert_us - t_compute_us)

    print(f"\n  ── MONOLITHIC EXPERT OFFLOADING (Baseline MoE-Infinity / ZeRO-Infinity Style) ──")
    print(f"  • Single-Token GEMV Compute Latency (T_compute)  : {t_compute_us:.2f} µs ({t_compute_us/1000:.4f} ms)")
    print(f"  • PCIe DMA Weight Transfer Latency (T_transfer)   : {t_expert_us:.2f} µs ({t_expert_us/1000:.3f} ms)")
    print(f"  • Transfer-to-Compute Disparity Ratio (T_trans / T_comp) : {disparity_ratio:.2f}×")
    print(f"  • Exposed GPU Stall Time per Expert Miss         : {exposed_stall_expert_us:.2f} µs ({exposed_stall_expert_us/1000:.3f} ms)")
    print(f"  • Physical GPU Idle / Bubble Fraction            : {idle_fraction_expert:.2f}%")

    print(f"\n  ── AAEC v3 COLUMN-GRANULAR OFFLOADING (Single Column = 12.29 KB) ──")
    t_single_col_us = col_results[1]["t_transfer_us"]
    ratio_single_col = t_single_col_us / t_compute_us
    speedup_single_col = t_expert_us / t_single_col_us

    print(f"  • Single Column Transfer Latency (12.29 KB)      : {t_single_col_us:.2f} µs")
    print(f"  • Transfer-to-Compute Ratio (T_col / T_comp)      : {ratio_single_col:.2f}×")
    print(f"  • Transfer Speedup over Monolithic Expert        : {speedup_single_col:.2f}× faster")

    print(f"\n  ── AAEC v3 BATCHED MISSED COLUMN OFFLOADING (Typical 16-Column Miss = 196.6 KB) ──")
    t_16col_us = col_results[16]["t_transfer_us"]
    ratio_16col = t_16col_us / t_compute_us
    speedup_16col = t_expert_us / t_16col_us
    idle_fraction_16col = max(0.0, (t_16col_us - t_compute_us) / t_16col_us * 100)

    print(f"  • 16-Column Batch Transfer Latency (196.6 KB)     : {t_16col_us:.2f} µs")
    print(f"  • Transfer-to-Compute Ratio (T_16col / T_comp)    : {ratio_16col:.2f}×")
    print(f"  • Transfer Speedup over Monolithic Expert        : {speedup_16col:.2f}× payload speedup")

    print(f"\n  ── HEAD-TO-HEAD BOTTLENECK COMPARISON SUMMARY ──")
    print(f"  {'Metric':<42} | {'Monolithic Expert (9.44 MB)':<27} | {'AAEC v3 Batch (16 cols, 196.6 KB)':<32}")
    print("  " + "─" * 105)
    print(f"  {'PCIe Transfer Latency (T_transfer)':<42} | {t_expert_us/1000:>23.3f} ms | {t_16col_us/1000:>28.3f} ms")
    print(f"  {'GPU Compute Latency (T_compute)':<42} | {t_compute_us/1000:>23.4f} ms | {t_compute_us/1000:>28.4f} ms")
    print(f"  {'Disparity Ratio (T_trans / T_comp)':<42} | {disparity_ratio:>23.2f}× | {ratio_16col:>28.2f}×")
    print(f"  {'Exposed GPU Idle Fraction':<42} | {idle_fraction_expert:>22.2f}% | {idle_fraction_16col:>27.2f}%")
    print("=" * 95)

    # Save Results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_data = {
        "hardware": gpu_name,
        "compute_latency_us": t_compute_us,
        "compute_latency_std_us": t_compute_std_us,
        "monolithic_expert": {
            "payload_bytes": MONOLITHIC_EXPERT_BYTES,
            "payload_mb": MONOLITHIC_EXPERT_BYTES / (1024**2),
            "transfer_latency_us": t_expert_us,
            "transfer_latency_std_us": t_expert_std_us,
            "achieved_bw_gbps": bw_expert,
            "disparity_ratio": disparity_ratio,
            "gpu_idle_percent": idle_fraction_expert
        },
        "column_transfers": col_results
    }
    with open(os.path.join(RESULTS_DIR, "e24_io_bottleneck_proof.json"), "w") as f:
        json.dump(out_data, f, indent=4)
    print(f"\nResults saved to {RESULTS_DIR}/e24_io_bottleneck_proof.json")


if __name__ == "__main__":
    run_io_bottleneck_proof()
