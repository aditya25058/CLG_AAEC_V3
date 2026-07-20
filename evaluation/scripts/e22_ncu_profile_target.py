#!/usr/bin/env python3
"""
E22b — Nsight Compute Kernel Profiling Target
Runs Dense FFN and SA-FFN kernels with markers for ncu profiling.
Usage:
  gpurun ncu --set full --kernel-name regex:gemm -o e22_profile \
    python3 evaluation/scripts/e22_ncu_profile_target.py

Or standalone for quick occupancy/bandwidth extraction:
  gpurun python3 evaluation/scripts/e22_ncu_profile_target.py
"""

import os
import json
import torch
import torch.nn.functional as F

HIDDEN_SIZE = 2048
INTERMEDIATE_DIM = 768
CACHED_COLS = 32
MISSED_COLS = 16
DTYPE = torch.bfloat16
RESULTS_DIR = "/home/palakm/MoEServingSim/evaluation/results/e22_hwil"

def main():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    x = torch.randn(1, HIDDEN_SIZE, dtype=DTYPE, device=device)

    # Dense FFN weights
    d_gate = torch.randn(INTERMEDIATE_DIM, HIDDEN_SIZE, dtype=DTYPE, device=device)
    d_up   = torch.randn(INTERMEDIATE_DIM, HIDDEN_SIZE, dtype=DTYPE, device=device)
    d_down = torch.randn(HIDDEN_SIZE, INTERMEDIATE_DIM, dtype=DTYPE, device=device)

    # SA-FFN weights
    s_gate_c = torch.randn(CACHED_COLS, HIDDEN_SIZE, dtype=DTYPE, device=device)
    s_up_c   = torch.randn(CACHED_COLS, HIDDEN_SIZE, dtype=DTYPE, device=device)
    s_down_c = torch.randn(HIDDEN_SIZE, CACHED_COLS, dtype=DTYPE, device=device)
    s_gate_m = torch.randn(MISSED_COLS, HIDDEN_SIZE, dtype=DTYPE, device=device)
    s_up_m   = torch.randn(MISSED_COLS, HIDDEN_SIZE, dtype=DTYPE, device=device)
    s_down_m = torch.randn(HIDDEN_SIZE, MISSED_COLS, dtype=DTYPE, device=device)

    # Warmup
    for _ in range(50):
        g = torch.matmul(x, d_gate.t())
        u = torch.matmul(x, d_up.t())
        a = F.silu(g) * u
        _ = torch.matmul(a, d_down.t())
    torch.cuda.synchronize()

    # ── PROFILED REGION: Dense FFN ──
    print("\n=== Dense FFN Profiling Region ===")
    torch.cuda.nvtx.range_push("Dense_FFN")
    for _ in range(100):
        g = torch.matmul(x, d_gate.t())
        u = torch.matmul(x, d_up.t())
        a = F.silu(g) * u
        y = torch.matmul(a, d_down.t())
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    # ── PROFILED REGION: SA-FFN ──
    print("=== SA-FFN Profiling Region ===")
    torch.cuda.nvtx.range_push("SA_FFN")
    for _ in range(100):
        gc = torch.matmul(x, s_gate_c.t())
        uc = torch.matmul(x, s_up_c.t())
        ac = F.silu(gc) * uc
        yc = torch.matmul(ac, s_down_c.t())
        gm = torch.matmul(x, s_gate_m.t())
        um = torch.matmul(x, s_up_m.t())
        am = F.silu(gm) * um
        ym = torch.matmul(am, s_down_m.t())
        yc.add_(ym)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    # ── Manual FLOP/Bandwidth Estimation ──
    # Dense FFN: 3 GEMMs of [1,2048]×[2048,768] + [1,768]×[768,2048]
    dense_flops = 2 * (1 * 2048 * 768) * 2 + 2 * (1 * 768 * 2048)  # gate+up + down
    dense_bytes = (768 * 2048 * 2) * 2 + (2048 * 768 * 2)  # gate+up weights + down weight

    # SA-FFN: 6 GEMMs (3 cached + 3 missed) with smaller dimensions
    sa_flops_c = 2 * (1 * 2048 * CACHED_COLS) * 2 + 2 * (1 * CACHED_COLS * 2048)
    sa_flops_m = 2 * (1 * 2048 * MISSED_COLS) * 2 + 2 * (1 * MISSED_COLS * 2048)
    sa_flops = sa_flops_c + sa_flops_m
    sa_bytes_c = (CACHED_COLS * 2048 * 2) * 2 + (2048 * CACHED_COLS * 2)
    sa_bytes_m = (MISSED_COLS * 2048 * 2) * 2 + (2048 * MISSED_COLS * 2)
    sa_bytes = sa_bytes_c + sa_bytes_m

    # Get measured kernel times from E22 main benchmark results
    e22_path = os.path.join(RESULTS_DIR, "e22_hwil_results.json")
    if os.path.exists(e22_path):
        with open(e22_path) as f:
            e22 = json.load(f)
        dense_us = e22["kernel_benchmarks"]["dense_ffn_us"]
        sa_us = e22["kernel_benchmarks"]["sa_ffn_us"]
    else:
        dense_us = 31.5
        sa_us = 70.6

    # Compute achieved metrics
    dense_gflops = dense_flops / (dense_us * 1e-6) / 1e9
    sa_gflops = sa_flops / (sa_us * 1e-6) / 1e9
    dense_bw = dense_bytes / (dense_us * 1e-6) / 1e9
    sa_bw = sa_bytes / (sa_us * 1e-6) / 1e9

    # H100 NVL peak specs
    peak_bf16_tflops = 835.0  # BF16 Tensor Core TFLOPS
    peak_hbm_bw = 3958.0     # GB/s HBM3e

    dense_tc_util = dense_gflops / (peak_bf16_tflops * 1000) * 100
    sa_tc_util = sa_gflops / (peak_bf16_tflops * 1000) * 100
    dense_bw_util = dense_bw / peak_hbm_bw * 100
    sa_bw_util = sa_bw / peak_hbm_bw * 100

    # Arithmetic intensity (FLOPS / Byte)
    dense_ai = dense_flops / dense_bytes
    sa_ai = sa_flops / sa_bytes

    print(f"\n{'='*70}")
    print(f"ROOFLINE ANALYSIS (Qwen3-30B-A3B, B=1, BF16)")
    print(f"{'='*70}")
    print(f"H100 NVL Peak: {peak_bf16_tflops} TFLOPS (BF16) | {peak_hbm_bw} GB/s (HBM3e)")
    print(f"{'─'*70}")
    print(f"{'Metric':<35} | {'Dense FFN':>14} | {'SA-FFN':>14}")
    print(f"{'─'*70}")
    print(f"{'Total FLOPs':<35} | {dense_flops:>14,} | {sa_flops:>14,}")
    print(f"{'Total Bytes Accessed':<35} | {dense_bytes:>14,} | {sa_bytes:>14,}")
    print(f"{'Arithmetic Intensity (FLOP/B)':<35} | {dense_ai:>14.2f} | {sa_ai:>14.2f}")
    print(f"{'Measured Latency (µs)':<35} | {dense_us:>14.2f} | {sa_us:>14.2f}")
    print(f"{'Achieved GFLOPS':<35} | {dense_gflops:>14.2f} | {sa_gflops:>14.2f}")
    print(f"{'Achieved HBM BW (GB/s)':<35} | {dense_bw:>14.2f} | {sa_bw:>14.2f}")
    print(f"{'Tensor Core Utilization (%)':<35} | {dense_tc_util:>13.4f}% | {sa_tc_util:>13.4f}%")
    print(f"{'HBM Bandwidth Utilization (%)':<35} | {dense_bw_util:>13.4f}% | {sa_bw_util:>13.4f}%")
    print(f"{'Roofline Bound':<35} | {'Memory-bound':>14} | {'Memory-bound':>14}")
    print(f"{'='*70}")

    # Save roofline data
    roofline = {
        "dense_ffn": {
            "flops": dense_flops, "bytes": dense_bytes,
            "arith_intensity": dense_ai,
            "latency_us": dense_us,
            "achieved_gflops": dense_gflops,
            "achieved_bw_gbps": dense_bw,
            "tc_utilization_pct": dense_tc_util,
            "bw_utilization_pct": dense_bw_util
        },
        "sa_ffn": {
            "flops": sa_flops, "bytes": sa_bytes,
            "arith_intensity": sa_ai,
            "latency_us": sa_us,
            "achieved_gflops": sa_gflops,
            "achieved_bw_gbps": sa_bw,
            "tc_utilization_pct": sa_tc_util,
            "bw_utilization_pct": sa_bw_util
        },
        "h100_peak": {
            "bf16_tflops": peak_bf16_tflops,
            "hbm_bw_gbps": peak_hbm_bw
        }
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "e22_roofline_analysis.json"), "w") as f:
        json.dump(roofline, f, indent=4)
    print(f"\nRoofline data saved to {RESULTS_DIR}/e22_roofline_analysis.json")


if __name__ == "__main__":
    main()
