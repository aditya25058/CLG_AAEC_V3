#!/bin/bash
# evaluation/run_remaining.sh
# Runs only E05 to E14 and generates all plots
set -e

BASE_DIR="/home/palakm/MoEServingSim/evaluation"

echo "[6/11] Running E05 End-to-End Latency Benchmarks..."
python3 -u ${BASE_DIR}/scripts/e05_e2e_latency.py

echo "[7/11] Running E06 Transfer Overlap Bandwidth Profiling..."
python3 -u ${BASE_DIR}/scripts/e06_bandwidth_utilization.py

echo "[8/11] Running E07 Kernel execution timing benchmarks..."
python3 -u ${BASE_DIR}/scripts/e07_kernel_benchmarks.py

echo "[9/11] Running E08 Causal Predictor Stress Tests..."
python3 -u ${BASE_DIR}/scripts/e08_stress_tests.py

echo "[10/11] Running E09 Quality Benchmarks (lm-eval checks)..."
python3 -u ${BASE_DIR}/scripts/e09_quality_benchmarks.py

echo "[11/11] Running Ablation, E11, E12, E13, E14, E15, & E16 sweeps..."
python3 -u ${BASE_DIR}/scripts/e10_ablation_study.py
python3 -u ${BASE_DIR}/scripts/e11_baseline_comparison.py
python3 -u ${BASE_DIR}/scripts/e12_scalability.py
python3 -u ${BASE_DIR}/scripts/e13_distributed_expert_serving.py
python3 -u ${BASE_DIR}/scripts/e14_distributed_prefetcher.py
python3 -u ${BASE_DIR}/scripts/e15_batch_scaling_tradeoffs.py
python3 -u ${BASE_DIR}/scripts/e16_physical_io_benchmark.py

echo "Generating all plots..."
python3 -u ${BASE_DIR}/scripts/generate_all_plots.py

echo "Remaining evaluations completed successfully!"
