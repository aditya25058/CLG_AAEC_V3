#!/bin/bash
# Run all fixed evaluation scripts sequentially
set -e

echo "=========================================="
echo "RUNNING ALL FIXED EVALUATION EXPERIMENTS"
echo "=========================================="

echo ""
echo "--- E04: Cache Policy Comparison ---"
python3 /home/palakm/MoEServingSim/evaluation/scripts/e04_cache_policy_comparison.py

echo ""
echo "--- E08: Cold Start Stress Test ---"
python3 /home/palakm/MoEServingSim/evaluation/scripts/e08_stress_tests.py

echo ""
echo "--- E10: Ablation Study ---"
python3 /home/palakm/MoEServingSim/evaluation/scripts/e10_ablation_study.py

echo ""
echo "--- E11: Baseline Comparison ---"
python3 /home/palakm/MoEServingSim/evaluation/scripts/e11_baseline_comparison.py

echo ""
echo "--- E12: Scalability Sweep ---"
python3 /home/palakm/MoEServingSim/evaluation/scripts/e12_scalability.py

echo ""
echo "--- E13: Distributed Expert Serving ---"
python3 /home/palakm/MoEServingSim/evaluation/scripts/e13_distributed_expert_serving.py

echo ""
echo "--- E14: Distributed Prefetcher ---"
python3 /home/palakm/MoEServingSim/evaluation/scripts/e14_distributed_prefetcher.py

echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETED SUCCESSFULLY"
echo "=========================================="
