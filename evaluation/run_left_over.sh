#!/bin/bash
# Run only the remaining evaluation scripts sequentially
set -e

echo "=========================================="
echo "RUNNING LEFT-OVER EVALUATION EXPERIMENTS"
echo "=========================================="

echo ""
echo ""
echo "--- E13: Distributed Expert Serving ---"
python3 /home/palakm/MoEServingSim/evaluation/scripts/e13_distributed_expert_serving.py

echo ""
echo "--- E14: Distributed Prefetcher ---"
python3 /home/palakm/MoEServingSim/evaluation/scripts/e14_distributed_prefetcher.py

echo ""
echo "=========================================="
echo "ALL LEFT-OVER EXPERIMENTS COMPLETED"
echo "=========================================="
