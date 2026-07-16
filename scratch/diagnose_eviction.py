#!/usr/bin/env python3
"""Quick diagnostic: run a single Qwen3 SAB_AAE scenario with memory tracing
to see if eviction is ever triggered and what the memory pressure looks like."""

import subprocess
import re
import sys

venv_python = "venv/bin/python3"

# Run with SAB_AAE on Qwen3 with verbose logging
cmd = [
    venv_python, "-m", "serving",
    "--cluster-config", "configs/cluster/single_node_qwen3_a22b_h100_low_mem.json",
    "--dtype", "bfloat16",
    "--dataset", "datasets/qwen3_remote_10req_concurrent.jsonl",
    "--expert-routing-policy", "DATASET",
    "--num-reqs", "10",
    "--max-num-seqs", "8",
    "--enable-affinity-batching",
    "--enable-affinity-eviction",
    "--output", "outputs/phase1/diag_qwen3_aae.csv",
    "--log-level", "INFO"  
]

print("Running diagnostic with INFO logging...")
print(f"Command: {' '.join(cmd)}")
res = subprocess.run(cmd, capture_output=True, text=True)

stdout = res.stdout
stderr = res.stderr

# Count eviction events
eviction_count = len(re.findall(r"Eviction of the request", stdout + stderr))
print(f"\n=== EVICTION COUNT: {eviction_count} ===")

# Look for memory-related messages
for line in (stdout + stderr).split('\n'):
    if any(kw in line.lower() for kw in ['evict', 'memory', 'preempt', 'avail']):
        print(f"  {line.strip()}")

print("\n=== FULL STDOUT (last 30 lines) ===")
for line in stdout.strip().split('\n')[-30:]:
    print(f"  {line}")

if stderr.strip():
    print("\n=== STDERR (last 10 lines) ===")
    for line in stderr.strip().split('\n')[-10:]:
        print(f"  {line}")
