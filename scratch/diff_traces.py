import sys
import os
import json

# Adjust sys.path to import serving
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from serving.core.trace_generator import generate_trace
from serving.core.config_builder import build_cluster_config
from serving.core.request import Batch, Request

def main():
    # Mock requests that have already finished prefill (num_computed_tokens = 16)
    req1 = Request(0, "Qwen/Qwen3-235B-A22B", 16, 128, 0, 0)
    req1.num_computed_tokens = 16
    req2 = Request(1, "Qwen/Qwen3-235B-A22B", 16, 128, 0, 0)
    req2.num_computed_tokens = 16
    
    # Mock routing maps for the requests
    # In decode phase, each request routes 1 token to some expert
    req1.moe_profile = {"routing_map": {f"layer_{i}": [0] for i in range(94)}}
    req2.moe_profile = {"routing_map": {f"layer_{i}": [1] for i in range(94)}}
    
    # Decode phase batch
    # __init__(batch_id, model, total_len, kv_len, q_list, k_list, num_prefill, num_decode, prefill_q_list, prefill_k_list, decode_k_list, batch_time, kv_size, evict=0, load=0)
    batch0 = Batch(
        batch_id=1,
        model="Qwen/Qwen3-235B-A22B",
        total_len=2,
        kv_len=34,
        q_list=[1, 1],
        k_list=[17, 17],
        num_prefill=0,
        num_decode=2,
        prefill_q_list=[],
        prefill_k_list=[],
        decode_k_list=[17, 17],
        batch_time=0,
        kv_size=0,
        evict=0,
        load=0
    )
    batch0.requests = [req1, req2]
    batch0.scheduled_tokens = {0: 1, 1: 1}
    
    # Mock requests for instance 1 (different routing)
    req3 = Request(2, "Qwen/Qwen3-235B-A22B", 16, 128, 0, 1)
    req3.num_computed_tokens = 16
    req4 = Request(3, "Qwen/Qwen3-235B-A22B", 16, 128, 0, 1)
    req4.num_computed_tokens = 16
    
    req3.moe_profile = {"routing_map": {f"layer_{i}": [2] for i in range(94)}}
    req4.moe_profile = {"routing_map": {f"layer_{i}": [3] for i in range(94)}}
    
    batch1 = Batch(
        batch_id=1,
        model="Qwen/Qwen3-235B-A22B",
        total_len=2,
        kv_len=34,
        q_list=[1, 1],
        k_list=[17, 17],
        num_prefill=0,
        num_decode=2,
        prefill_q_list=[],
        prefill_k_list=[],
        decode_k_list=[17, 17],
        batch_time=0,
        kv_size=0,
        evict=0,
        load=0
    )
    batch1.requests = [req3, req4]
    batch1.scheduled_tokens = {2: 1, 3: 1}
    
    os.chdir("astra-sim")
    # Build a cluster config
    cluster = build_cluster_config(
        astra_sim=".",
        cluster_config_path="configs/cluster/test_dual_node_tp2_ep4.json",
        enable_twr=False
    )
    
    inst0 = cluster["instances"][0]
    inst1 = cluster["instances"][1]
    
    # Generate TWR trace for instance 0
    os.environ["SIM_RUN_ID"] = "diff_inst0_dec"
    generate_trace(
        batch0, inst0["hardware"], inst0["tp_size"], inst0["pp_size"],
        inst0["local_ep"], inst0["ep_total"], inst0["pd_type"],
        node_id=0, instance_id=0,
        placement=cluster["placement"][0],
        tp_dim=inst0.get("tp_dim"), ep_dim=inst0.get("ep_dim"),
        dp_sum_total_len=2, enable_block_copy=False,
        expert_routing_policy="DATASET",
        enable_epeg=True,
        enable_twr=True
    )
    
    # Generate TWR trace for instance 1
    os.environ["SIM_RUN_ID"] = "diff_inst1_dec"
    generate_trace(
        batch1, inst1["hardware"], inst1["tp_size"], inst1["pp_size"],
        inst1["local_ep"], inst1["ep_total"], inst1["pd_type"],
        node_id=1, instance_id=1,
        placement=cluster["placement"][1],
        tp_dim=inst1.get("tp_dim"), ep_dim=inst1.get("ep_dim"),
        dp_sum_total_len=2, enable_block_copy=False,
        expert_routing_policy="DATASET",
        enable_epeg=True,
        enable_twr=True
    )
    
    print("Traces generated successfully!")
    
    inst0_file = f"inputs/trace/{inst0['hardware']}/{batch0.model}/instance0_batch1_diff_inst0_dec.txt"
    inst1_file = f"inputs/trace/{inst1['hardware']}/{batch1.model}/instance1_batch1_diff_inst1_dec.txt"
    
    with open(inst0_file, 'r') as f:
        inst0_lines = f.readlines()
    with open(inst1_file, 'r') as f:
        inst1_lines = f.readlines()
        
    print(f"Inst0 lines: {len(inst0_lines)}, Inst1 lines: {len(inst1_lines)}")
    
    diff_count = 0
    for i, (l0, l1) in enumerate(zip(inst0_lines, inst1_lines)):
        if "ALLTOALL" in l0 or "ALLTOALL" in l1:
            if l0.strip() != l1.strip():
                print(f"Line {i+1} mismatch:")
                print(f"  Inst0: {l0.strip()}")
                print(f"  Inst1: {l1.strip()}")
                diff_count += 1
                if diff_count >= 20:
                    break

if __name__ == "__main__":
    main()
