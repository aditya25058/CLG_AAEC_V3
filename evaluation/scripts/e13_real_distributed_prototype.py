# evaluation/scripts/e13_real_distributed_prototype.py
"""
AAEC v3 Real Distributed Multi-Node serving Benchmarking Engine (E13)

This script executes a real, end-to-end distributed weight-streaming MoE
inference benchmark using PyTorch Distributed and NCCL/Gloo backends.

It implements:
- E13A: Real distributed network execution across nodes/GPUs.
- E13B: Actual database routing traces and LRU cache.
- E13C: Asynchronous overlap of weight streaming with Phase-1 cached compute
  using custom CUDA streams, dist.irecv (non-blocking), and cudaEvents.

Usage:
  # Master Machine (Node 0, Rank 0 & 1):
  gpurun torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr=<IP> --master_port=29500 evaluation/scripts/e13_real_distributed_prototype.py --model qwen3_30b --cache_size 32
  
  # Worker Machine (Node 1, Rank 2 & 3):
  gpurun torchrun --nproc_per_node=2 --nnodes=2 --node_rank=1 --master_addr=<IP> --master_port=29500 evaluation/scripts/e13_real_distributed_prototype.py --model qwen3_30b --cache_size 32
"""

import os
import sys
import json
import sqlite3
import argparse
import time
import torch
import torch.distributed as dist
import torch.nn.functional as F
from collections import OrderedDict

MODELS = {
    "qwen3_30b": {
        "db_path": "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/qwen3_30b_real_v2.db",
        "num_layers": 48,
        "num_experts": 128,
        "intermediate_dim": 768,
        "hidden_size": 2048,
        "active_experts": 8
    },
    "deepseek_v2_lite": {
        "db_path": "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/deepseek_lite_real.db",
        "num_layers": 27,
        "num_experts": 64,
        "intermediate_dim": 1408,
        "hidden_size": 2048,
        "active_experts": 6
    }
}

def load_db_traces(db_path: str):
    print(f"Loading traces from database: {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT prompt_id, token_pos, layer, expert_id, active_indices, energy_k_50
        FROM activations
        ORDER BY prompt_id, token_pos, layer
    """)
    rows = cursor.fetchall()
    conn.close()
    
    prompt_ids = sorted(list(set(row[0] for row in rows)))
    split_idx = len(prompt_ids) // 2
    eval_prompts = set(prompt_ids[split_idx:])
    
    evaluation_db = {}
    for row in rows:
        p_id, t_pos, layer, exp_id, indices_str, k50 = row
        if p_id not in eval_prompts:
            continue
        indices = json.loads(indices_str)[:k50]
        active_set = set(indices)
        
        if p_id not in evaluation_db:
            evaluation_db[p_id] = {}
        if t_pos not in evaluation_db[p_id]:
            evaluation_db[p_id][t_pos] = {}
            
        evaluation_db[p_id][t_pos][layer] = (exp_id, active_set)
        
    print(f"Traces loaded. Total prompts: {len(evaluation_db)}")
    return evaluation_db

class RealDistributedAAECEngine:
    def __init__(self, rank: int, world_size: int, model_name: str, cache_size: int, backend: str = "nccl"):
        self.rank = rank
        self.world_size = world_size
        self.model_name = model_name
        self.cache_size = cache_size
        self.backend = backend
        
        self.spec = MODELS[model_name]
        self.NL = self.spec["num_layers"]
        self.NE = self.spec["num_experts"]
        self.H = self.spec["hidden_size"]
        self.I = self.spec["intermediate_dim"]
        
        # Experts partition across ranks
        self.experts_per_node = self.NE // world_size
        
        # Local device setup
        self.is_cuda = torch.cuda.is_available()
        self.device = torch.device(f"cuda:0" if self.is_cuda else "cpu")
        print(f"[Rank {rank}] Initializing serving engine on {self.device}...")
        
        # Local expert parameters storage (Node R owns R * experts_per_node ..)
        # Note: We load weights as random matrices (or real weights if available)
        # to focus on evaluation of network transfer and kernel execution.
        self.local_experts = {}
        for idx in range(rank * self.experts_per_node, (rank + 1) * self.experts_per_node):
            self.local_experts[idx] = {
                "gate": torch.randn(self.I, self.H, dtype=torch.bfloat16, device=self.device),
                "up": torch.randn(self.I, self.H, dtype=torch.bfloat16, device=self.device),
                "down": torch.randn(self.H, self.I, dtype=torch.bfloat16, device=self.device)
            }
            
        # Pinned host memory storage for local experts to simulate async PCIe copies
        self.local_experts_cpu = {}
        for idx in range(rank * self.experts_per_node, (rank + 1) * self.experts_per_node):
            self.local_experts_cpu[idx] = {
                "gate": torch.randn(self.I, self.H, dtype=torch.bfloat16).pin_memory(),
                "up": torch.randn(self.I, self.H, dtype=torch.bfloat16).pin_memory(),
                "down": torch.randn(self.H, self.I, dtype=torch.bfloat16).pin_memory()
            }
            
        # Cache Capacity & Specifications
        # Total cached columns is cache_size * NE
        self.gpu_caches = {l: OrderedDict() for l in range(self.NL)}
        self.cache_capacity = cache_size * self.NE
        
        # Memory Footprint Math
        self.column_bytes = 3 * self.H * 2  # gate + up + down columns
        self.cache_bytes = self.cache_capacity * self.column_bytes
        self.total_model_columns = self.NE * self.I
        self.cache_ratio = self.cache_capacity / self.total_model_columns
        
        # Shared CUDA stream for non-blocking communication
        self.comm_stream = torch.cuda.Stream(device=self.device) if self.is_cuda else None
        
        # Command communication buffer
        # [expert_idx, num_misses, is_shutdown]
        self.cmd_tensor = torch.zeros(3, dtype=torch.long, device="cpu" if backend == "gloo" else self.device)

    def run_worker_loop(self):
        """Worker rank loop: listens for slice requests, extracts columns, and streams them."""
        print(f"[Rank {self.rank}] Worker loop started.")
        while True:
            # Receive command from coordinator (Rank 0)
            dist.recv(tensor=self.cmd_tensor, src=0)
            
            # Check shutdown signal using a CPU copy to prevent GPU synchronization stalls
            cmd_cpu = self.cmd_tensor.cpu() if self.backend == "nccl" else self.cmd_tensor
            if cmd_cpu[2] == 1:
                print(f"[Rank {self.rank}] Shutdown command received. Exiting.")
                break
                
            expert_idx = int(cmd_cpu[0])
            num_misses = int(cmd_cpu[1])
            
            if num_misses > 0:
                if self.backend == "nccl":
                    # --- CUDA-AWARE NCCL DIRECT TRANSFER ---
                    m_idx = torch.empty(num_misses, dtype=torch.long, device=self.device)
                    dist.recv(tensor=m_idx, src=0)
                    
                    # Extract slices from local weight store
                    exp_w = self.local_experts[expert_idx]
                    g_send = exp_w["gate"][m_idx]
                    u_send = exp_w["up"][m_idx]
                    d_send = exp_w["down"][:, m_idx].contiguous()
                    
                    # Stream slices back asynchronously using non-blocking dist.isend
                    req_g = dist.isend(tensor=g_send, dst=0, tag=1)
                    req_u = dist.isend(tensor=u_send, dst=0, tag=2)
                    req_d = dist.isend(tensor=d_send, dst=0, tag=3)
                    
                    # Wait for transmission to complete
                    req_g.wait()
                    req_u.wait()
                    req_d.wait()
                else:
                    # --- GLOO CPU BUFFERS TRANSFER ---
                    m_idx_cpu = torch.empty(num_misses, dtype=torch.long, device="cpu")
                    dist.recv(tensor=m_idx_cpu, src=0)
                    
                    # Copy to GPU
                    m_idx = m_idx_cpu.to(self.device)
                    
                    # Extract slices from local weight store
                    exp_w = self.local_experts[expert_idx]
                    g_send = exp_w["gate"][m_idx]
                    u_send = exp_w["up"][m_idx]
                    d_send = exp_w["down"][:, m_idx].contiguous()
                    
                    # Copy send weights to CPU buffers for Gloo transfer
                    g_send_cpu = g_send.cpu()
                    u_send_cpu = u_send.cpu()
                    d_send_cpu = d_send.cpu()
                    
                    # Stream slices back asynchronously using non-blocking dist.isend
                    req_g = dist.isend(tensor=g_send_cpu, dst=0, tag=1)
                    req_u = dist.isend(tensor=u_send_cpu, dst=0, tag=2)
                    req_d = dist.isend(tensor=d_send_cpu, dst=0, tag=3)
                    
                    # Wait for transmission to complete
                    req_g.wait()
                    req_u.wait()
                    req_d.wait()

    def forward_coordinator(self, evaluation_db):
        """Coordinator loop (Rank 0): Replays database traces and measures real latency/stalls with async overlap."""
        print(f"[Rank 0] Starting serving execution loop...")
        print(f"[Rank 0] Cache Footprint: {self.cache_bytes / (1024**2):.2f} MB per layer ({self.cache_ratio*100:.2f}% of model FFN layer)")
        print(f"[Rank 0] Total Cache Footprint across all {self.NL} layers: {self.cache_bytes * self.NL / (1024**3):.3f} GB")
        
        total_tokens = 0
        total_network_bytes = 0
        
        # Timing registers
        total_comp_time_ms = 0.0
        total_comm_time_ms = 0.0
        total_overlap_time_ms = 0.0
        
        eval_prompt_ids = sorted(evaluation_db.keys())
        
        # Token activations placeholder (B=1)
        x = torch.randn(1, self.H, dtype=torch.bfloat16, device=self.device)
        
        # CUDA Events for fine-grained timing overlap (if CUDA available)
        if self.is_cuda:
            step_start = torch.cuda.Event(enable_timing=True)
            step_end = torch.cuda.Event(enable_timing=True)
        
        for p_id in eval_prompt_ids:
            t_positions = sorted(evaluation_db[p_id].keys())
            
            # Reset cache on prompt boundary
            for l in range(self.NL):
                self.gpu_caches[l].clear()
                
            for t in t_positions:
                total_tokens += 1
                
                # List of segment timings to process at the end of the token step
                segments = []
                
                # Start token step timer
                if self.is_cuda:
                    step_start.record()
                else:
                    t0_step = time.perf_counter()
                
                for l in range(self.NL):
                    if l not in evaluation_db[p_id][t]:
                        continue
                    exp_id, active_cols = evaluation_db[p_id][t][l]
                    
                    owner_rank = exp_id // self.experts_per_node
                    is_local = (owner_rank == 0)
                    
                    cache = self.gpu_caches[l]
                    active_keys = {(exp_id, col) for col in active_cols}
                    local_active = active_keys.intersection(cache.keys())
                    missed_keys = active_keys - local_active
                    
                    # Update cache access order
                    for key in active_keys:
                        if key in cache:
                            cache.move_to_end(key)
                        else:
                            if len(cache) >= self.cache_capacity:
                                cache.popitem(last=False)
                            cache[key] = True
                            
                    # Local cache slices (simulated statically for Phase 1 compute)
                    W_g_c = torch.randn(self.cache_size, self.H, dtype=torch.bfloat16, device=self.device)
                    W_u_c = torch.randn(self.cache_size, self.H, dtype=torch.bfloat16, device=self.device)
                    W_d_c = torch.randn(self.H, self.cache_size, dtype=torch.bfloat16, device=self.device)
                    
                    # Handle weight loading with async overlap
                    if missed_keys:
                        num_misses = len(missed_keys)
                        miss_cols = sorted([col for (_, col) in missed_keys])
                        m_idx = torch.tensor(miss_cols, dtype=torch.long, device=self.device)
                        
                        # Byte calculation: 3 tensors (gate, up, down)
                        total_bytes = 3 * num_misses * self.H * 2
                        
                        gate_slice = torch.empty(num_misses, self.H, dtype=torch.bfloat16, device=self.device)
                        up_slice = torch.empty(num_misses, self.H, dtype=torch.bfloat16, device=self.device)
                        down_slice = torch.empty(self.H, num_misses, dtype=torch.bfloat16, device=self.device)
                        
                        if is_local:
                            # --- LOCAL ACCESS: ASYNC PCIe COPY ---
                            m_idx_cpu = m_idx.cpu()
                            if self.is_cuda:
                                c_start = torch.cuda.Event(enable_timing=True)
                                c_end = torch.cuda.Event(enable_timing=True)
                                p_start = torch.cuda.Event(enable_timing=True)
                                p_end = torch.cuda.Event(enable_timing=True)
                                
                                # Step 1: Start async memory copy on the communication stream
                                c_start.record()
                                with torch.cuda.stream(self.comm_stream):
                                    gate_slice.copy_(self.local_experts_cpu[exp_id]["gate"][m_idx_cpu], non_blocking=True)
                                    up_slice.copy_(self.local_experts_cpu[exp_id]["up"][m_idx_cpu], non_blocking=True)
                                    down_slice.copy_(self.local_experts_cpu[exp_id]["down"][:, m_idx_cpu], non_blocking=True)
                                    c_end.record(self.comm_stream)
                                    
                                # Step 2: Run Phase 1 compute concurrently on the main stream
                                p_start.record()
                                g_c = torch.matmul(x, W_g_c.t())
                                u_c = torch.matmul(x, W_u_c.t())
                                y_c = torch.matmul(F.silu(g_c) * u_c, W_d_c.t())
                                p_end.record()
                                
                                # Step 3: Synchronize streams before Phase 2
                                torch.cuda.current_stream().wait_stream(self.comm_stream)
                                
                                # Step 4: Run Phase 2 compute
                                g_m = torch.matmul(x, gate_slice.t())
                                u_m = torch.matmul(x, up_slice.t())
                                y_m = torch.matmul(F.silu(g_m) * u_m, down_slice.t())
                                y = y_c + y_m
                                
                                segments.append(("comm", c_start, c_end))
                                segments.append(("comp", p_start, p_end))
                            else:
                                t0 = time.perf_counter()
                                gate_slice.copy_(self.local_experts_cpu[exp_id]["gate"][m_idx_cpu])
                                up_slice.copy_(self.local_experts_cpu[exp_id]["up"][m_idx_cpu])
                                down_slice.copy_(self.local_experts_cpu[exp_id]["down"][:, m_idx_cpu])
                                t_c = (time.perf_counter() - t0) * 1000.0
                                
                                t1 = time.perf_counter()
                                g_c = torch.matmul(x, W_g_c.t())
                                u_c = torch.matmul(x, W_u_c.t())
                                y_c = torch.matmul(F.silu(g_c) * u_c, W_d_c.t())
                                t_cp = (time.perf_counter() - t1) * 1000.0
                                
                                g_m = torch.matmul(x, gate_slice.t())
                                u_m = torch.matmul(x, up_slice.t())
                                y_m = torch.matmul(F.silu(g_m) * u_m, down_slice.t())
                                y = y_c + y_m
                                
                                total_comm_time_ms += t_c
                                total_comp_time_ms += t_cp
                            
                        else:
                            # --- REMOTE ACCESS: TRUE ASYNC STREAMING ---
                            total_network_bytes += total_bytes
                            
                            if self.backend == "nccl" and self.is_cuda:
                                c_start = torch.cuda.Event(enable_timing=True)
                                c_end = torch.cuda.Event(enable_timing=True)
                                p_start = torch.cuda.Event(enable_timing=True)
                                p_end = torch.cuda.Event(enable_timing=True)
                                
                                # Step 1: Start async memory copy/request on comm stream
                                c_start.record()
                                with torch.cuda.stream(self.comm_stream):
                                    # Send command & indices asynchronously on GPU using single copy
                                    cmd_cpu = torch.tensor([exp_id, num_misses, 0], dtype=torch.long, device="cpu")
                                    self.cmd_tensor.copy_(cmd_cpu)
                                    
                                    req_cmd = dist.isend(self.cmd_tensor, dst=owner_rank)
                                    req_idx = dist.isend(m_idx, dst=owner_rank)
                                    
                                    # Receive weight slices asynchronously on GPU
                                    req_g = dist.irecv(tensor=gate_slice, src=owner_rank, tag=1)
                                    req_u = dist.irecv(tensor=up_slice, src=owner_rank, tag=2)
                                    req_d = dist.irecv(tensor=down_slice, src=owner_rank, tag=3)
                                    
                                    c_end.record(self.comm_stream)
                                    
                                # Step 2: Run Phase 1 compute concurrently on the main stream
                                p_start.record()
                                g_c = torch.matmul(x, W_g_c.t())
                                u_c = torch.matmul(x, W_u_c.t())
                                y_c = torch.matmul(F.silu(g_c) * u_c, W_d_c.t())
                                p_end.record()
                                
                                # Step 3: Wait for communication requests and stream to complete
                                req_cmd.wait()
                                req_idx.wait()
                                req_g.wait()
                                req_u.wait()
                                req_d.wait()
                                torch.cuda.current_stream().wait_stream(self.comm_stream)
                                
                                # Step 4: Run Phase 2 compute
                                g_m = torch.matmul(x, gate_slice.t())
                                u_m = torch.matmul(x, up_slice.t())
                                y_m = torch.matmul(F.silu(g_m) * u_m, down_slice.t())
                                y = y_c + y_m
                                
                                segments.append(("comm", c_start, c_end))
                                segments.append(("comp", p_start, p_end))
                            else:
                                # --- GLOO OR NON-CUDA FALLBACK: CPU STAGING ---
                                if self.is_cuda:
                                    c_start = torch.cuda.Event(enable_timing=True)
                                    c_end = torch.cuda.Event(enable_timing=True)
                                    p_start = torch.cuda.Event(enable_timing=True)
                                    p_end = torch.cuda.Event(enable_timing=True)
                                    
                                    c_start.record()
                                    
                                    # Send command & indices asynchronously on CPU
                                    self.cmd_tensor[0] = exp_id
                                    self.cmd_tensor[1] = num_misses
                                    self.cmd_tensor[2] = 0
                                    
                                    m_idx_cpu = m_idx.cpu()
                                    gate_slice_cpu = torch.empty(num_misses, self.H, dtype=torch.bfloat16, device="cpu")
                                    up_slice_cpu = torch.empty(num_misses, self.H, dtype=torch.bfloat16, device="cpu")
                                    down_slice_cpu = torch.empty(self.H, num_misses, dtype=torch.bfloat16, device="cpu")
                                    
                                    req_cmd = dist.isend(self.cmd_tensor, dst=owner_rank)
                                    req_idx = dist.isend(m_idx_cpu, dst=owner_rank)
                                    
                                    # Receive weight slices asynchronously on CPU
                                    req_g = dist.irecv(tensor=gate_slice_cpu, src=owner_rank, tag=1)
                                    req_u = dist.irecv(tensor=up_slice_cpu, src=owner_rank, tag=2)
                                    req_d = dist.irecv(tensor=down_slice_cpu, src=owner_rank, tag=3)
                                    
                                    # Step 2: Run Phase 1 compute concurrently on the main stream
                                    p_start.record()
                                    g_c = torch.matmul(x, W_g_c.t())
                                    u_c = torch.matmul(x, W_u_c.t())
                                    y_c = torch.matmul(F.silu(g_c) * u_c, W_d_c.t())
                                    p_end.record()
                                    
                                    # Step 3: Wait for communication requests and stream to complete
                                    req_cmd.wait()
                                    req_idx.wait()
                                    req_g.wait()
                                    req_u.wait()
                                    req_d.wait()
                                    
                                    # Copy received CPU slices to GPU
                                    gate_slice.copy_(gate_slice_cpu, non_blocking=True)
                                    up_slice.copy_(up_slice_cpu, non_blocking=True)
                                    down_slice.copy_(down_slice_cpu, non_blocking=True)
                                    
                                    # Step 4: Run Phase 2 compute
                                    g_m = torch.matmul(x, gate_slice.t())
                                    u_m = torch.matmul(x, up_slice.t())
                                    y_m = torch.matmul(F.silu(g_m) * u_m, down_slice.t())
                                    y = y_c + y_m
                                    
                                    c_end.record()
                                    segments.append(("comm", c_start, c_end))
                                    segments.append(("comp", p_start, p_end))
                                else:
                                    t0 = time.perf_counter()
                                    self.cmd_tensor[0] = exp_id
                                    self.cmd_tensor[1] = num_misses
                                    self.cmd_tensor[2] = 0
                                    
                                    m_idx_cpu = m_idx.cpu()
                                    gate_slice_cpu = torch.empty(num_misses, self.H, dtype=torch.bfloat16, device="cpu")
                                    up_slice_cpu = torch.empty(num_misses, self.H, dtype=torch.bfloat16, device="cpu")
                                    down_slice_cpu = torch.empty(self.H, num_misses, dtype=torch.bfloat16, device="cpu")
                                    
                                    dist.send(self.cmd_tensor, dst=owner_rank)
                                    dist.send(m_idx_cpu, dst=owner_rank)
                                    dist.recv(tensor=gate_slice_cpu, src=owner_rank, tag=1)
                                    dist.recv(tensor=up_slice_cpu, src=owner_rank, tag=2)
                                    dist.recv(tensor=down_slice_cpu, src=owner_rank, tag=3)
                                    
                                    gate_slice.copy_(gate_slice_cpu)
                                    up_slice.copy_(up_slice_cpu)
                                    down_slice.copy_(down_slice_cpu)
                                    t_c = (time.perf_counter() - t0) * 1000.0
                                    
                                    t1 = time.perf_counter()
                                    g_c = torch.matmul(x, W_g_c.t())
                                    u_c = torch.matmul(x, W_u_c.t())
                                    y_c = torch.matmul(F.silu(g_c) * u_c, W_d_c.t())
                                    t_cp = (time.perf_counter() - t1) * 1000.0
                                    
                                    g_m = torch.matmul(x, gate_slice.t())
                                    u_m = torch.matmul(x, up_slice.t())
                                    y_m = torch.matmul(F.silu(g_m) * u_m, down_slice.t())
                                    y = y_c + y_m
                                    
                                    total_comm_time_ms += t_c
                                    total_comp_time_ms += t_cp
                            
                    else:
                        # Full Cache Hit: Compute cached portion only
                        if self.is_cuda:
                            p_start = torch.cuda.Event(enable_timing=True)
                            p_end = torch.cuda.Event(enable_timing=True)
                            
                            p_start.record()
                            g_c = torch.matmul(x, W_g_c.t())
                            u_c = torch.matmul(x, W_u_c.t())
                            y = torch.matmul(F.silu(g_c) * u_c, W_d_c.t())
                            p_end.record()
                            
                            segments.append(("comp", p_start, p_end))
                        else:
                            t1 = time.perf_counter()
                            g_c = torch.matmul(x, W_g_c.t())
                            u_c = torch.matmul(x, W_u_c.t())
                            y = torch.matmul(F.silu(g_c) * u_c, W_d_c.t())
                            total_comp_time_ms += (time.perf_counter() - t1) * 1000.0
                        
                # End step timer
                if self.is_cuda:
                    step_end.record()
                    torch.cuda.synchronize(self.device)
                    total_overlap_time_ms += step_start.elapsed_time(step_end)
                    
                    # Accumulate fine-grained timing segments without per-layer synchronization
                    for cat, start_ev, end_ev in segments:
                        ev_time = start_ev.elapsed_time(end_ev)
                        if cat == "comm":
                            total_comm_time_ms += ev_time
                        else:
                            total_comp_time_ms += ev_time
                else:
                    total_overlap_time_ms += (time.perf_counter() - t0_step) * 1000.0
                
        # Shutdown worker ranks
        print("[Rank 0] Shutting down worker processes...")
        for r in range(1, self.world_size):
            if self.backend == "nccl":
                cmd_cpu = torch.tensor([0, 0, 1], dtype=torch.long, device="cpu")
                self.cmd_tensor.copy_(cmd_cpu)
                dist.send(self.cmd_tensor, dst=r)
            else:
                self.cmd_tensor[0] = 0
                self.cmd_tensor[1] = 0
                self.cmd_tensor[2] = 1  # Shutdown signal
                dist.send(self.cmd_tensor, dst=r)
            
        # Results summary
        net_gb = total_network_bytes / 1e9
        avg_comp_ms = total_comp_time_ms / max(1, total_tokens)
        avg_comm_ms = total_comm_time_ms / max(1, total_tokens)
        avg_step_ms = total_overlap_time_ms / max(1, total_tokens)
        tps = 1000.0 / avg_step_ms
        
        # Calculate hidden latency ratio
        avg_hidden_ms = max(0.0, (avg_comm_ms + avg_comp_ms) - avg_step_ms)
        hiding_ratio = avg_hidden_ms / max(1e-6, avg_comm_ms)
        
        print("\n" + "="*60)
        print(f"Distributed serving benchmark completed for {self.model_name}!")
        print(f"  Total tokens processed: {total_tokens}")
        print(f"  Total network bytes transferred: {net_gb:.4f} GB")
        print(f"  Average compute latency per token: {avg_comp_ms:.4f} ms")
        print(f"  Average Remote Weight Fetch Time:  {avg_comm_ms:.4f} ms")
        print(f"  Average overlapped step latency:   {avg_step_ms:.4f} ms")
        print(f"  Average latency hidden by overlap: {avg_hidden_ms:.4f} ms ({hiding_ratio*100:.1f}%)")
        print(f"  Real serving throughput: {tps:.2f} tokens/sec")
        print("="*60 + "\n")
        
        return {
            "network_gb": net_gb,
            "avg_compute_ms": avg_comp_ms,
            "avg_comm_ms": avg_comm_ms,
            "avg_step_ms": avg_step_ms,
            "avg_hidden_ms": avg_hidden_ms,
            "hiding_ratio": hiding_ratio,
            "throughput_tps": tps
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3_30b", choices=["qwen3_30b", "deepseek_v2_lite"])
    parser.add_argument("--cache_size", type=int, default=32, help="Columns per expert cached locally")
    parser.add_argument("--rank", type=int, default=0, help="Process rank")
    parser.add_argument("--world_size", type=int, default=4, help="World size (ranks count)")
    parser.add_argument("--master_addr", type=str, default="127.0.0.1", help="Master IP")
    parser.add_argument("--master_port", type=str, default="29500", help="Master Port")
    parser.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo"], help="Torch distributed backend")
    args = parser.parse_args()
    
    # Initialize torch.distributed process group
    is_cuda = torch.cuda.is_available()
    backend = args.backend if is_cuda or args.backend == "gloo" else "gloo"
    
    # Check env variables if run via torchrun
    env_rank = os.environ.get("RANK")
    env_world_size = os.environ.get("WORLD_SIZE")
    env_master_addr = os.environ.get("MASTER_ADDR")
    env_master_port = os.environ.get("MASTER_PORT")
    
    rank = int(env_rank) if env_rank is not None else args.rank
    world_size = int(env_world_size) if env_world_size is not None else args.world_size
    master_addr = env_master_addr if env_master_addr is not None else args.master_addr
    master_port = env_master_port if env_master_port is not None else args.master_port
    
    dist.init_process_group(
        backend=backend,
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size
    )
    
    engine = RealDistributedAAECEngine(
        rank=rank,
        world_size=world_size,
        model_name=args.model,
        cache_size=args.cache_size,
        backend=backend
    )
    
    if rank == 0:
        eval_db = load_db_traces(engine.spec["db_path"])
        stats = engine.forward_coordinator(eval_db)
        
        # Save real serving output report
        out_dir = f"/home/palakm/MoEServingSim/evaluation/results/e13_distributed/{args.model}"
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "real_distributed_report.json"), "w") as f:
            json.dump(stats, f, indent=4)
    else:
        engine.run_worker_loop()
        
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
