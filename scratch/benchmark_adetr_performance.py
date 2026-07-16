import os
import json
import sqlite3
import torch
import numpy as np
import time

DB_PATH = "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/qwen3_30b_real_v2.db"
PERM_PATH = "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/scratch/adetr_permutations.json"
RESULTS_PATH = "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/scratch/adetr_real_hardware_results.json"

def count_contiguous_blocks(indices):
    if not indices:
        return 0
    sorted_indices = sorted(list(indices))
    blocks = 0
    prev = -2
    for idx in sorted_indices:
        if idx != prev + 1:
            blocks += 1
        prev = idx
    return blocks

def get_contiguous_slices(indices):
    """
    Returns list of (start, end) tuples representing contiguous blocks of indices.
    """
    if not indices:
        return []
    sorted_indices = sorted(list(indices))
    slices = []
    start = sorted_indices[0]
    prev = start
    for idx in sorted_indices[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            slices.append((start, prev + 1))
            start = idx
            prev = idx
    slices.append((start, prev + 1))
    return slices

def run_adetr_evaluation():
    print("=== ADETR Evaluation & Physical Hardware Benchmarking ===")
    
    # 1. Load permutations
    if not os.path.exists(PERM_PATH):
        print(f"Error: Permutation file {PERM_PATH} not found. Please run compute_adetr_clusters.py first.")
        return
        
    with open(PERM_PATH, "r") as f:
        permutations = json.load(f)
    print(f"Loaded permutations for {len(permutations)} expert configurations.")
    
    # 2. Connect to database to replay real activation traces
    print("Connecting to database to retrieve activation traces...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT layer, expert_id, active_indices, energy_k_50, energy_k_70, energy_k_90 FROM activations")
    rows = cursor.fetchall()
    print(f"Retrieved {len(rows)} activation traces.")
    
    # Cache parameters
    cache_size = 128 # 128 neurons per expert reside in cache
    
    # We will simulate block counts for three energy levels: 50%, 70%, and 99% (all active_indices)
    targets = ["50%", "70%", "90%", "99%"]
    target_stats = {t: {"orig": [], "adetr": []} for t in targets}
    
    sample_size = min(len(rows), 50000)
    print(f"Analyzing block count reduction on a representative sample of {sample_size} tokens across energy targets...")
    
    # Pre-cache mapping of permutation mappings to speed up simulation
    perm_maps = {}
    inv_perm_maps = {}
    for key, order in permutations.items():
        perm_maps[key] = np.array(order)
        inv_map = np.zeros(len(order), dtype=int)
        for idx, val in enumerate(order):
            inv_map[val] = idx
        inv_perm_maps[key] = inv_map
        
    np.random.seed(42)
    sample_indices = np.random.choice(len(rows), sample_size, replace=False)
    
    for count, idx in enumerate(sample_indices):
        layer, expert_id, active_indices_str, k50, k70, k90 = rows[idx]
        active_list_99 = json.loads(active_indices_str)
        if not active_list_99:
            continue
            
        key = f"{layer}_{expert_id}"
        if key not in permutations:
            continue
            
        inv_map = inv_perm_maps[key]
        
        # Populate sets for different energy levels
        # active_list_99 was stored sorted by magnitude, so slicing it gives the sub-energy active sets
        active_sets = {
            "50%": set(active_list_99[:k50]),
            "70%": set(active_list_99[:k70]),
            "90%": set(active_list_99[:k90]),
            "99%": set(active_list_99)
        }
        
        for t in targets:
            active_set = active_sets[t]
            if not active_set:
                continue
                
            # 1. Original Layout cache: first 128 indices
            orig_cache = set(range(cache_size))
            orig_miss = active_set - orig_cache
            orig_blk = count_contiguous_blocks(orig_miss)
            target_stats[t]["orig"].append(orig_blk)
            
            # 2. ADETR Layout cache: first 128 indices of the permuted order
            adetr_miss_positions = [inv_map[n] for n in active_set if inv_map[n] >= cache_size]
            adetr_blk = count_contiguous_blocks(adetr_miss_positions)
            target_stats[t]["adetr"].append(adetr_blk)
            
    print("\n--- Cache Miss Block Count Statistics ---")
    for t in targets:
        avg_orig_blocks = np.mean(target_stats[t]["orig"]) if target_stats[t]["orig"] else 0.0
        avg_adetr_blocks = np.mean(target_stats[t]["adetr"]) if target_stats[t]["adetr"] else 0.0
        ratio = avg_orig_blocks / avg_adetr_blocks if avg_adetr_blocks > 0 else 1.0
        avg_miss_size = np.mean([len(target_sets[t] - set(range(cache_size))) for count, idx in enumerate(sample_indices) for target_sets in [{
            "50%": set(json.loads(rows[idx][2])[:rows[idx][3]]),
            "70%": set(json.loads(rows[idx][2])[:rows[idx][4]]),
            "90%": set(json.loads(rows[idx][2])[:rows[idx][5]]),
            "99%": set(json.loads(rows[idx][2]))
        }] if target_sets[t]])
        print(f"Energy Target {t} (Avg Miss Size = {avg_miss_size:.1f} neurons):")
        print(f"  Avg contiguous blocks (Original): {avg_orig_blocks:.4f}")
        print(f"  Avg contiguous blocks (ADETR):   {avg_adetr_blocks:.4f}")
        print(f"  Reduction Ratio:                  {ratio:.4f}x")
        
    # Use 70% target as our default for the physical DMA benchmark
    avg_orig_blocks = np.mean(target_stats["70%"]["orig"]) if target_stats["70%"]["orig"] else 1.0
    avg_adetr_blocks = np.mean(target_stats["70%"]["adetr"]) if target_stats["70%"]["adetr"] else 1.0
    reduction_ratio = avg_orig_blocks / avg_adetr_blocks
    
    # -----------------------------------------------------------------
    # 3. FFN Mathematical Correctness
    # -----------------------------------------------------------------
    print("\n--- Verifying Mathematical Correctness ---")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    hidden_dim = 2048
    intermediate_dim = 768
    
    W_gate_up = torch.randn(intermediate_dim, hidden_dim * 2, dtype=torch.bfloat16, device=device)
    W_down = torch.randn(hidden_dim, intermediate_dim, dtype=torch.bfloat16, device=device)
    x = torch.randn(128, hidden_dim, dtype=torch.bfloat16, device=device)
    
    # Pick a random permutation
    test_key = "10_0"
    P = torch.tensor(permutations.get(test_key, list(range(intermediate_dim))), dtype=torch.long, device=device)
    
    # Permute weights
    W_gate_up_new = W_gate_up[P, :]
    W_down_new = W_down[:, P]
    
    # Forward Pass 1: Standard
    with torch.no_grad():
        gate = torch.matmul(x, W_gate_up[:, :hidden_dim].t())
        up = torch.matmul(x, W_gate_up[:, hidden_dim:].t())
        act = torch.nn.functional.silu(gate) * up
        y_standard = torch.matmul(act, W_down.t())
        
        # Forward Pass 2: Permuted
        gate_new = torch.matmul(x, W_gate_up_new[:, :hidden_dim].t())
        up_new = torch.matmul(x, W_gate_up_new[:, hidden_dim:].t())
        act_new = torch.nn.functional.silu(gate_new) * up_new
        y_permuted = torch.matmul(act_new, W_down_new.t())
        
        diff = torch.abs(y_standard - y_permuted).max().item()
        max_val = torch.max(torch.abs(y_standard)).item()
        rel_diff = diff / max_val if max_val > 0 else 0.0
        
        print(f"  Max Absolute Discrepancy (BF16): {diff:.2e}")
        print(f"  Relative Discrepancy (BF16):     {rel_diff:.2e}")
        if rel_diff < 1e-2:
            print("  [SUCCESS] BF16 output difference is within normal accumulation noise.")
            
        # Float64 double precision check to prove mathematical identity
        W_gate_up_double = W_gate_up.double()
        W_down_double = W_down.double()
        x_double = x.double()
        W_gate_up_new_double = W_gate_up_double[P, :]
        W_down_new_double = W_down_double[:, P]
        
        gate_d = torch.matmul(x_double, W_gate_up_double[:, :hidden_dim].t())
        up_d = torch.matmul(x_double, W_gate_up_double[:, hidden_dim:].t())
        act_d = torch.nn.functional.silu(gate_d) * up_d
        y_standard_d = torch.matmul(act_d, W_down_double.t())
        
        gate_new_d = torch.matmul(x_double, W_gate_up_new_double[:, :hidden_dim].t())
        up_new_d = torch.matmul(x_double, W_gate_up_new_double[:, hidden_dim:].t())
        act_new_d = torch.nn.functional.silu(gate_new_d) * up_new_d
        y_permuted_d = torch.matmul(act_new_d, W_down_new_double.t())
        
        diff_d = torch.abs(y_standard_d - y_permuted_d).max().item()
        max_val_d = torch.max(torch.abs(y_standard_d)).item()
        rel_diff_d = diff_d / max_val_d if max_val_d > 0 else 0.0
        print(f"  Max Absolute Discrepancy (FP64): {diff_d:.2e}")
        print(f"  Relative Discrepancy (FP64):     {rel_diff_d:.2e}")
        if rel_diff_d < 1e-12:
            print("  [SUCCESS] FP64 output matches perfectly. ADETR is mathematically identical!")
        else:
            print("  [WARNING] FP64 output mismatch!")

    # -----------------------------------------------------------------
    # 4. Host-to-Device PCIe Gen5 DMA Copy Telemetry
    # -----------------------------------------------------------------
    if not torch.cuda.is_available():
        print("\nCUDA not available. Skipping PCIe DMA latency benchmark.")
        return
        
    print("\n--- Physical PCIe Gen5 DMA Latency Benchmark ---")
    
    # Find a real sequence of missed indices from the database to benchmark
    real_orig_slices = None
    real_adetr_slices = None
    real_miss_size = 0
    
    # Let's search for a representative token record
    for idx in sample_indices:
        layer, expert_id, active_indices_str, k50, k70, k90 = rows[idx]
        active_list_99 = json.loads(active_indices_str)
        active_set_70 = active_list_99[:k70]
        if not active_set_70:
            continue
            
        key = f"{layer}_{expert_id}"
        if key not in permutations:
            continue
            
        inv_map = inv_perm_maps[key]
        
        orig_cache = set(range(cache_size))
        orig_miss = set(active_set_70) - orig_cache
        adetr_miss_positions = [inv_map[n] for n in active_set_70 if inv_map[n] >= cache_size]
        
        if len(orig_miss) >= 16:
            real_orig_slices = get_contiguous_slices(sorted(list(orig_miss)))
            real_adetr_slices = get_contiguous_slices(sorted(adetr_miss_positions))
            real_miss_size = len(orig_miss)
            break
            
    if real_orig_slices is None:
        # fallback
        real_orig_slices = [(128, 160)]
        real_adetr_slices = [(128, 160)]
        real_miss_size = 32
        
    print(f"Benchmarking real trace-based copying for missed size = {real_miss_size} neurons:")
    print(f"  Original strided slices: {len(real_orig_slices)} slices")
    print(f"  ADETR contiguous slices: {len(real_adetr_slices)} slices")
    
    # Pinned CPU memory weights
    cpu_w_gate_up = torch.randn(intermediate_dim, hidden_dim * 2, dtype=torch.bfloat16).pin_memory()
    gpu_w_gate_up = torch.empty(intermediate_dim, hidden_dim * 2, dtype=torch.bfloat16, device=device)
    
    iters = 200
    
    # 1. Benchmark Original Strided copies
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(iters):
        for start, end in real_orig_slices:
            gpu_w_gate_up[start:end].copy_(cpu_w_gate_up[start:end], non_blocking=True)
    end_event.record()
    torch.cuda.synchronize()
    orig_dma_time_us = (start_event.elapsed_time(end_event) * 1000.0) / iters
    
    # 2. Benchmark ADETR copies
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(iters):
        for start, end in real_adetr_slices:
            gpu_w_gate_up[start:end].copy_(cpu_w_gate_up[start:end], non_blocking=True)
    end_event.record()
    torch.cuda.synchronize()
    adetr_dma_time_us = (start_event.elapsed_time(end_event) * 1000.0) / iters
    
    speedup = orig_dma_time_us / adetr_dma_time_us
    
    print(f"  Original Strided DMA Transfer Time: {orig_dma_time_us:.2f} us")
    print(f"  ADETR Contiguous DMA Transfer Time: {adetr_dma_time_us:.2f} us")
    print(f"  PCIe Gen5 DMA Speedup:              {speedup:.2f}x")
    
    # Save results to JSON
    results = {
        "avg_original_blocks": avg_orig_blocks,
        "avg_adetr_blocks": avg_adetr_blocks,
        "block_reduction_ratio": reduction_ratio,
        "correctness_diff": diff,
        "correctness_rel_diff": rel_diff,
        "original_dma_time_us": orig_dma_time_us,
        "adetr_dma_time_us": adetr_dma_time_us,
        "pcie_dma_speedup": speedup
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved evaluation results to {RESULTS_PATH}")
    
    conn.close()

if __name__ == "__main__":
    run_adetr_evaluation()
