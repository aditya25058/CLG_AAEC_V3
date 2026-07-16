import os
import json
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def main():
    db_path = "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/qwen3_30b_real_v2.db"
    out_dir = "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)
    
    print("Loading activations data...")
    cursor = conn.execute(
        "SELECT layer, expert_id, prompt_id, token_pos, active_indices, "
        "energy_k_50, intermediate_dim FROM activations ORDER BY prompt_id, token_pos, layer, expert_id"
    )
    rows = cursor.fetchall()
    print(f"Loaded {len(rows)} records.")
    
    # ---------------------------------------------------------
    # 1. Cache Residency & Lifetime Metrics (Sweep at Cache Size = 128)
    # ---------------------------------------------------------
    print("\n--- Running Cache Residency Study ---")
    
    expert_sequences = defaultdict(list)
    for r in rows:
        key = (r[0], r[1])
        indices = json.loads(r[4])[:r[5]] # 50% energy set
        expert_sequences[key].append(indices)
        
    # We will track:
    # - Neuron Lifetime: number of steps a neuron stays in the cache from insertion to eviction
    # - Reuse Distance: number of steps between consecutive hits on the same neuron
    # - Eviction Frequency: evictions per token step
    
    neuron_lifetimes = []
    reuse_distances = []
    eviction_counts = []
    total_token_steps = 0
    
    for key, seq in expert_sequences.items():
        if len(seq) < 20:
            continue
            
        # Simulate LRU cache of size 128
        cache = []
        inserted_step = {} # neuron -> step index when inserted
        last_hit_step = {} # neuron -> step index of last hit
        
        for step, act in enumerate(seq):
            total_token_steps += 1
            evictions = 0
            
            # Hit/Miss checks
            for neuron in act:
                if neuron in cache:
                    # Hit: Record reuse distance
                    if neuron in last_hit_step:
                        reuse_distances.append(step - last_hit_step[neuron])
                    last_hit_step[neuron] = step
                    
                    # Update LRU queue
                    cache.remove(neuron)
                    cache.append(neuron)
                else:
                    # Miss: Insert
                    if len(cache) >= 128:
                        # Evict LRU (index 0)
                        evicted = cache.pop(0)
                        evictions += 1
                        # Record lifetime
                        if evicted in inserted_step:
                            neuron_lifetimes.append(step - inserted_step[evicted])
                            inserted_step.pop(evicted)
                            if evicted in last_hit_step:
                                last_hit_step.pop(evicted)
                                
                    cache.append(neuron)
                    inserted_step[neuron] = step
                    last_hit_step[neuron] = step
                    
            eviction_counts.append(evictions)
            
    print("Cache Residency Statistics (at Cache Size = 128):")
    print(f"  Mean Neuron Cache Lifetime: {np.mean(neuron_lifetimes):.2f} token visits")
    print(f"  Median Neuron Cache Lifetime: {np.median(neuron_lifetimes):.1f} token visits")
    print(f"  Mean Cache Reuse Distance:  {np.mean(reuse_distances):.2f} tokens")
    print(f"  Median Cache Reuse Distance: {np.median(reuse_distances):.1f} tokens")
    print(f"  Average Evictions per Step: {np.mean(eviction_counts):.2f} evictions/step")
    
    # ---------------------------------------------------------
    # 2. Multi-Request Concurrent Serving Simulation
    # ---------------------------------------------------------
    print("\n--- Running Multi-Request Concurrent Serving Study ---")
    # We simulate User A (Coding prompt 0) and User B (Coding prompt 1) running concurrently.
    # Interleave tokens: A1, B1, A2, B2, A3, B3...
    # Compare Single-User Cache vs Shared Multi-Tenant Cache (Size = 128)
    
    # Group activations by prompt
    prompt_activations = defaultdict(list)
    for r in rows:
        # We group layer-expert activations at each token pos
        prompt_activations[r[2]].append((r[0], r[1], json.loads(r[4])[:r[5]]))
        
    def run_concurrent_cache(p1, p2, shared=True):
        # We interleave the tokens of prompt 1 and prompt 2
        seq1 = prompt_activations[p1]
        seq2 = prompt_activations[p2]
        
        # We separate into tokens
        tokens1 = defaultdict(list)
        for layer, exp, indices in seq1:
            tokens1[layer].append(indices)
            
        tokens2 = defaultdict(list)
        for layer, exp, indices in seq2:
            tokens2[layer].append(indices)
            
        # Cache per layer
        if shared:
            caches = {l: [] for l in range(48)}
        else:
            caches1 = {l: [] for l in range(48)}
            caches2 = {l: [] for l in range(48)}
            
        hits = 0
        totals = 0
        
        max_tokens = max(len(tokens1[0]), len(tokens2[0]))
        for t in range(max_tokens):
            # User 1 token
            for l in range(48):
                if t < len(tokens1[l]):
                    act = tokens1[l][t]
                    cache = caches[l] if shared else caches1[l]
                    for n in act:
                        totals += 1
                        if n in cache:
                            hits += 1
                            cache.remove(n)
                            cache.append(n)
                        else:
                            if len(cache) >= 128:
                                cache.pop(0)
                            cache.append(n)
                            
            # User 2 token
            for l in range(48):
                if t < len(tokens2[l]):
                    act = tokens2[l][t]
                    cache = caches[l] if shared else caches2[l]
                    for n in act:
                        totals += 1
                        if n in cache:
                            hits += 1
                            cache.remove(n)
                            cache.append(n)
                        else:
                            if len(cache) >= 128:
                                cache.pop(0)
                            cache.append(n)
                            
        return hits / totals if totals > 0 else 0.0

    # Single-user baselines
    single_hit_rates = []
    for p in [0, 1, 2]:
        single_hit_rates.append(run_concurrent_cache(p, p, shared=False))
    mean_single = np.mean(single_hit_rates)
    
    # Shared multi-tenant hit rates
    shared_same_category = []
    # Same category pairs: Coding (0,1, 1,2, 0,2)
    shared_same_category.append(run_concurrent_cache(0, 1, shared=True))
    shared_same_category.append(run_concurrent_cache(1, 2, shared=True))
    
    shared_diff_category = []
    # Different category pairs: Coding 0 vs Math 10, Coding 1 vs Math 11
    shared_diff_category.append(run_concurrent_cache(0, 10, shared=True))
    shared_diff_category.append(run_concurrent_cache(1, 11, shared=True))
    
    print("Multi-Request Concurrent Serving Hit Rates:")
    print(f"  Single-User Isolated Cache:      {mean_single*100:6.2f}%")
    print(f"  Shared Same-Category Tenant:    {np.mean(shared_same_category)*100:6.2f}%")
    print(f"  Shared Diff-Category Tenant:    {np.mean(shared_diff_category)*100:6.2f}%")

    # ---------------------------------------------------------
    # 3. Serving Latency Modeling
    # ---------------------------------------------------------
    print("\n--- Running Serving Latency Modeling Study ---")
    # Constants:
    # Hidden dimension = 4096, Intermediate dimension = 768.
    # swiglu intermediate size per active token routed = 115.5 neurons (at 50% energy)
    # Param packet size per neuron = SwiGLU: (2 * 4096) + down_proj: (4096) = 12288 parameters * 2 bytes = 24.576 KB
    # (Note: SwiGLU has gate_proj and up_proj, each size 4096. down_proj has size 4096.
    #  Total weight size per neuron = (4096 + 4096 + 4096) * 2 bytes = 24.576 KB)
    # Baseline full expert weight size = 768 neurons * 24.576 KB = 18.874 MB
    
    # Link specifications:
    # 1. HBM (High Bandwidth Memory): 3.2 TB/s (3200 GB/s) - weights on GPU.
    # 2. PCIe Gen5 x16: 63 GB/s - offloaded weights on CPU.
    # Overheads:
    # DMA launch overhead = 5 us, Kernel launch overhead = 10 us. Total = 15 us per FFN layer.
    
    # Let's compute average FFN execution latency per token step:
    # FFN Latency = Transfer Time + Kernel overhead + Compute Time (neglect compute since it's identical)
    # We sweep cache size C in {32, 64, 96, 128, 160}
    cache_sizes = [32, 64, 96, 128, 160]
    
    # Miss rates for AAEC (from Experiment 11)
    aaec_miss_rates = {
        32:  1.0 - 0.0454,
        64:  1.0 - 0.0942,
        96:  1.0 - 0.1614,
        128: 1.0 - 0.2447,
        160: 1.0 - 0.3273
    }
    
    # SwiGLU FFN active neurons per token = 115.5 (mean 50% energy)
    # Baseline loads full expert (18.874 MB) per token visit
    # Under AAEC, we only load the missed neurons: misses = 115.5 * miss_rate
    # Bytes transferred under AAEC = misses * 24.576 KB
    
    neuron_weight_size_kb = 24.576
    full_expert_size_mb = 18.874
    
    hbm_bw = 3200.0 * 1024 * 1024 # KB/s
    pcie_bw = 63.0 * 1024 * 1024 # KB/s
    
    print("Serving Latency Projections (in microseconds per token-expert):")
    print(f"{'Cache Size':<10} | {'HBM Baseline':<14} | {'HBM AAEC':<10} | {'HBM Speedup':<12} | {'PCIe Baseline':<14} | {'PCIe AAEC':<10} | {'PCIe Speedup':<12}")
    print("-" * 95)
    
    hbm_lats_aaec = []
    pcie_lats_aaec = []
    hbm_lats_base = []
    pcie_lats_base = []
    
    for size in cache_sizes:
        m_rate = aaec_miss_rates[size]
        misses = 115.5 * m_rate
        aaec_bytes_kb = misses * neuron_weight_size_kb
        base_bytes_kb = 768.0 * neuron_weight_size_kb
        
        # 1. HBM Serving (Weights on GPU)
        # Latency = Bytes / BW + overhead
        t_hbm_base = (base_bytes_kb / hbm_bw) * 1e6 + 15.0 # us
        t_hbm_aaec = (aaec_bytes_kb / hbm_bw) * 1e6 + 15.0 # us
        hbm_speedup = t_hbm_base / t_hbm_aaec
        
        # 2. PCIe Serving (Offloaded weights on CPU)
        t_pcie_base = (base_bytes_kb / pcie_bw) * 1e6 + 15.0 # us
        t_pcie_aaec = (aaec_bytes_kb / pcie_bw) * 1e6 + 15.0 # us
        pcie_speedup = t_pcie_base / t_pcie_aaec
        
        hbm_lats_aaec.append(t_hbm_aaec)
        pcie_lats_aaec.append(t_pcie_aaec)
        hbm_lats_base.append(t_hbm_base)
        pcie_lats_base.append(t_pcie_base)
        
        print(f"{size:<10d} | {t_hbm_base:11.2f} us | {t_hbm_aaec:7.2f} us | {hbm_speedup:9.2f}x | {t_pcie_base:11.2f} us | {t_pcie_aaec:7.2f} us | {pcie_speedup:9.2f}x")
        
    # Generate Latency Sweep Plots
    plt.figure(figsize=(7, 4.5))
    plt.plot(cache_sizes, pcie_lats_base, ls=':', color='black', label='PCIe Baseline (No Caching)')
    plt.plot(cache_sizes, pcie_lats_aaec, marker='o', linewidth=2.5, color='#e63946', label='PCIe AAEC (Offloaded serving)')
    plt.xlabel('Cache Size (Neurons)', fontsize=11, fontweight='bold')
    plt.ylabel('Latency per Layer (us)', fontsize=11, fontweight='bold')
    plt.title('PCIe Offloaded Serving Latency vs. Cache Size', fontsize=12, fontweight='bold', pad=12)
    plt.grid(True, ls='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "latency_pcie_sweep.png"), dpi=200)
    plt.close()
    
    plt.figure(figsize=(7, 4.5))
    plt.plot(cache_sizes, hbm_lats_base, ls=':', color='black', label='HBM Baseline')
    plt.plot(cache_sizes, hbm_lats_aaec, marker='s', linewidth=2.5, color='#1d3557', label='HBM AAEC (Weights on GPU)')
    plt.xlabel('Cache Size (Neurons)', fontsize=11, fontweight='bold')
    plt.ylabel('Latency per Layer (us)', fontsize=11, fontweight='bold')
    plt.title('HBM GPU Serving Latency vs. Cache Size', fontsize=12, fontweight='bold', pad=12)
    plt.grid(True, ls='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "latency_hbm_sweep.png"), dpi=200)
    plt.close()
    
    print("\nSystems modeling run complete and plots saved successfully!")
    conn.close()

if __name__ == "__main__":
    main()
