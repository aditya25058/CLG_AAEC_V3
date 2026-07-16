import torch
import time

def benchmark_hnc():
    print("Initializing Hierarchical Neuron Cache (HNC) Hardware Benchmark...")
    hidden_dim = 4096
    
    # Simulate a cache miss fetching 10 neurons = 245.8 KB
    packet_neurons = 10
    packet_size_bytes = packet_neurons * hidden_dim * 2 * 3 # 3 matrices (w1, w2, w3), 2 bytes per bf16
    print(f"Neuron Packet Size: {packet_size_bytes / 1024:.2f} KB\n")
    
    # Initialize physical hierarchy
    print("Allocating memory hierarchy...")
    
    # L2: Local GPU 0 (Cache Hit)
    l2_cache = torch.randn((packet_neurons, hidden_dim * 3), dtype=torch.bfloat16, device="cuda:0")
    
    # L3: Neighbor GPU 1 (NVLink Peer Cache)
    l3_cache = torch.randn((packet_neurons, hidden_dim * 3), dtype=torch.bfloat16, device="cuda:1")
    
    # L4: Host CPU (PCIe Pinned Memory)
    l4_cache = torch.randn((packet_neurons, hidden_dim * 3), dtype=torch.bfloat16).pin_memory()
    
    # Recv buffers on GPU 0
    l3_recv = torch.empty_like(l2_cache)
    l4_recv = torch.empty_like(l2_cache)
    
    # Phase 1 Compute: 100 cached neurons
    phase1_neurons = 100
    local_weights = torch.randn((phase1_neurons, hidden_dim * 3), dtype=torch.bfloat16, device="cuda:0")
    x = torch.randn((1, hidden_dim), dtype=torch.bfloat16, device="cuda:0")
    
    # CUDA Streams for parallel Async transfers
    stream_compute = torch.cuda.Stream(device="cuda:0")
    stream_nvlink = torch.cuda.Stream(device="cuda:0")
    stream_pcie = torch.cuda.Stream(device="cuda:0")
    
    print("Warming up streams...")
    with torch.cuda.stream(stream_compute):
        for _ in range(20):
            gate = torch.matmul(x, local_weights[:, :hidden_dim].t())
            up = torch.matmul(x, local_weights[:, hidden_dim:hidden_dim*2].t())
            act = torch.nn.functional.silu(gate) * up
            out = torch.matmul(act, local_weights[:, hidden_dim*2:])
            
    with torch.cuda.stream(stream_nvlink):
        for _ in range(20):
            l3_recv.copy_(l3_cache, non_blocking=True)
            
    with torch.cuda.stream(stream_pcie):
        for _ in range(20):
            l4_recv.copy_(l4_cache, non_blocking=True)
            
    torch.cuda.synchronize()
    
    # Measure L3 NVLink
    iters = 100
    
    start = time.perf_counter()
    with torch.cuda.stream(stream_nvlink):
        for _ in range(iters):
            l3_recv.copy_(l3_cache, non_blocking=True)
    stream_nvlink.synchronize()
    l3_latency = ((time.perf_counter() - start) / iters) * 1e6
    
    # Measure L4 PCIe
    start = time.perf_counter()
    with torch.cuda.stream(stream_pcie):
        for _ in range(iters):
            l4_recv.copy_(l4_cache, non_blocking=True)
    stream_pcie.synchronize()
    l4_latency = ((time.perf_counter() - start) / iters) * 1e6
    
    # Measure Phase 1 Compute
    start = time.perf_counter()
    with torch.cuda.stream(stream_compute):
        for _ in range(iters):
            gate = torch.matmul(x, local_weights[:, :hidden_dim].t())
            up = torch.matmul(x, local_weights[:, hidden_dim:hidden_dim*2].t())
            act = torch.nn.functional.silu(gate) * up
            out = torch.matmul(act, local_weights[:, hidden_dim*2:])
    stream_compute.synchronize()
    phase1_latency = ((time.perf_counter() - start) / iters) * 1e6
    
    print(f"--- Hierarchical Neuron Cache (HNC) Latencies ---")
    print(f"Level 3 (Neighbor GPU via NVLink): {l3_latency:.2f} us")
    print(f"Level 4 (Host CPU via PCIe Gen5):  {l4_latency:.2f} us")
    print(f"Level 5 (Remote Node InfiniBand):  ~150.00 us (Hardware theoretical bound)")
    print(f"-------------------------------------------------")
    print(f"Phase 1 Local Compute Window:      {phase1_latency:.2f} us")
    print(f"-------------------------------------------------")
    
    print("\n--- HNC Latency Hiding Analysis ---")
    if l3_latency < phase1_latency:
        print(f"✅ L3 NVLink Miss is 100% HIDDEN! ({l3_latency:.2f}us < {phase1_latency:.2f}us)")
    else:
        print(f"❌ L3 NVLink Miss EXPOSED: {l3_latency - phase1_latency:.2f}us delay")
        
    if l4_latency < phase1_latency:
        print(f"✅ L4 PCIe Miss is 100% HIDDEN! ({l4_latency:.2f}us < {phase1_latency:.2f}us)")
    else:
        print(f"❌ L4 PCIe Miss EXPOSED: {l4_latency - phase1_latency:.2f}us delay")

    if 150.0 < phase1_latency:
        print(f"✅ L5 InfiniBand Miss is 100% HIDDEN! (150.00us < {phase1_latency:.2f}us)")
    else:
        print(f"❌ L5 InfiniBand Miss EXPOSED: {150.0 - phase1_latency:.2f}us delay")
        
if __name__ == "__main__":
    benchmark_hnc()
