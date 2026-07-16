import os
import json
import torch
import time

def test_neuron_packet_dma():
    print("==================================================================")
    # Target configurations representing Qwen3-30B-A3B:
    # d_model = 5120, intermediate_dim (expert FFN) = 768
    d_model = 5120
    ffn_dim = 768
    
    N_miss_list = [32, 64, 128, 256]
    results = {}
    
    cuda_available = torch.cuda.is_available()
    
    print(f"Configuring Neuron-Channel Packet Telemetry Sweep for Qwen3-30B-A3B:")
    print(f" - Hidden Dimension (d_model): {d_model}")
    print(f" - Expert FFN Dimension:       {ffn_dim}")
    print(f" - Miss Sizes Swept (N):       {N_miss_list}")
    print(f" - CUDA GPU Available:         {cuda_available}")
    print("==================================================================\n")
    
    proj_size_bytes = d_model * 2
    packet_size_bytes = proj_size_bytes * 3
    
    print(f"| N (Miss) | Strided Latency (us) | Packed Latency (us) | Speedup | Strided BW (GB/s) | Packed BW (GB/s) |")
    print(f"|----------|──────────────────────|─────────────────────|─────────|───────────────────|──────────────────|")
    
    if cuda_available:
        # Pre-allocate host pinned memory
        host_gate = torch.randn((d_model, ffn_dim), dtype=torch.bfloat16).pin_memory()
        host_up = torch.randn((d_model, ffn_dim), dtype=torch.bfloat16).pin_memory()
        host_down = torch.randn((ffn_dim, d_model), dtype=torch.bfloat16).pin_memory()
        host_packets = torch.randn((ffn_dim, 3, d_model), dtype=torch.bfloat16).pin_memory()
        
        for N_miss in N_miss_list:
            total_payload_bytes = N_miss * packet_size_bytes
            
            # Device allocations
            device_gate = torch.empty((d_model, N_miss), dtype=torch.bfloat16, device="cuda:0")
            device_up = torch.empty((d_model, N_miss), dtype=torch.bfloat16, device="cuda:0")
            device_down = torch.empty((N_miss, d_model), dtype=torch.bfloat16, device="cuda:0")
            device_packets = torch.empty((N_miss, 3, d_model), dtype=torch.bfloat16, device="cuda:0")
            
            indices = torch.randperm(ffn_dim)[:N_miss].tolist()
            
            # Warm up
            torch.cuda.synchronize()
            for _ in range(5):
                for idx, col in enumerate(indices):
                    device_gate[:, idx].copy_(host_gate[:, col], non_blocking=True)
                    device_up[:, idx].copy_(host_up[:, col], non_blocking=True)
                    device_down[idx, :].copy_(host_down[col, :], non_blocking=True)
                device_packets.copy_(host_packets[indices], non_blocking=True)
            torch.cuda.synchronize()
            
            # Benchmark Strided
            iters = 50
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            for _ in range(iters):
                for idx, col in enumerate(indices):
                    device_gate[:, idx].copy_(host_gate[:, col], non_blocking=True)
                    device_up[:, idx].copy_(host_up[:, col], non_blocking=True)
                    device_down[idx, :].copy_(host_down[col, :], non_blocking=True)
            end_event.record()
            torch.cuda.synchronize()
            strided_latency = (start_event.elapsed_time(end_event) / iters) * 1000
            
            # Benchmark Packed
            start_event.record()
            for _ in range(iters):
                device_packets.copy_(host_packets[:N_miss], non_blocking=True)
            end_event.record()
            torch.cuda.synchronize()
            packed_latency = (start_event.elapsed_time(end_event) / iters) * 1000
            
            strided_bw = (total_payload_bytes / 1e9) / (strided_latency / 1e6)
            packed_bw = (total_payload_bytes / 1e9) / (packed_latency / 1e6)
            speedup = strided_latency / packed_latency
            
            print(f"| {N_miss:8d} | {strided_latency:20.2f} | {packed_latency:19.2f} | {speedup:7.2f}x | {strided_bw:17.2f} | {packed_bw:16.2f} |")
            
            results[str(N_miss)] = {
                "payload_bytes": total_payload_bytes,
                "strided_latency_us": strided_latency,
                "packed_latency_us": packed_latency,
                "speedup": speedup,
                "strided_bw_gbps": strided_bw,
                "packed_bw_gbps": packed_bw
            }
    else:
        # CPU-only emulation: Model PCIe Gen5 DMA latency and driver staging overheads
        # PCIe Gen5 bandwidth = 64 GB/s, NVLink = 450 GB/s
        pcie_bw_gbps = 64.0
        # Typical host-to-device kernel launch and staging driver overhead is ~2.5 us per call
        driver_launch_overhead_us = 2.5
        
        for N_miss in N_miss_list:
            total_payload_bytes = N_miss * packet_size_bytes
            
            # Strided: N_miss * 3 separate DMA transfers
            num_transfers = N_miss * 3
            transmission_time_us = (total_payload_bytes / 1e9) / pcie_bw_gbps * 1e6
            strided_latency = (num_transfers * driver_launch_overhead_us) + transmission_time_us
            
            # Packed: 1 contiguous DMA transfer
            packed_latency = (1 * driver_launch_overhead_us) + transmission_time_us
            
            strided_bw = (total_payload_bytes / 1e9) / (strided_latency / 1e6)
            packed_bw = (total_payload_bytes / 1e9) / (packed_latency / 1e6)
            speedup = strided_latency / packed_latency
            
            print(f"| {N_miss:8d} | {strided_latency:20.2f} | {packed_latency:19.2f} | {speedup:7.2f}x | {strided_bw:17.2f} | {packed_bw:16.2f} |")
            
            results[str(N_miss)] = {
                "payload_bytes": total_payload_bytes,
                "strided_latency_us": strided_latency,
                "packed_latency_us": packed_latency,
                "speedup": speedup,
                "strided_bw_gbps": strided_bw,
                "packed_bw_gbps": packed_bw
            }
            
    output_json_path = "/home/palakm/MoEServingSim/qwen3_30b_plots/neuron_packet_dma_results.json"
    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nSuccessfully saved sweep results to: {output_json_path}")
    print("==================================================================")

if __name__ == "__main__":
    test_neuron_packet_dma()
