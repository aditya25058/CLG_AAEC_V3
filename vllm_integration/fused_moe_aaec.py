# vllm_integration/fused_moe_aaec.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

# Import our custom Triton kernel wrapper
from sa_ffn_triton import sa_ffn_forward

class CacheController:
    """
    Simulates or coordinates the dynamic VRAM caching directory.
    Manages CPU DRAM pinned memory coordinates and issues async PCIe transfers.
    """
    def __init__(self, num_layers: int, num_experts: int, cache_size: int, hidden_size: int):
        self.cache_size = cache_size
        self.device = torch.device("cuda:0")
        
        # Allocate L2 Dynamic Cache GPU HBM partitions
        self.gpu_cache_gate = torch.zeros((num_layers, num_experts, cache_size, hidden_size), dtype=torch.bfloat16, device=self.device)
        self.gpu_cache_up   = torch.zeros((num_layers, num_experts, cache_size, hidden_size), dtype=torch.bfloat16, device=self.device)
        self.gpu_cache_down = torch.zeros((num_layers, num_experts, hidden_size, cache_size), dtype=torch.bfloat16, device=self.device)
        
        # Dedicated CUDA stream for non-blocking PCIe DMA transfers
        self.dma_stream = torch.cuda.Stream(device=self.device)
        
        # Pre-allocated receiving buffers for missed columns
        self.recv_gate = torch.empty((num_layers, num_experts, 256, hidden_size), dtype=torch.bfloat16, device=self.device)
        self.recv_up   = torch.empty((num_layers, num_experts, 256, hidden_size), dtype=torch.bfloat16, device=self.device)
        self.recv_down = torch.empty((num_layers, num_experts, hidden_size, 256), dtype=torch.bfloat16, device=self.device)

    def fetch_misses_async(self, layer_idx: int, expert_idx: int, cpu_gate_src, cpu_up_src, cpu_down_src, miss_indices: torch.Tensor):
        """
        Asynchronously fetches missed columns from host memory over PCIe Gen5.
        """
        num_misses = miss_indices.numel()
        if num_misses == 0:
            return None, None, None
            
        miss_indices_cpu = miss_indices.cpu()
        with torch.cuda.stream(self.dma_stream):
            # Non-blocking PCIe transfers using pinned memory src tensors
            self.recv_gate[layer_idx, expert_idx, :num_misses].copy_(cpu_gate_src[miss_indices_cpu], non_blocking=True)
            self.recv_up[layer_idx, expert_idx, :num_misses].copy_(cpu_up_src[miss_indices_cpu], non_blocking=True)
            self.recv_down[layer_idx, expert_idx, :, :num_misses].copy_(cpu_down_src[:, miss_indices_cpu], non_blocking=True)
            
        return (
            self.recv_gate[layer_idx, expert_idx, :num_misses],
            self.recv_up[layer_idx, expert_idx, :num_misses],
            self.recv_down[layer_idx, expert_idx, :, :num_misses]
        )


class FusedMoEWithAAEC(nn.Module):
    """
    A replacement layer class for vLLM's FusedMoE layer.
    Intercepts routing decisions, retrieves cached weight columns,
    triggers async transfers for misses, and calls the Triton SA-FFN kernel.
    """
    def __init__(
        self,
        layer_idx: int,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        cache_controller: CacheController
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.cache_controller = cache_controller
        
        # Router linear layer (gate projection scores computation)
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Complete model weights placed in host RAM (simulating pinned DRAM storage)
        # Note: In production, these are loaded from checkpoint shards
        self.cpu_w_gate = torch.randn(num_experts, intermediate_size, hidden_size, dtype=torch.bfloat16).pin_memory()
        self.cpu_w_up   = torch.randn(num_experts, intermediate_size, hidden_size, dtype=torch.bfloat16).pin_memory()
        self.cpu_w_down = torch.randn(num_experts, hidden_size, intermediate_size, dtype=torch.bfloat16).pin_memory()
        
        # Local mapping directories for cache indices
        # In this demo, we assume the first `cache_size` columns are statically cached.
        # In production, this directory is dynamically updated by the EMA cache manager.
        self.cached_neuron_indices = torch.arange(0, self.cache_controller.cache_size, device="cuda:0")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        M, K = hidden_states.shape # Token activations size
        
        # 1. Evaluate routing gating scores
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=-1)
        topk_weights, topk_ids = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # Normalize topk weights
        topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
        
        # Prepare output buffer
        final_output = torch.zeros_like(hidden_states)
        
        # Compute events to synchronize execution
        compute_done = torch.cuda.Event()
        dma_done = torch.cuda.Event()
        
        # 2. Iterate through experts routing tokens
        for expert_idx in range(self.num_experts):
            # Locate tokens allocated to this expert
            token_mask = (topk_ids == expert_idx).any(dim=-1)
            active_tokens_idx = torch.nonzero(token_mask).squeeze(-1)
            
            if active_tokens_idx.numel() == 0:
                continue
                
            x_expert = hidden_states[active_tokens_idx]
            
            # Determine active neurons based on token activation values
            # For demonstration, we assume we miss a subset of columns [cache_size : cache_size+16]
            miss_indices = torch.arange(
                self.cache_controller.cache_size,
                self.cache_controller.cache_size + 16,
                device="cuda:0"
            )
            
            # Step A: Run local cached compute on the main compute stream
            with torch.cuda.stream(torch.cuda.current_stream()):
                W_g_c = self.cache_controller.gpu_cache_gate[self.layer_idx, expert_idx]
                W_u_c = self.cache_controller.gpu_cache_up[self.layer_idx, expert_idx]
                W_d_c = self.cache_controller.gpu_cache_down[self.layer_idx, expert_idx]
                compute_done.record()
                
            # Step B: Trigger PCIe copy of missed columns in parallel on the DMA stream
            W_g_m, W_u_m, W_d_m = self.cache_controller.fetch_misses_async(
                self.layer_idx,
                expert_idx,
                self.cpu_w_gate[expert_idx],
                self.cpu_w_up[expert_idx],
                self.cpu_w_down[expert_idx],
                miss_indices
            )
            
            # Record DMA completion event
            with torch.cuda.stream(self.cache_controller.dma_stream):
                self.cache_controller.dma_stream.wait_event(compute_done)
                dma_done.record()
                
            # Step C: Sync streams at execution boundary
            torch.cuda.current_stream().wait_event(dma_done)
            
            # Step D: Call our custom Triton kernel to accumulate Phase 1 and Phase 2 GEMMs in-place
            if W_g_m is not None:
                y_expert = sa_ffn_forward(
                    x_expert,
                    W_g_c, W_u_c, W_d_c,
                    W_g_m, W_u_m, W_d_m
                )
            else:
                # Cache-only fallback computation
                g = torch.matmul(x_expert, W_g_c.t())
                u = torch.matmul(x_expert, W_u_c.t())
                y_expert = torch.matmul(F.silu(g) * u, W_d_c.t())
                
            # Multiply by router gating weights and add to final output
            expert_weights = topk_weights[active_tokens_idx]
            expert_mask = (topk_ids[active_tokens_idx] == expert_idx)
            gating_scale = expert_weights[expert_mask].unsqueeze(-1)
            
            final_output[active_tokens_idx] += y_expert * gating_scale
            
        return final_output
