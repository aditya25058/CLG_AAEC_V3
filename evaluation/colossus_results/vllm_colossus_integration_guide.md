# Systems Integration Guide: Implementing COLOSSUS in vLLM

To validate COLOSSUS in a production inference stack and publish at a top systems venue (OSDI, SOSP, NSDI, EuroSys), we must move from an offline PyTorch benchmark to an online integration. This guide provides a concrete, step-by-step engineering plan to integrate COLOSSUS into the **vLLM** serving framework.

---

## Architectural Overview: COLOSSUS in vLLM

To implement COLOSSUS, we must intercept the data flow at three layers of the vLLM engine:
1.  **Gating Layer (`moe.py`):** Intercept routing decisions to determine column activations.
2.  **Memory Worker (`model_runner.py`):** Manage the HBM dynamic cache and issue asynchronous DMA PCIe/NVLink copies.
3.  **GEMM Layer (`fused_moe_triton.py`):** Implement the Streaming Accumulation FFN (SA-FFN) using a custom Triton kernel.

```
       [ Token Batch ] 
              │
              ▼
    [ Router (Gate Projs) ] ──(Active Neurons list)──► [ Cache Controller ]
              │                                                 │
      (Compute Stream)                                   (Async DMA Stream)
              │                                                 │
    [ Phase 1 Triton GEMM ]                              [ PCIe / NVLink Copy ]
      (Cached HBM Columns)                               (Stream Missed Columns)
              │                                                 │
              ▼                                                 ▼
              └───────────────► [ CUDA Event Sync ] ◄───────────┘
                                        │
                                        ▼
                              [ Phase 2 Triton GEMM ]
                              (Accumulate Misses In-Place)
```

---

## Step 1: Modifying the Router to Track Active Neurons

In vLLM, Mixture-of-Experts routing is handled in `vllm/model_executor/layers/moe/fused_moe.py`. We must intercept the token gating scores.

### Code Modification Blueprint:
1.  Navigate to the `FusedMoE` class.
2.  Locate where the routing scores are computed (after the gating linear projection and softmax/top-k mask):
    ```python
    # vllm/model_executor/layers/moe/fused_moe.py
    topk_weights, topk_ids = self.router(hidden_states)
    ```
3.  **Active-Neuron Masking:** Compute the activation score for each intermediate neuron. For GeLU/SiLU, the activation score is given by the gate magnitude:
    $$A(x) = \text{softmax}(x \cdot W_g)$$
4.  Write a helper function to extract the indices of the columns that exceed our target activation energy threshold $\eta = 80\%$ (or retrieve a fixed $Top-K$ neuron mask):
    ```python
    # Intercept active neuron columns per expert
    active_mask = torch.nonzero(activation_scores > threshold)
    ```

---

## Step 2: Designing the Async Cache Controller

We need a C++ or PyTorch-level **Cache Controller** running on a dedicated CUDA stream (`stream_dma`) to manage the VRAM neuron columns.

### Implementation steps:
1.  Allocate a **Dynamic Cache Directory** in GPU HBM. This directory holds $15\%$ of the FFN intermediate weight parameters.
2.  Maintain a lookup table mapping `(layer_idx, expert_idx, neuron_idx)` to the memory address of the column in the HBM cache.
3.  **Cache Miss Handling:**
    *   For columns that are present in the HBM cache: retrieve their pointers.
    *   For columns that are not present (cache misses): select victim columns using a Least Recently Used (LRU) policy, write them back if needed, and issue a non-blocking `copy_` from pinned CPU DRAM over PCIe:
        ```python
        # Trigger async transfer on the DMA stream
        with torch.cuda.stream(stream_dma):
            gpu_recv_buffer.copy_(cpu_pinned_weights[miss_indices], non_blocking=True)
        ```

---

## Step 3: Custom Triton Kernel (SA-FFN)

Instead of calling separate PyTorch GEMM operations, we must compile a custom Triton kernel in `vllm/model_executor/layers/moe/fused_moe_triton.py` to execute Phase 1 and Phase 2 accumulation in-place.

### Triton Kernel Code Blueprint:
```python
import triton
import triton.language as tl

@triton.jit
def sa_ffn_kernel(
    X_ptr,          # Input hidden states (M, K)
    W_cached_ptr,   # VRAM cached expert columns (DC, K)
    W_missed_ptr,   # Asynchronously copied expert columns (MS, K)
    Y_ptr,          # Output tensor (M, N)
    # Stride parameters...
):
    # Program ID & offset calculation
    pid = tl.program_id(axis=0)
    
    # 1. Phase 1 GEMM: Compute on Warm Cached Columns
    # Read block of inputs and cached weights
    x_block = tl.load(X_ptr + offsets_x)
    w_cached_block = tl.load(W_cached_ptr + offsets_wc)
    
    # Compute gate * up activation
    gate_cached = tl.dot(x_block, w_cached_block)
    act_cached = silu(gate_cached)  # custom silu logic
    
    # 2. Phase 2 GEMM: Wait for Streamed Columns and Accumulate
    w_missed_block = tl.load(W_missed_ptr + offsets_wm)
    gate_missed = tl.dot(x_block, w_missed_block)
    
    # In-place accumulate output
    out = act_cached + silu(gate_missed)
    tl.store(Y_ptr + offsets_y, out)
```

---

## Step 4: End-to-End Pipelined Execution

To maximize performance, we schedule token execution dynamically so that memory transfer overlaps with compute. We hook this into vLLM's worker execution loop in `vllm/worker/model_runner.py`:

```python
# Create synchronization events
compute_done = torch.cuda.Event()
dma_done = torch.cuda.Event()

# Layer loop
for layer_idx in range(num_layers):
    # 1. Start Phase 1 compute on GPU local cache
    with torch.cuda.stream(compute_stream):
        run_phase1_gemm(hidden_states, cached_weights)
        compute_done.record()
        
    # 2. Start Async copy of Missed Columns over PCIe/NVLink
    with torch.cuda.stream(dma_stream):
        dma_stream.wait_event(compute_done) # start copy after router outputs are ready
        fetch_missed_columns_async(miss_indices)
        dma_done.record()
        
    # 3. Synchronize streams at layer boundary
    compute_stream.wait_event(dma_done)
    
    # 4. Start Phase 2 accumulation
    with torch.cuda.stream(compute_stream):
        run_phase2_accumulation(hidden_states, recv_buffer)
```

---

## Step 5: Setting Up the Telemetry Benchmarks

To publish, we must benchmark the system under standard workloads using real pre-trained weights (e.g. `DeepSeek-V2-Lite` or `Mixtral-8x7B`).

### 1. Test Workloads
We evaluate serving performance on standard datasets:
*   **ShareGPT dataset:** Evaluates conversational throughput.
*   **GSM8K / HumanEval:** Evaluates quality and code correctness.

### 2. Physical Metrics Recording
We use the NVIDIA Profiler (`nsys`) programmatically to track physical data bus utilization:
```bash
# Capture PCIe and NVLink bandwidth utilization during vLLM serving
nsys profile --trace=cuda,nvtx,pcie --output=colossus_telemetry \
    python3 -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-V2-Lite \
    --device-mapping custom_colossus_map
```

### 3. Key Metrics to Report
To construct the final publication tables:
*   **Decode Latency (ms/token):** Inter-token latency during text generation.
*   **Time-to-First-Token (TTFT):** Prompt processing latency.
*   **Throughput (tokens/sec):** Maximum concurrent generation speed under batch constraints.
*   **Perplexity (PPL):** Evaluated on WikiText-2 to confirm that offloaded COLOSSUS serving matches GPU-native perplexity.
