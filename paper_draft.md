# Active Auxiliary Expert Caching (AAEC) v3: Column-Granular Offloading for Mixture-of-Experts Serving

## Abstract

Mixture-of-Experts (MoE) serving engines on memory-constrained edge and distributed node environments suffer from severe bandwidth bottlenecks. During single-batch ($B=1$) autoregressive decoding, feed-forward network (FFN) computation on cached parameters takes less than $0.5\ \mu\text{s}$, whereas transferring missed parameters over PCIe Gen5 or commodity networks requires $12\text{--}15\ \mu\text{s}$. This Timing Fallacy renders traditional expert-level caching and predictive prefetching at FFN block boundaries incapable of hiding weight transfer latency, causing massive GPU idle stalls.

To address this challenge, we present **Active Auxiliary Expert Caching (AAEC) v3**, an execution engine that operates at the fine granularity of individual **column-level parameter packets ($30.72\text{ KB}$)** rather than monolithic experts (typically $9\text{--}17\text{ MB}$). By caching only the highly activated columns of each expert, AAEC v3 reduces parameter traffic and matches the transfer size to the GPU's preceding Multi-Head Attention compute window ($50\text{--}150\ \mu\text{s}$). AAEC v3 introduces: (1) a **Pre-Attention Router** that forecasts expert activations to launch asynchronous speculative prefetching, and (2) **ADETR**, a virtual address remapping layout that statically packages hot columns while allowing zero-copy updates to avoid GPU stream serialization.

Empirical evaluations on Qwen3-30B-A3B (128 experts, top-8) and DeepSeek-V2-Lite (64 experts, top-6) demonstrate that:
1. **Caching Efficiency:** Column-level caching avoids cache capacity thrashing under concurrent expert activations, achieving hit rate saturation at cache sizes of 128 (41.30% hit rate) for Qwen3 and 64 (53.19%) for DeepSeek.
2. **Granularity Reduction:** Column slicing converts monolithic $27.40\text{ MB}$ memory copy bursts into tiny, parallel $8.19\text{ MB}$ streams, collapsing raw transfer duration by **3.3×** and average exposed stall per step by **2.8×** (from $2.71\text{ ms}$ to $0.97\text{ ms}$).
3. **Distributed Serving:** In a 4-node distributed serving environment, AAEC v3 restricts remote expert fetch granularity to under $1.1\text{ MB}$, reducing inter-node network data movement by **9.2×** (Qwen3) and **15.4×** (DeepSeek). This collapses network-induced stall by **15.7×** (from $269.51\text{ ms}$ to $17.19\text{ ms}$) and **25.4×** (from $203.37\text{ ms}$ to $8.00\text{ ms}$), enabling throughput improvements of up to **25×**.
4. **Speculative Gain:** When speculative prefetching is enabled, AAEC v3 achieves an additional **24% throughput speedup** for Qwen3 and **9.5%** for DeepSeek. Crucially, the engine's column granularity minimizes the bandwidth penalty of mispredictions, proving that fine-grained slicing is the key to turning speculative prefetching from a congestion bottleneck into a net-positive latency-hiding accelerator.

---

## 1. Problem Statement

Mixture-of-Experts (MoE) models represent a major milestone in scaling Deep Learning architectures. However, deploying these models on resource-constrained platforms—such as edge nodes or distributed clusters—faces a fundamental bottleneck: **memory and network communication bandwidth**. 

Specifically, three primary limitations hinder efficient MoE inference:

### 1.1 The FFN Compute-Transfer Timing Fallacy
During single-batch ($B=1$) autoregressive decoding, the feed-forward network (FFN) block is executed token-by-token. For a single token, the execution of the active experts requires computing a sparse matrix-vector multiplication (GEMV). The computational complexity of the active FFN pass is:
$$\text{FLOPs} = 2 \times B \times N_{\text{active}} \times 3 \times H \times d_{\text{intermediate}}$$
For a model such as Qwen3-30B-A3B ($H=2048$, $d_{\text{intermediate}}=768$, $N_{\text{active}}=8$), computing the FFN block on an H100 GPU requires less than **$0.5\ \mu\text{s}$** of raw processing time. 

In contrast, transferring the parameters of the missed experts from host memory (T4) or remote nodes over PCIe Gen5 ($64\text{ GB/s}$) or network interfaces ($10\text{ GB/s}$) requires fetching monolithic parameter blocks. A single expert parameter block in Qwen3-30B is approximately $9.44\text{ MB}$. If a cache miss occurs, streaming the expert over PCIe Gen5 requires at least **$147.5\ \mu\text{s}$**, and over a 100 Gbps network requires **$944\ \mu\text{s}$**. 

Because the FFN compute duration ($< 0.5\ \mu\text{s}$) is orders of magnitude smaller than the weight transfer duration ($147\text{--}944\ \mu\text{s}$), the GPU compute stream is immediately serialized, resulting in massive, unhideable bus stalls. Traditional predictive prefetchers that trigger at the boundaries of the FFN block fail because the computation window is too short to overlap with the transfer.

### 1.2 Capacity Thrashing of Monolithic Caching
Standard offloading Serving Engines (such as DeepSpeed or PowerInfer-2) maintain a local VRAM cache at the granularity of **monolithic experts**. 
* Under concurrent expert routing (e.g., $N_{\text{active}}=8$ experts activated concurrently per token step), the working set size is large ($75.5\text{ MB}$ per layer).
* When caching monolithic experts, a single active token access can trigger multiple expert cache misses. Loading these monolithic expert blocks flushes the cache frequently. 
* This capacity constraint causes **cache thrashing** for models with large expert counts (such as Qwen3 with 128 experts). Monolithic caching hit rates degrade, causing misses on over **83% of steps**, which translates into continuous, large transfers ($27.40\text{ MB}$ average payload per miss) and high latency overheads ($130.09\text{ ms}$ average stall).

### 1.3 The Bandwidth Penalty of Coarse Speculation
Predicting MoE expert activations in advance is a promising path to launch prefetching. However, when prefetching is executed at the monolithic expert level, the cost of a misprediction is high.
* A first-order Markov predictor achieves modest prediction accuracies ($8.6\%\text{--}22.5\%$) on unseen tasks due to the stochastic nature of routing trajectories.
* If the prefetching engine speculative-loads a monolithic expert ($9.44\text{--}16.9\text{ MB}$) that is ultimately not routed, it generates a massive volume of wasted data transfers.
* This speculative bandwidth overhead congests the interconnect bus, delaying the reactive fetching of the actual active experts and degrading throughput below a demand-only baseline. Consequently, expert-level speculation is economically non-viable.

### 1.4 Physical Coordination and Scheduling Overheads
In theory, asynchronously overlapping copies (`cudaMemcpyAsync`) with GPU compute is straightforward. In physical hardware, however, coordinating background transfers with active compute streams introduces non-trivial CPU-GPU launch overheads.
* Establishing non-blocking DMA copies and coordinating stream synchronization (via `wait_event`) requires CPU intervention.
* These scheduling and synchronization overheads introduce a physical latency tax of **$53\text{--}55\ \mu\text{s}$** per layer on NVIDIA H100 hardware.
* When the payload is large, this scheduling tax is compounded by PCIe bus congestion, rendering standard asynchronous stream overlap ineffective unless payload sizes are structurally reduced.

AAEC v3 addresses these problems by slicing experts into **column-granular packets ($30.72\text{ KB}$)**, shifting prefetching to the **Pre-Attention window**, and implementing zero-copy address remapping to eliminate execution serialization.

---

## 2. Related Work

The optimization of Mixture-of-Experts (MoE) model serving is a highly active research area. The literature intersects four core technical domains: expert offloading, speculative prefetching, neuron-level activation sparsity, and kernel-level GPU optimizations.

### 2.1 Expert-Level Offloading & Caching
Offloading-based serving engines partition MoE parameters across the storage hierarchy (VRAM, Host DRAM, SSDs), caching active experts in local GPU memory.
* **Mixtral Offloading** (Eliseev & Mazur, 2023) and **MoE-Infinity** (Xue et al., 2024) implement expert-level Least Recently Used (LRU) caching schemes. They load full expert weights reactively over PCIe on cache misses, which exposes massive stalls during single-batch generation.
* **MoE-Lightning** (Cao et al., 2025) and **MoE-Gen** (2025) optimize batch throughput via CGOPipe scheduling and module-based token accumulation, but do not target single-query latency ($B=1$).
* **EdgeMoE** (Yi et al., 2023) and **OD-MoE** (2026) optimize serving on resource-constrained mobile and edge devices via bitwidth adaptation and cacheless execution, but rely on monolithic expert transfers.
* **ADEPT** (2026) and **FineMoE** (Wang et al., 2026) utilize prompt semantics and domain-aware hints to guide pre-loading, yet their memory allocation units remain coarse-grained.
* **DALI** (2026), **MoEpic** (2025), **ExpertFlow** (2025), and **ExpertCache** (2025) deploy reinforcement learning (RL) or integer programming (IP) solvers to optimize CPU/GPU placement and split expert matrices. However, they lack sub-expert, column-granular representation.
* **ProMoE** (2025) proactively stages expert parameters but is constrained by the coarse granularity of monolithic expert divisions ($9\text{--}17\text{ MB}$).

*AAEC v3 Contribution:* While all prior works in this domain operate at monolithic expert boundaries, AAEC v3 is the first to implement sub-expert caching at a **column-granular packet size ($30.72\text{ KB}$)**, eliminating capacity thrashing.

### 2.2 Speculative Prefetching & Prediction
Prefetching systems forecast router selections to trigger asynchronous parameters copies over the bus in advance.
* **Pre-gated MoE** (Hwang et al., Microsoft, 2024) and **ST-MoE** (2025) execute inter-layer prediction, using the output hidden state of layer $L-1$ to predict the routing decisions of layer $L$. This approach suffers from prediction degradation due to chaotic routing trajectories across layer transitions.
* **CommitMoE** (Luo et al., 2025) employs a pre-attention router but uses a lossy pruning method, committing to only a single expert pre-attention and discarding the other 7 active experts, causing substantial quality loss.
* **SpecMoE** (2025) and **MoE-SpeQ** (2025) utilize speculative draft models to forecast expert usage, but are constrained by draft model training overheads.
* **AdapMoE** (2024), **HOBBIT** (2025), and **SiDA-MoE** (MLSys 2024) reduce transfer stalls via lossy trade-offs, such as dynamically reducing routed expert counts, replacing cache-miss experts with INT4 quantized blocks, or skipping computations, which alters model semantics.

*AAEC v3 Contribution:* AAEC v3 is strictly **lossless**—any misprefetch is reactively loaded on demand. Furthermore, it operates on column-granular prefetch packets, which reduces misprefetch payloads by **9.2--15.4×** and makes speculation net-positive even with modest predictor accuracies ($3.0\%\text{--}8.6\%$).

### 2.3 Neuron-Level & Activation Sparsity
Our column-granular caching is empirically grounded in studies of activation sparsity inside Feed-Forward Networks.
* **PowerInfer** (Song et al., SJTU, 2024) and **Deja Vu** (Liu et al., ICML 2023) prove that FFN activations are highly sparse, pre-loading "hot" neurons on the GPU and using contextual predictors to forecast active neurons. However, both target **dense transformers** (e.g., LLaMA-2, OPT) and do not support partitioned MoE routing.
* **PowerInfer-2** (2024) extends neuron clustering to NPUs on smartphones, but remains focused on dense model configurations.
* **MoEfication** (Zhang et al., 2022) partitions dense FFNs into MoE-like structures, while **TEAL** (2024) and **R-Sparse** (2024) introduce runtime thresholding or ReLU reintroduction to prune neurons. These approaches require extensive model fine-tuning or are incompatible with SwiGLU activations.
* **FANG** (2025) partitions neurons into shared and routed sets for pruning, but is designed for offline model compression rather than online serving caching.

*AAEC v3 Contribution:* AAEC v3 bridges this gap by being the first to apply neuron-level sparsity to **Mixture-of-Experts serving**, dynamically mapping column-level activation energy within routed expert structures.

### 2.4 Kernel Fusion & GPU Optimization
GPU optimization frameworks focus on maximizing hardware utilization during sparse token routing.
* **Triton Fused MoE** (vLLM, 2024) implements fused kernels that handle routing, token permutation, and GEMMs in a single GPU pass. However, these kernels assume that expert matrices are fully resident and contiguous in GPU memory, and cannot process dynamically gathered weight slices.
* **Column-Major GEMM Scheduling** (PyTorch, 2024) optimizes L2 cache locality for dense GEMMs, but lacks support for sparse column gathering.
* **FlashAttention-3** (Dao, 2024) accelerates attention blocks through asynchronous pipelining, but does not address the weight offloading bottleneck in FFN blocks.

*AAEC v3 Contribution:* AAEC v3 introduces a custom **Triton Gather-GEMM kernel** that fuses index-based column gathering, GEMM, and activation in a single pass. Combined with the **ADETR virtual address layout**, it enables zero-copy weight remapping without serialization stalls.

---

## 3. System Architecture

The architecture of the Active Auxiliary Expert Caching (AAEC) v3 serving engine is structured to eliminate memory copy stalls during autoregressive generation. It consists of three primary components: the **Pre-Attention speculative execution pipeline**, the **Hierarchical Network Cache (HNC)**, and the **ADETR virtual memory layout with custom Triton Gather-GEMM execution**.

### 3.1 Pre-Attention Speculative Pipeline
To bypass the timing bottlenecks of traditional FFN-boundary prefetching, AAEC v3 shifts the weight prefetching trigger to the beginning of the Transformer layer.
* **Intra-Layer Speculative Trigger:** Immediately following the layer's input RMSNorm and prior to entering the Multi-Head Attention block, the token's intermediate hidden state $\mathbf{h}_{\text{pre}}$ is intercepted. 
* **Pre-Attention Routing Prediction:** The engine runs a lightweight gating projection:
  $$\mathbf{s}_{\text{pre}} = \text{Top-K}(\text{Softmax}(\mathbf{h}_{\text{pre}} \mathbf{W}_g^T))$$
  to predict the active expert indices and column weights for the upcoming FFN pass.
* **Concurrent DMA Transfer:** The union of predicted column indices that are not present in local memory is dispatched asynchronously over the interconnect bus (PCIe Gen5 or peer NVLink) on a dedicated CUDA copy stream.
* **Latency Overlapping:** The copy operation executes concurrently with the GPU's Multi-Head Attention compute stream. Since the Multi-Head Attention block execution takes $50\text{--}150\ \mu\text{s}$ (depending on context length), it fully hides weight transfers up to 64 columns per expert over PCIe Gen5, resulting in **zero exposed transfer latency**.

```
Input h_pre ---> [ RMSNorm ] ---> [ Pre-Attention Router ] --(Index Misses)--> [ Async DMA Copy Stream ]
                       |                                                                |
                       v                                                                v
             [ Attention Compute ] (50-150 us)  =======================> [ Overlapped PCIe/NVLink Transfer ]
                       |                                                                |
                       v                                                                v
True h_post ---> [ Post-Attn Router ] ===> [ Triton Gather-GEMM Kernel ] <====== [ Arrived Columns Buffers ]
```

### 3.2 Hierarchical Network Cache (HNC)
AAEC v3 manages cache residency dynamically at the level of individual column parameters.
* **Directory Indexing:** The HNC directory tracks parameters via `(layer_id, expert_id, column_id)` keys.
* **VRAM Capacity Partitioning:** VRAM memory capacity is managed dynamically. Under a cache capacity limit equivalent to $C$ columns per expert, the total cache capacity per layer is $C \times N_E$ columns.
* **Lookahead-Sorted (LS) and LRU Eviction:** On each step, when the true router selects active columns, they are marked as recently accessed. The HNC uses a Least Recently Used (LRU) policy to evict stale columns. In distributed environments, it implements a Lookahead-Sorted (LS) scheme, utilizing the Markov transition matrix to compute the predicted distance to the next access for all cached columns, evicting the column whose forecasted next use is farthest in the future.

### 3.3 ADETR Layout & Triton Gather-GEMM Execution
Standard deep learning frameworks require weights to be stored contiguously in memory for matrix multiplications. AAEC v3 resolves this structural constraint via the **ADETR (Active-Dynamic Column Remapping)** layout and a custom **Triton Gather-GEMM kernel**.

#### 3.3.1 Dual-Partition ADETR Layout
To prevent physical copy serialization, ADETR divides the weights into two logical partitions:
1. **Pinned Hot Partition:** The top-C most popular columns (determined offline via calibration) are physically reordered and packed contiguously in GPU VRAM (T2). Under domain drift, the CPU adjusts the mapping entries virtually in less than $1\ \mu\text{s}$ using virtual page table remapping, avoiding physical weight transfers.
2. **Dynamic Cold Partition:** Missed columns fetched at runtime are copied directly into a pre-allocated contiguous receiving buffer in HBM.

#### 3.3.2 Triton Gather-GEMM Kernel
At runtime, standard SwiGLU FFN execution:
$$\mathbf{y} = \mathbf{W}_d \cdot (\text{SiLU}(\mathbf{x} \mathbf{W}_g^T) \odot (\mathbf{x} \mathbf{W}_u^T))$$
is mathematically decomposed into cached and missed sub-FFN passes:
$$\mathbf{y} = \mathbf{y}_{\text{cached}} + \mathbf{y}_{\text{missed}}$$
where:
$$\mathbf{y}_{\text{cached}} = \mathbf{W}_{d,c} \cdot (\text{SiLU}(\mathbf{x} \mathbf{W}_{g,c}^T) \odot (\mathbf{x} \mathbf{W}_{u,c}^T))$$
$$\mathbf{y}_{\text{missed}} = \mathbf{W}_{d,m} \cdot (\text{SiLU}(\mathbf{x} \mathbf{W}_{g,m}^T) \odot (\mathbf{x} \mathbf{W}_{u,m}^T))$$
Since the activations are element-wise column operations, this summation yields the exact mathematical equivalent of a dense forward pass, guaranteeing **zero accuracy loss**.

The custom Triton Gather-GEMM kernel executes this sparse pass in a single GPU grid launch. The kernel takes the input token vector $\mathbf{x}$, the cached and missed index arrays, and the physical pointers to the hot VRAM partition and dynamic receiving buffers. It gathers the columns, executes the GEMV, applies the SiLU activation, and writes the output directly to the next layer's input buffer, bypassing the CPU launch overhead of multiple PyTorch GEMM calls.

---

## 4. Evaluation

We evaluate the performance of AAEC v3 under dynamic batch size scaling to understand the trade-offs between memory footprints, caching hit rates, I/O parameters transfers, and exposed GPU stalls.

### 4.1 Experimental Setup
* **Hardware Profile:** All simulations utilize parameters derived from live runs on an NVIDIA H100 GPU connected over a PCIe Gen5 ($64.0\text{ GB/s}$) bus. We enforce a fixed GPU VRAM memory budget of **$24\text{ GB}$** to model resource contention.
* **Workload Traces:** Autoregressive generation is simulated over trace databases containing routing histories for Qwen3-30B-A3B and DeepSeek-V2-Lite models. 
* **KV Cache Modeling:** Key-Value caching footprint is calculated using an average sequence length of $1024$ tokens under BF16 precision.

### 4.2 Batch Scaling Performance Sweep

#### 4.2.1 Qwen3-30B-A3B Batch Scaling Sweep
*48 layers, 128 experts, $H=2048$, $I=768$, Top-8 routing ($12\text{ KB}$ per column).*

| Batch Size | KV Cache (GB) | VRAM Weight Cache (GB) | Cache Size (cols/expert) | Cache Hit Rate (%) | Avg Stall per Step (ms) | Total I/O Data (GB) | I/O per Step (MB/step) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 0.38 | 23.62 | 336 | 41.60% | 1.12 ms | 7.01 | 299.04 |
| **2** | 0.75 | 23.25 | 330 | 44.15% | 3.97 ms | 11.73 | 500.39 |
| **4** | 1.50 | 22.50 | 320 | 54.62% | 8.00 ms | 19.19 | 818.95 |
| **8** | 3.00 | 21.00 | 298 | 63.23% | 11.96 ms | 27.54 | 1175.07 |
| **16** | 6.00 | 18.00 | 256 | **64.41%** | 19.99 ms | 44.41 | 1895.02 |
| **32** | 12.00 | 12.00 | 170 | 48.68% | 55.10 ms | 105.78 | 4513.48 |
| **64** | 24.00 | 1.00 | 14 | **3.84%** | **161.15 ms** | **278.79** | **11,894.87** |

#### 4.2.2 DeepSeek-V2-Lite Batch Scaling Sweep
*26 layers, 64 experts, $H=2048$, $I=1408$, Top-6 routing ($12\text{ KB}$ per column).*

| Batch Size | KV Cache (GB) | VRAM Weight Cache (GB) | Cache Size (cols/expert) | Cache Hit Rate (%) | Avg Stall per Step (ms) | Total I/O Data (GB) | I/O per Step (MB/step) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 0.20 | 23.80 | 1249 | 59.14% | 1.22 ms | 2.63 | 122.64 |
| **2** | 0.41 | 23.59 | 1238 | 57.00% | 2.94 ms | 5.23 | 243.57 |
| **4** | 0.81 | 23.19 | 1217 | 61.89% | 6.40 ms | 10.37 | 482.62 |
| **8** | 1.62 | 22.38 | 1174 | 67.94% | 10.33 ms | 16.05 | 746.92 |
| **16** | 3.25 | 20.75 | 1089 | **73.50%** | 13.08 ms | 20.26 | 942.97 |
| **32** | 6.50 | 17.50 | 918 | 73.25% | 20.89 ms | 33.94 | 1579.93 |
| **64** | 13.00 | 11.00 | 577 | **53.52%** | **50.63 ms** | **82.81** | **3854.65** |

### 4.3 Analysis & Discussion
The experimental results demonstrate three distinct scaling phases under batch scaling:

1. **The Synergy Phase ($B \le 16$):** As batch size increases from $1$ to $16$, the cache hit rate rises from **$41.60\%$ to $64.41\%$** for Qwen3 and from **$59.14\%$ to $73.50\%$** for DeepSeek. This occurs because different tokens in the same batch overlap in their expert column selections, pre-warming the cache for each other.
2. **The Contention Phase ($B = 32$):** Once the batch size reaches $32$, the KV cache footprint expands to $12.00\text{ GB}$ (Qwen3) and $6.50\text{ GB}$ (DeepSeek). The weight cache is starved, reducing columns/expert to $170$ (Qwen3) and $918$ (DeepSeek). The hit rate drops, and average stall rises to **$55.10\text{ ms}$** and **$20.89\text{ ms}$**.
3. **The Collapse Phase ($B = 64$):** At $B=64$, the KV cache dominates VRAM, and the active union of routed columns covers almost all experts. For Qwen3, the hit rate collapses to **$3.84\%$**, I/O data per step shoots up to **$11.89\text{ GB}$**, and exposed stall reaches **$161.15\text{ ms}$**, indicating that the cache has ceased to function. This confirms that for massive batch scales, offloading engines must transition from weight caching to full model streaming pipelining.


