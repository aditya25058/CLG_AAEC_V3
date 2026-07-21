# COLOSSUS Evaluation Plan (OSDI/SOSP/MLSys Grade)
## Comprehensive Benchmarking Strategy

---

## Evaluation Philosophy

A top-tier systems paper must answer **six** distinct questions through its evaluation. Each question maps to a section below. Every experiment is designed to prove or disprove a specific hypothesis about COLOSSUS's design.

---

## Section A: End-to-End Serving Benchmarks (The Bottom Line)

> **Question:** Does COLOSSUS deliver faster, cheaper inference than existing systems on real workloads?

### A.1 Hardware Configurations

| Config ID | GPU | VRAM | Host RAM | Interconnect | Target |
|---|---|---|---|---|---|
| **H100-Full** | 1× NVIDIA H100 SXM5 | 80 GB HBM3 | 512 GB DDR5 | PCIe Gen5 ×16 | Server-grade |
| **A100-Constrained** | 1× NVIDIA A100 | 40 GB HBM2e | 256 GB DDR4 | PCIe Gen4 ×16 | Memory-constrained server |
| **RTX4090-Edge** | 1× NVIDIA RTX 4090 | 24 GB GDDR6X | 64 GB DDR5 | PCIe Gen4 ×16 | Edge/consumer |
| **2×H100-Multi** | 2× NVIDIA H100 NVL | 2×80 GB HBM3 | 1 TB DDR5 | NVLink + PCIe Gen5 | Multi-GPU inter-node |

### A.2 Models Under Test

| Model | Total Params | Active Params | Experts | Top-K | Why |
|---|---|---|---|---|---|
| **Qwen3-30B-A3B** | 30.5B | 3.3B | 128 per layer | 8 | Primary target (high expert count, extreme sparsity) |
| **Qwen3-235B-A22B** | 235B | 22B | 128 per layer | 8 | Scale stress test |
| **DeepSeek-V3-671B** | 671B | 37B | 256 per layer | 8 | Cross-architecture generalization |
| **Mixtral-8x7B** | 46.7B | 12.9B | 8 per layer | 2 | Low-expert-count control (tests if COLOSSUS helps when routing is already simple) |

### A.3 Workloads

| Workload | Description | Why |
|---|---|---|
| **ShareGPT** | Real multi-turn chat conversations (variable length, diverse topics) | Production-representative conversational serving |
| **LongBench** | Long-context QA and summarization (4K–32K tokens) | Tests cache pressure under long sequences |
| **HumanEval / MBPP** | Code generation (short, bursty, high-entropy routing) | High expert churn stress test |
| **MMLU (5-shot)** | Multi-choice knowledge QA (short prompts, deterministic) | Accuracy parity verification |
| **Synthetic Uniform** | Random tokens with uniform expert distribution | Worst-case: no temporal locality, no prediction signal |

### A.4 Baselines (8 Systems, Fairly Tuned)

| # | System | Paper / Source | Why It Must Be Included |
|---|---|---|---|
| 1 | **vLLM (Dense Baseline)** | SOSP'23 | Industry-standard serving engine. Shows the "full GPU" upper bound. |
| 2 | **vLLM + EP (Expert Parallelism)** | Standard distributed MoE | The default multi-GPU MoE serving approach. Measures All-to-All networking cost. |
| 3 | **PowerInfer-2** | MobiSys'25 | CPU-GPU hybrid neuron-level offloading. Direct competitor at neuron granularity. |
| 4 | **ProMoE** | arXiv:2410.22134 | Proactive caching with reordered execution. Tests whether reordering outperforms overlapping. |
| 5 | **HOBBIT** | arXiv:2411.01433 | Mixed-precision expert offloading. Tests whether quantized full-expert loading beats sparse packet pulling. |
| 6 | **DALI** | arXiv:2602.03495 | Workload-aware offloading with residual-based prediction. Tests prediction quality head-to-head. |
| 7 | **AdapMoE** | 2025 | Dynamic cache allocation via DP. Tests cache management strategy. |
| 8 | **Fate** | 2025 | Edge-optimized shallow-favoring cache. Tests edge deployment scenario. |

### A.5 Metrics & Tables to Produce

**Table E2E-1: Decode Latency (ms/token)**

| System | Qwen3-30B (H100) | Qwen3-30B (RTX4090) | Qwen3-235B (2×H100) | DeepSeek-V3 (2×H100) |
|---|---|---|---|---|
| vLLM (EP) | — | — | — | — |
| PowerInfer-2 | — | — | — | — |
| ProMoE | — | — | — | — |
| HOBBIT | — | — | — | — |
| DALI | — | — | — | — |
| **COLOSSUS** | — | — | — | — |

**Table E2E-2: Throughput (tokens/s) at Batch Sizes {1, 4, 16, 64}**

**Table E2E-3: Time-to-First-Token (TTFT) for Prefill**

**Table E2E-4: Peak GPU VRAM Usage (GB)**

**Table E2E-5: Total PCIe/NVLink Data Volume (GB per 1000 tokens)**

---

## Section B: Component Micro-Benchmarks (Why It Works)

> **Question:** Which specific hardware/software component is responsible for the gains?

### B.1 PCIe/NVLink Transfer Efficiency

| Experiment | What It Measures |
|---|---|
| **B.1.1: Scattered vs. Coalesced DMA** | Transfer latency and effective bandwidth for N={32, 64, 128, 256, 512} neuron columns, comparing individual `cudaMemcpy` calls vs. single coalesced SGL transfer. Already completed on H100 (101.7× speedup). |
| **B.1.2: SGL Descriptor Batching** | Measure the MMIO doorbell overhead as a function of SGL list length. Prove that batching 4 layers of misses into a single SGL keeps doorbell count below NIC TPS limits. |
| **B.1.3: NVLink P2P Write vs. Read** | Latency and throughput comparison of `cudaMemcpyPeerAsync` (read) vs. direct pointer write (push) for 12 KB, 64 KB, and 256 KB payloads across NVLink 4.0. |

### B.2 Triton Gather-GEMM Kernel Performance

| Experiment | What It Measures |
|---|---|
| **B.2.1: Gather-GEMM vs. Dense GEMM** | Compare Triton Gather-GEMM kernel (with index map) vs. standard cuBLAS dense GEMM for equivalent active neuron counts {64, 128, 256, 384}. Report TFLOPS and memory bandwidth utilization. |
| **B.2.2: Gather-GEMM vs. Sparse GEMM** | Compare against cuSPARSE structured sparsity (2:4) to show that packet-level gathering outperforms generic sparse kernels for this workload shape. |
| **B.2.3: Index Map Overhead** | Measure the overhead of the indirect addressing (index map load + pointer arithmetic) as a percentage of total kernel execution time. Target: <5%. |

### B.3 Prefetcher Accuracy & Bandwidth

| Experiment | What It Measures |
|---|---|
| **B.3.1: Expert Prediction Accuracy** | Top-1 and Top-2 expert prediction accuracy of the Pre-Attention Linear Predictor vs. Cross-Layer Markov (old NAWP) vs. Oracle (perfect prediction). Broken down per layer. |
| **B.3.2: Column Prediction Precision/Recall** | Measure how well the temporal + static prior predicts the actual active column set. Report Precision, Recall, F1 per layer. |
| **B.3.3: Prefetch Waste Ratio** | `Wasted Bytes / Total Prefetched Bytes` for each policy. Target: <15% waste for the hybrid prefetcher. |
| **B.3.4: Confidence Threshold Sweep** | Sweep α ∈ {0.0, 0.5, 1.0, 1.5, 2.0} for the confidence gating threshold. Plot hit rate vs. wasted bandwidth Pareto frontier. |

---

## Section C: Ablation Studies (What Matters)

> **Question:** How much does each COLOSSUS component contribute individually?

### C.1 Feature Ablation Matrix

| Variant | ADETR | HNC Cache | Pre-Attn Predictor | Confidence Gate | SA-FFN | Expected Effect |
|---|---|---|---|---|---|---|
| **Full COLOSSUS** | ✓ | ✓ | ✓ | ✓ | ✓ | Best overall |
| **–ADETR** | ✗ | ✓ | ✓ | ✓ | ✓ | Scattered memory access → lower kernel TFLOPS |
| **–HNC (flat cache)** | ✓ | ✗ | ✓ | ✓ | ✓ | No layer-heterogeneous budget → lower hit rates on middle layers |
| **–Predictor (demand only)** | ✓ | ✓ | ✗ | ✗ | ✓ | No prefetching → high PCIe stalls |
| **–Confidence Gate** | ✓ | ✓ | ✓ | ✗ | ✓ | No admission control → massive bandwidth waste |
| **–SA-FFN (blocking fetch)** | ✓ | ✓ | ✓ | ✓ | ✗ | Synchronous fetch → GPU idle bubbles |
| **Naive Baseline** | ✗ | ✗ | ✗ | ✗ | ✗ | Full expert offloading (LRU cache, demand fetch) |

For each variant, report: Decode latency (ms/token), cache hit rate (%), wasted prefetch (GB), and Triton kernel TFLOPS.

### C.2 Cache Policy Ablation

| Policy | Description |
|---|---|
| LRU (standard) | Evict least recently used across all layers uniformly |
| LFU | Evict least frequently used |
| Layer-Heterogeneous LRU (COLOSSUS) | Per-layer LRU with entropy-weighted capacity |
| Shallow-Favoring (Fate-style) | Reserve more cache for early layers |
| Oracle Bélády | Evict the item whose next use is farthest in the future (theoretical optimum) |

Report hit rate and latency for each policy. Oracle Bélády provides the theoretical ceiling.

---

## Section D: Scalability Studies (How Far It Goes)

> **Question:** Does COLOSSUS scale with model size, GPU count, batch size, and sequence length?

### D.1 Model Size Scaling

Plot decode latency as a function of total model parameters: {Mixtral-8×7B (47B), Qwen3-30B, Qwen3-235B, DeepSeek-V3 (671B)}. 
*   **Hypothesis:** COLOSSUS's latency scales sub-linearly with model size because the active parameter footprint (and thus the transfer volume) grows much slower than total parameters in MoE architectures.

### D.2 GPU Count Scaling (Strong Scaling)

Fix the model (Qwen3-235B) and increase GPU count: {1, 2, 4, 8} H100 GPUs.
*   **Metric:** Throughput (tokens/s) and per-GPU VRAM usage.
*   **Hypothesis:** COLOSSUS scales near-linearly because weight-pulling traffic is independent of batch size, unlike EP's All-to-All which grows with GPU count.

### D.3 Batch Size Scaling

Fix the model and hardware (Qwen3-30B on 1× H100). Sweep batch size: {1, 2, 4, 8, 16, 32, 64}.
*   **Metric:** Throughput, latency, and total PCIe transfer volume.
*   **Hypothesis:** COLOSSUS's PCIe traffic stays roughly constant (weight-pulling is per-layer, not per-token), while EP's All-to-All traffic scales linearly.

### D.4 Sequence Length Scaling

Fix the model and batch size. Sweep sequence length: {512, 1024, 2048, 4096, 8192, 16384, 32768}.
*   **Metric:** Prefetch hit rate and working set size (unique experts/columns accessed).
*   **Hypothesis:** Longer sequences exhibit higher temporal locality (same experts reappear), improving cache hit rates.

---

## Section E: Accuracy Verification (Lossless Guarantee)

> **Question:** Does COLOSSUS produce bit-identical outputs to the unmodified model?

### E.1 Exact Match Verification

| Test | Method | Target |
|---|---|---|
| **Logit Cosine Similarity** | Compare COLOSSUS output logits vs. HuggingFace reference on 1000 ShareGPT prompts | **1.0000** (exact) |
| **Top-1 Token Agreement** | Fraction of tokens where COLOSSUS and reference select the same next token | **100.00%** |
| **Perplexity Delta** | Compute perplexity on WikiText-2 for both COLOSSUS and reference | **Δ = 0.0000** |
| **Greedy Decode Match** | Generate 512 tokens with greedy decoding; compare output strings | **Byte-identical** |

### E.2 Downstream Task Accuracy

| Benchmark | Metric | COLOSSUS Score | Reference Score | Delta |
|---|---|---|---|---|
| **MMLU (5-shot)** | Accuracy (%) | — | — | Must be 0.0 |
| **HumanEval** | pass@1 (%) | — | — | Must be 0.0 |
| **GSM8K** | Accuracy (%) | — | — | Must be 0.0 |
| **ARC-Challenge** | Accuracy (%) | — | — | Must be 0.0 |

> [!IMPORTANT]
> Because COLOSSUS's SA-FFN is mathematically exact (all active neurons are computed, just in a different scheduling order), the accuracy delta **must be exactly zero** for all benchmarks. Any non-zero delta indicates a bug, not an inherent trade-off.

---

## Section F: Sensitivity Analysis (Robustness)

> **Question:** How sensitive is COLOSSUS to its configuration parameters?

### F.1 Cache Size Sensitivity

Sweep L2 cache capacity per expert: {32, 64, 128, 256, 384, 512, 768} columns.
*   **Plot:** Hit rate and decode latency vs. cache size.
*   **Find:** The "knee" where additional cache provides diminishing returns.

### F.2 Confidence Threshold (α) Sensitivity

Sweep α ∈ {0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0}.
*   **Plot:** Pareto frontier of hit rate vs. wasted prefetch bandwidth.
*   **Find:** The optimal α that maximizes (hit rate − waste penalty).

### F.3 Packet Size Sensitivity

Sweep the neuron-channel packet granularity: {1 column (10 KB), 4 columns (40 KB), 16 columns (160 KB), 64 columns (640 KB), full expert (9.44 MB)}.
*   **Plot:** DMA efficiency (effective bandwidth) vs. packet size.
*   **Find:** The sweet spot between transaction overhead (too small) and wasted transfer (too large).

### F.4 Link Bandwidth Sensitivity

Sweep PCIe/network link speed: {2, 4, 8, 16, 32, 64, 128} GB/s.
*   **Plot:** Decode latency vs. link bandwidth for COLOSSUS and all baselines.
*   **Find:** The crossover point where COLOSSUS's prefetching advantage disappears (i.e., the link is fast enough that demand-driven fetching has no stalls).

---

## Summary: What Each Section Proves

| Section | Core Claim Validated |
|---|---|
| **A (End-to-End)** | COLOSSUS delivers lower latency and higher throughput than all baselines on real workloads |
| **B (Micro-benchmarks)** | The performance comes from specific hardware-level optimizations (coalesced DMA, Gather-GEMM, prediction accuracy) |
| **C (Ablation)** | Every component (ADETR, HNC, predictor, confidence gate, SA-FFN) contributes measurably; removing any one degrades performance |
| **D (Scalability)** | COLOSSUS scales with model size, GPU count, batch size, and sequence length |
| **E (Accuracy)** | COLOSSUS is provably, bit-for-bit lossless |
| **F (Sensitivity)** | COLOSSUS is robust across a wide range of configuration parameters and hardware conditions |
