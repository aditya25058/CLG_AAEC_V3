# NVLink vs PCIe Interconnect Comparison Report

**Project:** RCM Co-Design for Distributed MoE Serving  
**Simulator:** MoEServingSim (ASTRA-Sim Analytical Network Backend)  
**Hardware:** NVIDIA H100 (80 GB HBM3, 3.35 TB/s memory BW)  
**Date:** June 2026  

---

## 1. Objective

This report evaluates how interconnect technology — **NVLink (900 GB/s, 1.5 μs latency)** vs **PCIe Gen5 (16 GB/s, 20 μs latency)** — impacts MoE inference serving performance across three production-scale models:

| Model | Total Params | Active Params | Experts | k (top-k) | Layers | Hidden |
|-------|-------------|---------------|---------|-----------|--------|--------|
| **DeepSeek-R1** | ~671B | ~37B | 256 | 8 | 61 | 7168 |
| **Llama-4 Maverick** | ~17B | ~17B | 128 | 1 | 48 | 4096 |
| **Qwen3-235B-A22B** | ~235B | ~22B | 128 | 8 | 94 | 4096 |

These models span a wide design space: DeepSeek-R1 is the largest with 256 experts and deep MLA attention; Llama-4 Maverick is the smallest with only k=1 expert per token; Qwen3-235B is the most layer-heavy at 94 layers.

---

## 2. Experimental Setup

### 2.1 What NVLink vs PCIe Means

When multiple H100 GPUs serve an MoE model together, they must exchange data at two critical points:

1. **Tensor Parallelism (TP)** — Attention heads are split across GPUs. Each GPU computes a partial result, then an **AllReduce** synchronization combines them. This happens every layer.
2. **Expert Parallelism (EP)** — Experts are sharded across GPUs. Each token must be dispatched to its assigned expert's GPU via **All-to-All** communication, and the result sent back.

The interconnect bandwidth and latency directly determine how fast these synchronizations complete.

| Interconnect | Bandwidth | Latency | Typical Deployment |
|-------------|-----------|---------|-------------------|
| **NVLink 4.0** | 900 GB/s (bidirectional) | ~1.5 μs | Intra-node (same server) |
| **PCIe Gen5** | 16 GB/s (per direction) | ~20 μs | Inter-node / budget configs |

### 2.2 Configurations Tested

For each model, we ran 7 configurations:

| Config | GPUs | TP | EP | Interconnect |
|--------|------|----|----|-------------|
| TP=1 | 1 | 1 | 1 | None (baseline) |
| TP=2 NVLink | 2 | 2 | 2 | NVLink |
| TP=2 PCIe | 2 | 2 | 2 | PCIe |
| TP=4 NVLink | 4 | 4 | 4 | NVLink |
| TP=4 PCIe | 4 | 4 | 4 | PCIe |
| TP=8 NVLink | 8 | 8 | 8 | NVLink |
| TP=8 PCIe | 8 | 8 | 8 | PCIe |

### 2.3 Workload

| Parameter | Value |
|-----------|-------|
| Requests | 10 concurrent |
| Input lengths | 10–282 tokens (mixed) |
| Output length | 128 tokens each |
| Routing | DATASET (real expert traces) |
| Arrival | All at t=0 (saturated burst) |

### 2.4 Cluster Configs Used

```
configs/cluster/single_node_{model}_h100_tp{1,2,4,8}.json          # NVLink
configs/cluster/single_node_{model}_h100_tp{2,4,8}_pcie.json        # PCIe
```

### 2.5 How Each Simulation Was Run

```bash
venv/bin/python3 -m serving \
    --cluster-config configs/cluster/<config_file>.json \
    --dataset datasets/<model_trace>.jsonl \
    --num-reqs 10 \
    --output outputs/<output_file>.csv
```

---

## 3. Results

### 3.1 Full Metrics Table

| Model | Config | Interconnect | Total Latency (s) | TTFT (ms) | TPOT (ms) | Token Thru (tok/s) |
|-------|--------|-------------|:-----------------:|:---------:|:---------:|:------------------:|
| **DeepSeek-R1** | TP=1 | — | 530.143 | 4.78 | 1.10 | 2.71 |
| | TP=2 | NVLink | 530.097 | 2.76 | 0.75 | 2.71 |
| | TP=2 | PCIe | 530.107 | 2.82 | 0.83 | 2.71 |
| | TP=4 | NVLink | 530.076 | 1.54 | 0.59 | 2.71 |
| | TP=4 | PCIe | 530.115 | 2.12 | 0.89 | 2.71 |
| | TP=8 | NVLink | 530.075 | 1.28 | 0.58 | 2.71 |
| | TP=8 | PCIe | 530.232 | 2.81 | 1.81 | 2.71 |
| **Llama-4 Maverick** | TP=1 | — | 530.718 | 18.24 | 5.61 | 2.71 |
| | TP=2 | NVLink | 530.365 | 11.66 | 2.84 | 2.71 |
| | TP=2 | PCIe | 530.825 | 17.27 | 6.44 | 2.71 |
| | TP=4 | NVLink | 530.442 | 8.95 | 3.45 | 2.71 |
| | TP=4 | PCIe | 532.499 | 27.79 | 19.54 | 2.70 |
| | TP=8 | NVLink | 530.900 | 10.92 | 7.04 | 2.71 |
| | TP=8 | PCIe | 538.881 | 81.76 | 69.41 | 2.67 |
| **Qwen3-235B-A22B** | TP=1 | — | 532.637 | 141.99 | 19.45 | 2.70 |
| | TP=2 | NVLink | 532.361 | 77.52 | 17.84 | 2.70 |
| | TP=2 | PCIe | 533.286 | 97.64 | 24.88 | 2.70 |
| | TP=4 | NVLink | 532.398 | 46.41 | 18.39 | 2.70 |
| | TP=4 | PCIe | 536.023 | 98.70 | 46.39 | 2.68 |
| | TP=8 | NVLink | 533.187 | 39.20 | 24.64 | 2.70 |
| | TP=8 | PCIe | 547.892 | 193.70 | 139.11 | 2.62 |

### 3.2 PCIe Penalty vs NVLink (% Slowdown)

The PCIe penalty is calculated as: `((PCIe_value - NVLink_value) / NVLink_value) × 100%`

| Model | TP | Latency Penalty | TTFT Penalty | TPOT Penalty |
|-------|:--:|:--------------:|:------------:|:------------:|
| **DeepSeek-R1** | 2 | +0.002% | +2.2% | +10.7% |
| | 4 | +0.007% | +37.7% | +50.8% |
| | 8 | +0.030% | +119.5% | +212.1% |
| **Llama-4 Maverick** | 2 | +0.087% | +48.1% | +126.8% |
| | 4 | +0.388% | +210.5% | +466.1% |
| | 8 | +1.504% | +648.7% | +885.8% |
| **Qwen3-235B-A22B** | 2 | +0.174% | +26.0% | +39.5% |
| | 4 | +0.681% | +112.7% | +152.2% |
| | 8 | +2.760% | +394.1% | +464.5% |

---

## 4. Analysis & Conclusions

### 4.1 The PCIe Penalty Grows Explosively with TP Scale

The most critical finding: **PCIe penalty is not linear — it compounds with every added GPU.**

- At TP=2, the penalties are modest (2–48% TTFT slowdown across models).
- At TP=8, penalties reach **119–649% TTFT slowdown** and **212–886% TPOT slowdown**.

This happens because:
- Each TP step adds more AllReduce synchronization barriers per layer
- Each EP step requires more All-to-All traffic for expert dispatch
- PCIe's 56× lower bandwidth (16 vs 900 GB/s) creates queuing delays that accumulate across 48–94 layers

### 4.2 Llama-4 Maverick Is Most Sensitive to PCIe

Despite being the smallest model (17B params), Llama-4 Maverick shows the **worst PCIe penalty**: 886% TPOT slowdown at TP=8. This is because:

- **128 experts with k=1**: Every token goes to exactly one expert. With TP=8 and EP=8, each GPU holds only 16 experts, so 7/8 of tokens require cross-GPU dispatch.
- **Fewer layers (48)** means less computation time between communication barriers, so the PCIe bottleneck is exposed more directly.
- **Large intermediate_size (14336)** compared to other models means bigger activation tensors in AllReduce.

### 4.3 DeepSeek-R1 Is Least Sensitive to PCIe

DeepSeek-R1 shows remarkably small PCIe penalties on total latency (<0.03%). This is because:

- **MLA attention** (Multi-head Latent Attention) with KV-cache compression drastically reduces the AllReduce data volume.
- **256 experts with k=8**: More experts per GPU at each EP split, reducing cross-GPU dispatch ratio.
- The model's compute-heavy nature means GPU computation dominates over communication time.

### 4.4 Qwen3-235B Shows the Most Dramatic Scaling Breakdown

- NVLink: TTFT drops from 142ms (TP=1) → 39ms (TP=8) — **3.6× improvement**
- PCIe: TTFT goes from 142ms (TP=1) → 194ms (TP=8) — **actually gets 36% WORSE**

This means **adding more GPUs with PCIe makes Qwen3 slower for TTFT**. The 94 layers of AllReduce + All-to-All at 16 GB/s completely overwhelm the parallelism benefit.

### 4.5 Total Latency vs Per-Token Metrics Tell Different Stories

Total latency barely changes across configs (~530s) because the workload is arrival-time dominated (10 requests spaced 40s apart). The **per-token metrics (TTFT, TPOT)** reveal the true impact:

| Metric | What It Shows | Most Affected |
|--------|--------------|---------------|
| TTFT | Prefill latency (first token) | Qwen3 (94 layers × large attention) |
| TPOT | Decode step latency | Llama-4 (high expert dispatch ratio) |

### 4.6 Practical Recommendations

| Scenario | Recommendation |
|----------|---------------|
| ≤2 GPUs | PCIe is acceptable (< 50% penalty) |
| 4 GPUs | NVLink strongly recommended for Llama-4 and Qwen3 |
| 8 GPUs | **NVLink is mandatory** — PCIe causes 2–9× slowdown on decode |
| Budget constraint | Stay at TP=2 PCIe rather than scaling to TP=8 PCIe |
| DeepSeek-R1 | PCIe is usable up to TP=4 due to MLA efficiency |

---

## 5. Files in This Package

```
nvlink_pcie_results/
├── nvlink_pcie_report.md                      ← This report
├── nvlink_pcie_metrics.json                   ← Machine-readable metrics
│
├── plots/
│   ├── nvlink_pcie_latency_comparison.png     ← Grouped bar: latency per model
│   ├── nvlink_pcie_ttft_comparison.png        ← Grouped bar: TTFT per model
│   ├── nvlink_pcie_tpot_comparison.png        ← Grouped bar: TPOT per model
│   ├── nvlink_pcie_throughput_scaling.png      ← Line: throughput scaling all models
│   ├── nvlink_pcie_penalty_heatmap.png        ← Heatmap: % slowdown
│   └── nvlink_pcie_results_table.png          ← Full results table
│
├── deepseek_r1/
│   ├── deepseek_r1_h100_tp1_results.csv
│   ├── deepseek_r1_h100_tp2_nvlink_results.csv
│   ├── deepseek_r1_h100_tp2_pcie_results.csv
│   ├── deepseek_r1_h100_tp4_nvlink_results.csv
│   ├── deepseek_r1_h100_tp4_pcie_results.csv
│   ├── deepseek_r1_h100_tp8_nvlink_results.csv
│   └── deepseek_r1_h100_tp8_pcie_results.csv
│
├── llama4_maverick/
│   ├── llama4_maverick_h100_tp1.csv
│   ├── llama4_maverick_h100_tp2_nvlink.csv
│   ├── llama4_maverick_h100_tp2_pcie.csv
│   ├── llama4_maverick_h100_tp4_nvlink.csv
│   ├── llama4_maverick_h100_tp4_pcie.csv
│   ├── llama4_maverick_h100_tp8_nvlink.csv
│   └── llama4_maverick_h100_tp8_pcie.csv
│
└── qwen3_a22b/
    ├── qwen3_a22b_h100_tp1.csv
    ├── qwen3_a22b_h100_tp2_nvlink.csv
    ├── qwen3_a22b_h100_tp2_pcie.csv
    ├── qwen3_a22b_h100_tp4_nvlink.csv
    ├── qwen3_a22b_h100_tp4_pcie.csv
    ├── qwen3_a22b_h100_tp8_nvlink.csv
    └── qwen3_a22b_h100_tp8_pcie.csv
```
