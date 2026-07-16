#!/usr/bin/env python3
"""
instrument_aaec_real.py — Real-World MoE AAEC Activation Profiler

Instruments Qwen3 and DeepSeek MoE models to capture publication-grade
neuron activation statistics for the AAEC paper.

Usage:
  # Tier 1: BF16 (primary results)
  python3 instrument_aaec_real.py --model Qwen/Qwen3-30B-A3B --num-prompts 100
  python3 instrument_aaec_real.py --model deepseek-ai/DeepSeek-V2-Lite --num-prompts 100

  # Tier 2: NF4 quantized (scale validation)
  python3 instrument_aaec_real.py --model Qwen/Qwen3-235B-A22B --quantize nf4 --num-prompts 50

  # Dry run (verify hooks register, no inference)
  python3 instrument_aaec_real.py --model Qwen/Qwen3-30B-A3B --dry-run

  # CPU mock mode (test logic without GPU)
  python3 instrument_aaec_real.py --mock
"""

import argparse
import json
import math
import os
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn

# =====================================================================
# 1. PROMPT DATASET (Diverse categories for robust statistics)
# =====================================================================
PROMPT_DATASET = [
    # --- Coding ---
    "Write a Python function that implements a binary search tree with insert, delete, and search operations. Include proper error handling.",
    "Implement a concurrent hash map in Go that supports lock-free reads and write-locked updates. Explain the memory ordering guarantees.",
    "Write a CUDA kernel that performs matrix multiplication with shared memory tiling for a 1024x1024 matrix. Optimize for H100 GPU.",
    "Implement a B+ tree index in Rust with support for range queries. Use generics for the key type.",
    "Write a Python decorator that implements exponential backoff retry logic with jitter for API calls.",
    "Design a lock-free queue in C++ using compare-and-swap. Handle the ABA problem.",
    "Implement merge sort for a linked list in Java. Analyze the space complexity.",
    "Write a Redis-compatible LRU cache in Python with O(1) get and put operations.",
    "Implement a bloom filter with configurable false positive rate in Python.",
    "Write a SQL query optimizer that converts correlated subqueries into joins.",
    # --- Mathematics ---
    "Prove that the sum of the first n odd numbers equals n squared using mathematical induction.",
    "Solve the differential equation dy/dx = xy + x using the integrating factor method. Show all steps.",
    "Find the eigenvalues and eigenvectors of the matrix [[3, 1], [1, 3]]. Verify your answer.",
    "Prove that the set of all continuous functions on [0,1] forms a vector space over the reals.",
    "Compute the Fourier transform of e^(-|t|) and interpret the result in the frequency domain.",
    "Derive the closed-form solution for the Fibonacci sequence using the characteristic equation method.",
    "Prove that every bounded monotonic sequence converges using the completeness axiom.",
    "Find the volume of the solid obtained by rotating y = sin(x) about the x-axis from 0 to pi.",
    "Solve the system of linear equations: 2x + 3y - z = 1, x - y + 2z = 5, 3x + y + z = 8.",
    "Prove the Cauchy-Schwarz inequality for inner product spaces.",
    # --- General QA ---
    "Explain how a CPU cache hierarchy works, including L1, L2, and L3 caches. What is cache coherence?",
    "What are the trade-offs between TCP and UDP for real-time video streaming applications?",
    "Describe the differences between ACID and BASE consistency models in distributed databases.",
    "Explain how transformers work, starting from self-attention through to the full encoder-decoder architecture.",
    "What is the difference between optimistic and pessimistic concurrency control in databases?",
    "Explain the CAP theorem and give examples of systems that prioritize each pair of properties.",
    "How does garbage collection work in Java? Compare G1GC with ZGC.",
    "Explain the difference between virtual memory paging and segmentation.",
    "What is the Byzantine Generals Problem and how does PBFT solve it?",
    "Describe how RDMA works and why it matters for datacenter networking.",
    # --- Translation & Multilingual ---
    "Translate the following to French: 'The mixture of experts architecture routes each token to a subset of specialized neural networks.'",
    "Explain the concept of attention mechanisms in neural networks, writing your answer in both English and Chinese.",
    "Translate to German: 'Distributed systems face fundamental trade-offs between consistency, availability, and partition tolerance.'",
    "Write a haiku about machine learning in Japanese, then explain its meaning in English.",
    "Translate to Spanish: 'The cache hit rate determines the effective memory bandwidth utilization of the system.'",
    # --- Creative Writing ---
    "Write a short story about an AI that discovers it can dream. Include dialogue and sensory details.",
    "Compose a sonnet about the beauty of mathematical proofs, following the Shakespearean rhyme scheme.",
    "Write a technical blog post explaining why mixture-of-experts models are the future of efficient AI scaling.",
    "Create a dialogue between two engineers debating whether to use TCP or RDMA for their inference cluster.",
    "Write an abstract for a systems paper about reducing memory bandwidth bottlenecks in MoE serving.",
    # --- Reasoning & Logic ---
    "A farmer has a fox, a chicken, and a bag of grain. He needs to cross a river in a boat that can only carry him and one item. How does he do it?",
    "Three switches control three light bulbs in another room. You can only enter the room once. How do you determine which switch controls which bulb?",
    "There are 100 prisoners and 100 boxes. Each box contains one prisoner's number. Each prisoner can open 50 boxes. How can they maximize survival probability?",
    "You have 12 balls and a balance scale. One ball is heavier. Find it in 3 weighings.",
    "A snail climbs 3 feet up a wall during the day and slides 2 feet down at night. The wall is 30 feet. How many days to reach the top?",
    # --- Long-form Analytical ---
    "Compare and contrast the architectural designs of GPT-4, Llama 3, and Mixtral. Discuss parameter efficiency, training costs, and inference optimization strategies.",
    "Analyze the trade-offs between expert parallelism and tensor parallelism for serving large MoE models. Consider network bandwidth, load balancing, and memory utilization.",
    "Explain the evolution of attention mechanisms from Bahdanau attention to multi-head attention to flash attention. What are the computational complexity improvements at each stage?",
    "Discuss the implications of the scaling laws paper by Kaplan et al. on modern LLM training strategies. How do Chinchilla optimal ratios affect real-world training decisions?",
    "Compare PCIe Gen5, NVLink, and InfiniBand for GPU-to-GPU communication in inference clusters. Include bandwidth numbers and latency characteristics.",
]


# =====================================================================
# 2. ACTIVATION HOOK (SwiGLU Post-Multiplication)
# =====================================================================
@dataclass
class TokenRecord:
    """One record per token per expert invocation."""
    prompt_id: int
    layer: int
    token_pos: int
    expert_id: int
    router_prob: float
    # Active neuron counts at absolute thresholds
    active_eps_1e5: int = 0
    active_eps_1e4: int = 0
    active_eps_1e3: int = 0
    active_eps_1e2: int = 0
    # Energy concentration k values
    energy_k_50: int = 0
    energy_k_70: int = 0
    energy_k_80: int = 0
    energy_k_90: int = 0
    energy_k_95: int = 0
    energy_k_99: int = 0
    energy_k_999: int = 0
    # Total intermediate dim
    intermediate_dim: int = 0
    # Active neuron indices (for Jaccard, working-set, Zipf)
    active_indices: List[int] = field(default_factory=list)
    # Execution latency
    latency_ms: float = 0.0


class SwiGLUHook:
    """
    Hooks after the SwiGLU multiplication: y = SiLU(gate_proj(x)) * up_proj(x).
    Captures the true intermediate activation contributing to down_proj.
    """
    def __init__(self, layer_idx: int, expert_idx: int, prompt_id_ref: list):
        self.layer_idx = layer_idx
        self.expert_idx = expert_idx
        self.prompt_id_ref = prompt_id_ref  # mutable reference to current prompt id
        self.records: List[TokenRecord] = []
        self.router_probs: Dict[int, float] = {}  # token_pos -> prob

    def set_router_probs(self, probs: Dict[int, float]):
        self.router_probs = probs

    def hook_fn(self, module, input, output):
        y = output.detach().float()
        flat_acts = y.view(-1, y.size(-1))  # [num_tokens, intermediate_dim]
        intermediate_dim = flat_acts.size(1)

        for t_pos in range(flat_acts.size(0)):
            abs_act = flat_acts[t_pos].abs()
            total_energy = abs_act.sum().item()
            if total_energy < 1e-12:
                continue

            # A. Absolute thresholds
            a5 = int(torch.sum(abs_act > 1e-5).item())
            a4 = int(torch.sum(abs_act > 1e-4).item())
            a3 = int(torch.sum(abs_act > 1e-3).item())
            a2 = int(torch.sum(abs_act > 1e-2).item())

            # B. Energy concentration
            sorted_mags, sorted_idx = torch.sort(abs_act, descending=True)
            cum_energy = torch.cumsum(sorted_mags, dim=0) / total_energy

            def energy_k(eta):
                k = torch.where(cum_energy >= eta)[0]
                return int(k[0].item()) + 1 if len(k) > 0 else int(intermediate_dim)

            k50 = energy_k(0.50)
            k70 = energy_k(0.70)
            k80 = energy_k(0.80)
            k90 = energy_k(0.90)
            k95 = energy_k(0.95)
            k99 = energy_k(0.99)
            k999 = energy_k(0.999)

            # C. Active indices (using 99% energy concentration)
            active_idx = sorted_idx[:k99].cpu().tolist()

            rec = TokenRecord(
                prompt_id=self.prompt_id_ref[0],
                layer=self.layer_idx,
                token_pos=t_pos,
                expert_id=self.expert_idx,
                router_prob=self.router_probs.get(t_pos, 0.0),
                active_eps_1e5=a5, active_eps_1e4=a4,
                active_eps_1e3=a3, active_eps_1e2=a2,
                energy_k_50=k50, energy_k_70=k70, energy_k_80=k80, energy_k_90=k90,
                energy_k_95=k95, energy_k_99=k99, energy_k_999=k999,
                intermediate_dim=intermediate_dim,
                active_indices=active_idx,
            )
            self.records.append(rec)

    def hook_fn_fused(self, token_indices, intermediate_activations):
        """
        Handles fused expert activations where we have a subset of tokens routed to this expert.
        `token_indices` is a 1D array of token index positions.
        `intermediate_activations` is a 2D tensor of shape [len(token_indices), intermediate_dim].
        """
        y = intermediate_activations.detach().float()
        flat_acts = y.view(-1, y.size(-1))  # [num_tokens_active, intermediate_dim]
        intermediate_dim = flat_acts.size(1)

        for i, t_pos in enumerate(token_indices):
            # t_pos represents the absolute token position in the current batch/sequence
            abs_act = flat_acts[i].abs()
            total_energy = abs_act.sum().item()
            if total_energy < 1e-12:
                continue

            # A. Absolute thresholds
            a5 = int(torch.sum(abs_act > 1e-5).item())
            a4 = int(torch.sum(abs_act > 1e-4).item())
            a3 = int(torch.sum(abs_act > 1e-3).item())
            a2 = int(torch.sum(abs_act > 1e-2).item())

            # B. Energy concentration
            sorted_mags, sorted_idx = torch.sort(abs_act, descending=True)
            cum_energy = torch.cumsum(sorted_mags, dim=0) / total_energy

            def energy_k(eta):
                k = torch.where(cum_energy >= eta)[0]
                return int(k[0].item()) + 1 if len(k) > 0 else int(intermediate_dim)

            k50 = energy_k(0.50)
            k70 = energy_k(0.70)
            k80 = energy_k(0.80)
            k90 = energy_k(0.90)
            k95 = energy_k(0.95)
            k99 = energy_k(0.99)
            k999 = energy_k(0.999)

            # C. Active indices (using 99% energy concentration)
            active_idx = sorted_idx[:k99].cpu().tolist()

            rec = TokenRecord(
                prompt_id=self.prompt_id_ref[0],
                layer=self.layer_idx,
                token_pos=int(t_pos),
                expert_id=self.expert_idx,
                router_prob=self.router_probs.get(int(t_pos), 0.0),
                active_eps_1e5=a5, active_eps_1e4=a4,
                active_eps_1e3=a3, active_eps_1e2=a2,
                energy_k_50=k50, energy_k_70=k70, energy_k_80=k80, energy_k_90=k90,
                energy_k_95=k95, energy_k_99=k99, energy_k_999=k999,
                intermediate_dim=intermediate_dim,
                active_indices=active_idx,
            )
            self.records.append(rec)


class RouterHook:
    """Captures router logits/probabilities for routing entropy calculation."""
    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx
        self.routing_probs: List[np.ndarray] = []  # per-token softmax distributions
        self.expert_selections: List[List[int]] = []  # per-token top-k selections

    def hook_fn(self, module, input, output):
        # Router output varies by architecture; handle both tuple and tensor
        if isinstance(output, tuple):
            logits = output[0] if len(output) > 0 else output
        else:
            logits = output

        if isinstance(logits, torch.Tensor):
            logits = logits.detach().float()
            probs = torch.softmax(logits.view(-1, logits.size(-1)), dim=-1)
            self.routing_probs.append(probs.cpu().numpy())


# =====================================================================
# 3. MODEL LOADER
# =====================================================================
def load_model(model_id: str, quantize: Optional[str] = None):
    """Load model with appropriate precision."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_id} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    load_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto" if torch.cuda.is_available() else None,
    }

    if quantize == "nf4" and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        print("Using NF4 quantization.")
    elif torch.cuda.is_available():
        load_kwargs["torch_dtype"] = torch.bfloat16
        print("Using BF16 precision.")
    else:
        load_kwargs["torch_dtype"] = torch.float32
        print("Using FP32 (CPU mode).")

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()
    return model, tokenizer


# =====================================================================
# 4. HOOK REGISTRATION (Architecture-aware)
# =====================================================================
def register_hooks(model, prompt_id_ref: list):
    """
    Walk the model graph and register SwiGLU post-mult hooks on every
    MoE expert, plus router hooks for entropy calculation.
    Returns (expert_hooks, router_hooks, handle_list).
    """
    expert_hooks: Dict[Tuple[int, int], SwiGLUHook] = {}
    router_hooks: Dict[int, RouterHook] = {}
    handles = []

    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "layers"):
        layers = model.transformer.layers

    if layers is None:
        print("WARNING: Could not find model layers. No hooks registered.")
        return expert_hooks, router_hooks, handles

    for layer_idx, layer in enumerate(layers):
        mlp = None
        if hasattr(layer, "mlp"):
            mlp = layer.mlp
        elif hasattr(layer, "feed_forward"):
            mlp = layer.feed_forward

        if mlp is None:
            continue

        # Detect MoE layer (has .experts attribute)
        experts = None
        if hasattr(mlp, "experts"):
            experts = mlp.experts
        elif hasattr(mlp, "deepseek_experts"):
            experts = mlp.deepseek_experts

        if experts is None:
            continue  # Dense layer, skip

        # Register router hook
        gate = None
        if hasattr(mlp, "gate"):
            gate = mlp.gate
        elif hasattr(mlp, "router"):
            gate = mlp.router

        if gate is not None:
            rh = RouterHook(layer_idx)
            h = gate.register_forward_hook(rh.hook_fn)
            router_hooks[layer_idx] = rh
            handles.append(h)

        # Register expert hooks
        if isinstance(experts, nn.ModuleList):
            for exp_idx, expert in enumerate(experts):
                hook = SwiGLUHook(layer_idx, exp_idx, prompt_id_ref)
                _wrap_expert_for_hook(expert, hook)
                expert_hooks[(layer_idx, exp_idx)] = hook
        elif isinstance(experts, nn.ModuleDict):
            for k, v in experts.items():
                exp_idx = int(k)
                hook = SwiGLUHook(layer_idx, exp_idx, prompt_id_ref)
                _wrap_expert_for_hook(v, hook)
                expert_hooks[(layer_idx, exp_idx)] = hook
        elif hasattr(experts, "gate_up_proj") or hasattr(experts, "num_experts"):
            # Fused experts module (like Qwen3MoeExperts)
            num_exps = getattr(experts, "num_experts", 128)
            for exp_idx in range(num_exps):
                hook = SwiGLUHook(layer_idx, exp_idx, prompt_id_ref)
                expert_hooks[(layer_idx, exp_idx)] = hook
            _wrap_fused_experts_for_hook(experts, expert_hooks, layer_idx)
        else:
            continue

    n_experts = len(expert_hooks)
    n_layers = len(router_hooks)
    print(f"Registered {n_experts} expert hooks across {n_layers} MoE layers.")
    return expert_hooks, router_hooks, handles


def _wrap_expert_for_hook(expert_module, hook: SwiGLUHook):
    """
    Monkey-patch the expert's forward method to capture intermediate
    activations (SiLU(gate) * up) BEFORE down_proj.
    """
    original_forward = expert_module.forward

    def hooked_forward(*args, **kwargs):
        # Try to capture the intermediate activation
        x = args[0] if len(args) > 0 else kwargs.get("hidden_states", None)
        if x is None:
            return original_forward(*args, **kwargs)

        # Compute gate and up projections
        if hasattr(expert_module, "gate_proj") and hasattr(expert_module, "up_proj"):
            gate = expert_module.gate_proj(x)
            up = expert_module.up_proj(x)

            # Apply activation function
            if hasattr(expert_module, "act_fn"):
                gate = expert_module.act_fn(gate)
            else:
                gate = torch.nn.functional.silu(gate)

            intermediate = gate * up  # This is what we want to capture

            # Fire the hook manually with the intermediate tensor
            hook.hook_fn(expert_module, (x,), intermediate)

            # Complete the forward pass
            if hasattr(expert_module, "down_proj"):
                return expert_module.down_proj(intermediate)
            else:
                return intermediate
        else:
            # Fallback: just run original and hope for the best
            return original_forward(*args, **kwargs)

    expert_module.forward = hooked_forward


def _wrap_fused_experts_for_hook(experts_module, expert_hooks_dict, layer_idx):
    """
    Monkey-patch the forward method of Qwen3MoeExperts to intercept
    activations per expert.
    """
    original_forward = experts_module.forward
    num_experts = experts_module.num_experts

    def hooked_forward(hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx_tensor in expert_hit:
            expert_idx = int(expert_idx_tensor[0].item())
            if expert_idx == num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]

            # Recompute gate & up to capture intermediate activations
            gate, up = nn.functional.linear(current_state, experts_module.gate_up_proj[expert_idx]).chunk(2, dim=-1)

            if hasattr(experts_module, "act_fn"):
                gate = experts_module.act_fn(gate)
            else:
                gate = torch.nn.functional.silu(gate)

            intermediate = gate * up

            # Fire the hook for this expert
            hook = expert_hooks_dict.get((layer_idx, expert_idx))
            if hook is not None:
                hook.hook_fn_fused(token_idx.cpu().numpy(), intermediate)

            current_hidden_states = nn.functional.linear(intermediate, experts_module.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

    experts_module.forward = hooked_forward


# =====================================================================
# 5. DATABASE / OUTPUT
# =====================================================================
def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS activations (
            prompt_id INTEGER,
            layer INTEGER,
            token_pos INTEGER,
            expert_id INTEGER,
            router_prob REAL,
            active_eps_1e5 INTEGER,
            active_eps_1e4 INTEGER,
            active_eps_1e3 INTEGER,
            active_eps_1e2 INTEGER,
            energy_k_50 INTEGER,
            energy_k_70 INTEGER,
            energy_k_80 INTEGER,
            energy_k_90 INTEGER,
            energy_k_95 INTEGER,
            energy_k_99 INTEGER,
            energy_k_999 INTEGER,
            intermediate_dim INTEGER,
            active_indices TEXT,
            latency_ms REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS routing (
            prompt_id INTEGER,
            layer INTEGER,
            token_pos INTEGER,
            expert_id INTEGER,
            router_prob REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    conn.commit()
    return conn


def flush_records(conn: sqlite3.Connection, expert_hooks, router_hooks, prompt_id: int):
    """Write all captured records to the database and clear buffers."""
    cursor = conn.cursor()

    for (layer, exp_id), hook in expert_hooks.items():
        for rec in hook.records:
            cursor.execute(
                "INSERT INTO activations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (rec.prompt_id, rec.layer, rec.token_pos, rec.expert_id,
                 rec.router_prob,
                 rec.active_eps_1e5, rec.active_eps_1e4,
                 rec.active_eps_1e3, rec.active_eps_1e2,
                 rec.energy_k_50, rec.energy_k_70, rec.energy_k_80, rec.energy_k_90,
                 rec.energy_k_95, rec.energy_k_99, rec.energy_k_999,
                 rec.intermediate_dim,
                 json.dumps(rec.active_indices),
                 rec.latency_ms)
            )
        hook.records.clear()

    conn.commit()


# =====================================================================
# 6. POST-HOC ANALYSIS (Routing Entropy, Working-Set Growth, Jaccard)
# =====================================================================
def compute_routing_entropy(conn: sqlite3.Connection):
    """Compute per-layer routing entropy from the activations table."""
    cursor = conn.execute(
        "SELECT layer, expert_id, COUNT(*) as cnt FROM activations GROUP BY layer, expert_id"
    )
    layer_expert_counts = defaultdict(lambda: defaultdict(int))
    for layer, exp_id, cnt in cursor:
        layer_expert_counts[layer][exp_id] = cnt

    print("\n--- Routing Entropy per Layer ---")
    entropies = {}
    for layer in sorted(layer_expert_counts.keys()):
        counts = np.array(list(layer_expert_counts[layer].values()), dtype=float)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        max_entropy = np.log2(len(counts))
        entropies[layer] = (entropy, max_entropy)
        print(f"  Layer {layer:3d}: H = {entropy:.3f} bits (max {max_entropy:.3f}, "
              f"ratio {entropy/max_entropy:.3f})")

    return entropies


def compute_working_set_growth(conn: sqlite3.Connection):
    """
    For each expert, compute W(n) = |union_{i=1}^{n} A_i|.
    This is the cumulative unique neuron set size as tokens accumulate.
    If W(n) saturates, AAEC caching is justified.
    """
    cursor = conn.execute(
        "SELECT layer, expert_id, token_pos, active_indices "
        "FROM activations ORDER BY layer, expert_id, prompt_id, token_pos"
    )

    expert_sequences = defaultdict(list)  # (layer, expert_id) -> list of active_indices
    for layer, exp_id, t_pos, idx_json in cursor:
        indices = json.loads(idx_json)
        expert_sequences[(layer, exp_id)].append(indices)

    # Compute average working-set growth across all experts
    max_tokens = 200
    growth_curves = []

    for (layer, exp_id), sequences in expert_sequences.items():
        if len(sequences) < 10:
            continue
        cumulative_set = set()
        curve = []
        for i, indices in enumerate(sequences[:max_tokens]):
            cumulative_set.update(indices)
            curve.append(len(cumulative_set))
        growth_curves.append(curve)

    # Average across experts, padding shorter curves
    if growth_curves:
        max_len = max(len(c) for c in growth_curves)
        padded = []
        for c in growth_curves:
            padded.append(c + [c[-1]] * (max_len - len(c)))
        avg_curve = np.mean(padded, axis=0)

        print("\n--- Working-Set Growth W(n) ---")
        checkpoints = [1, 5, 10, 25, 50, 100, 150, 200]
        for n in checkpoints:
            if n <= len(avg_curve):
                print(f"  After {n:3d} tokens: W(n) = {avg_curve[n-1]:.1f} unique neurons")

    return growth_curves


def compute_jaccard_decay(conn: sqlite3.Connection):
    """Compute Jaccard similarity at powers-of-two distances."""
    # 1. Get 200 random active experts
    cursor = conn.execute("SELECT DISTINCT layer, expert_id FROM activations")
    all_experts = cursor.fetchall()
    
    import random
    random.seed(42)
    if len(all_experts) > 200:
        sampled_experts = random.sample(all_experts, 200)
    else:
        sampled_experts = all_experts

    # 2. Query activations only for the sampled experts
    expert_sequences = defaultdict(list)
    for layer, exp_id in sampled_experts:
        c = conn.execute(
            "SELECT active_indices FROM activations "
            "WHERE layer = ? AND expert_id = ? "
            "ORDER BY prompt_id, token_pos",
            (layer, exp_id)
        )
        for (idx_json,) in c.fetchall():
            expert_sequences[(layer, exp_id)].append(set(json.loads(idx_json)))

    distances = [1, 2, 4, 8, 16, 32, 64]
    jaccards = {d: [] for d in distances}

    for (layer, exp_id) in sampled_experts:
        sets = expert_sequences[(layer, exp_id)]
        n = len(sets)
        for dist in distances:
            for i in range(min(n - dist, 100)):  # cap to 100 per expert for performance
                union = len(sets[i].union(sets[i + dist]))
                inter = len(sets[i].intersection(sets[i + dist]))
                if union > 0:
                    jaccards[dist].append(inter / union)

    print("\n--- Jaccard Similarity vs Distance ---")
    means = {}
    for d in distances:
        if jaccards[d]:
            m = np.mean(jaccards[d])
            means[d] = m
            print(f"  Distance {d:3d}: J = {m:.4f} (n={len(jaccards[d])})")

    # Fit exponential decay: J(d) = (J0 - J_inf) * exp(-d/tau) + J_inf
    if len(means) >= 3:
        j_inf = min(means.values())
        x_fit, y_fit = [], []
        for d, j in means.items():
            if j > j_inf + 0.001:
                x_fit.append(d)
                y_fit.append(np.log(j - j_inf))
        if len(x_fit) >= 2:
            slope, _ = np.polyfit(x_fit, y_fit, 1)
            tau = -1.0 / slope if slope != 0 else float('inf')
            print(f"\n  Fitted decay time constant tau = {tau:.2f} tokens")
            print(f"  -> AAEC cache eviction window: ~{int(tau * 2)} tokens")

    return means


# =====================================================================
# 7. MOCK MODE (CPU testing without real model)
# =====================================================================
def run_mock():
    """
    Run a complete instrumentation pass using a tiny mock MoE model on CPU.
    Validates the entire pipeline: hooks, DB writes, and post-hoc analysis.
    """
    print("=" * 70)
    print("MOCK MODE: Testing instrumentation pipeline on CPU")
    print("=" * 70)

    db_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "mock_activations.db"
    )
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = init_db(db_path)

    # Create a tiny mock MoE model
    class MockExpert(nn.Module):
        def __init__(self, hidden, inter):
            super().__init__()
            self.gate_proj = nn.Linear(hidden, inter)
            self.up_proj = nn.Linear(hidden, inter)
            self.down_proj = nn.Linear(inter, hidden)

        def forward(self, x):
            gate = torch.nn.functional.silu(self.gate_proj(x))
            up = self.up_proj(x)
            intermediate = gate * up
            return self.down_proj(intermediate)

    class MockMoELayer(nn.Module):
        def __init__(self, hidden, inter, n_experts, topk):
            super().__init__()
            self.experts = nn.ModuleList([MockExpert(hidden, inter) for _ in range(n_experts)])
            self.gate = nn.Linear(hidden, n_experts)
            self.topk = topk

        def forward(self, x):
            # Route
            logits = self.gate(x)
            probs = torch.softmax(logits, dim=-1)
            _, indices = probs.topk(self.topk, dim=-1)

            # Dispatch (simplified: sum expert outputs)
            out = torch.zeros_like(x)
            for t in range(x.size(1)):
                for k in range(self.topk):
                    exp_idx = indices[0, t, k].item()
                    out[0, t] += self.experts[exp_idx](x[0, t].unsqueeze(0).unsqueeze(0)).squeeze()
            return out

    hidden_dim = 256
    inter_dim = 512
    n_experts = 16
    topk = 4
    seq_len = 64
    n_prompts = 5

    moe_layer = MockMoELayer(hidden_dim, inter_dim, n_experts, topk)

    # Register hooks
    prompt_id_ref = [0]
    expert_hooks = {}
    handles = []

    for exp_idx, expert in enumerate(moe_layer.experts):
        hook = SwiGLUHook(layer_idx=0, expert_idx=exp_idx, prompt_id_ref=prompt_id_ref)
        _wrap_expert_for_hook(expert, hook)
        expert_hooks[(0, exp_idx)] = hook

    router_hook = RouterHook(layer_idx=0)
    h = moe_layer.gate.register_forward_hook(router_hook.hook_fn)
    handles.append(h)

    # Run inference on mock prompts
    for p_id in range(n_prompts):
        prompt_id_ref[0] = p_id
        # Generate correlated hidden states (simulates sequential text)
        x = torch.randn(1, seq_len, hidden_dim)
        for i in range(1, seq_len):
            x[0, i] = 0.92 * x[0, i-1] + 0.08 * x[0, i]

        with torch.no_grad():
            _ = moe_layer(x)

        flush_records(conn, expert_hooks, {}, p_id)
        total_recs = conn.execute("SELECT COUNT(*) FROM activations").fetchone()[0]
        print(f"  Prompt {p_id+1}/{n_prompts} done ({total_recs} total records)")

        # Clear any remaining
        for h_obj in expert_hooks.values():
            h_obj.records.clear()

    # Remove handles
    for h in handles:
        h.remove()

    # Post-hoc analysis
    print("\n" + "=" * 70)
    print("POST-HOC ANALYSIS")
    print("=" * 70)

    compute_routing_entropy(conn)
    compute_working_set_growth(conn)
    compute_jaccard_decay(conn)

    # Summary stats
    row = conn.execute("SELECT COUNT(*) FROM activations").fetchone()
    print(f"\nTotal activation records: {row[0]}")

    row = conn.execute(
        "SELECT AVG(energy_k_99), AVG(energy_k_95), AVG(intermediate_dim) FROM activations"
    ).fetchone()
    if row[0]:
        print(f"Mean energy_k_99: {row[0]:.1f} / {row[2]:.0f} "
              f"({row[0]/row[2]*100:.1f}% of intermediate dim)")
        print(f"Mean energy_k_95: {row[1]:.1f} / {row[2]:.0f} "
              f"({row[1]/row[2]*100:.1f}% of intermediate dim)")

    conn.close()
    print(f"\nResults saved to: {db_path}")
    print("=" * 70)


# =====================================================================
# 8. MAIN ENTRY POINT
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="AAEC Real-World MoE Instrumentation")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-30B-A3B",
                        help="HuggingFace model ID")
    parser.add_argument("--quantize", type=str, choices=["nf4", "none"], default="none",
                        help="Quantization mode")
    parser.add_argument("--num-prompts", type=int, default=50,
                        help="Number of prompts to profile")
    parser.add_argument("--output", type=str, default="aaec_activations.db",
                        help="Output SQLite database path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only load model and register hooks, no inference")
    parser.add_argument("--mock", action="store_true",
                        help="Run CPU mock mode (no real model needed)")
    args = parser.parse_args()

    if args.mock:
        run_mock()
        return

    # Load model
    quant = args.quantize if args.quantize != "none" else None
    model, tokenizer = load_model(args.model, quantize=quant)

    # Register hooks
    prompt_id_ref = [0]
    expert_hooks, router_hooks, handles = register_hooks(model, prompt_id_ref)

    if args.dry_run:
        print(f"\nDry run complete. {len(expert_hooks)} expert hooks registered.")
        for h in handles:
            h.remove()
        return

    # Init database
    db_path = args.output
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = init_db(db_path)

    # Store metadata
    conn.execute("INSERT INTO metadata VALUES (?, ?)", ("model", args.model))
    conn.execute("INSERT INTO metadata VALUES (?, ?)", ("quantize", args.quantize))
    conn.execute("INSERT INTO metadata VALUES (?, ?)", ("num_prompts", str(args.num_prompts)))
    conn.commit()

    # Run inference
    device = next(model.parameters()).device
    prompts = PROMPT_DATASET[:args.num_prompts]

    print(f"\nProfiling {len(prompts)} prompts...")
    for p_id, prompt in enumerate(prompts):
        prompt_id_ref[0] = p_id

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            _ = model(**inputs)

        flush_records(conn, expert_hooks, router_hooks, p_id)

        n_recs = conn.execute("SELECT COUNT(*) FROM activations WHERE prompt_id=?", (p_id,)).fetchone()[0]
        print(f"  [{p_id+1}/{len(prompts)}] {n_recs} records | {prompt[:60]}...")

    # Cleanup hooks
    for h in handles:
        h.remove()

    # Post-hoc analysis
    print("\n" + "=" * 70)
    print("POST-HOC ANALYSIS")
    print("=" * 70)

    compute_routing_entropy(conn)
    compute_working_set_growth(conn)
    compute_jaccard_decay(conn)

    row = conn.execute("SELECT COUNT(*) FROM activations").fetchone()
    print(f"\nTotal activation records: {row[0]}")
    conn.close()
    print(f"Results saved to: {db_path}")


if __name__ == "__main__":
    main()
