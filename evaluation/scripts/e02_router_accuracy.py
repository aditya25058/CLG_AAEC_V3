# evaluation/scripts/e02_router_accuracy.py
# Rewritten: adds per-layer breakdown, random baseline, and cross-prompt holdout
import os
import json
import sqlite3
import numpy as np

MODELS = {
    "qwen3_30b": {
        "db_path": "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/qwen3_30b_real_v2.db",
        "num_layers": 48,
        "num_experts": 128,
        "top_k": 8
    },
    "deepseek_v2_lite": {
        "db_path": "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/deepseek_lite_real.db",
        "num_layers": 26,
        "num_experts": 64,
        "top_k": 6
    }
}

def evaluate_router_accuracy(model_name: str, spec: dict):
    db_path = spec["db_path"]
    NL = spec["num_layers"]
    NE = spec["num_experts"]
    top_k = spec["top_k"]

    if not os.path.exists(db_path):
        print(f"Skipping {model_name} (database not found)")
        return

    print(f"Evaluating router prediction accuracy for {model_name}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT prompt_id, token_pos, layer, expert_id, router_prob 
        FROM activations 
        ORDER BY prompt_id, token_pos, layer, router_prob DESC
    """)
    rows = cursor.fetchall()
    conn.close()

    # Group active experts per token
    activations_map = {}
    for p_id, t_pos, layer, exp_id, prob in rows:
        key = (p_id, t_pos, layer)
        if key not in activations_map:
            activations_map[key] = []
        activations_map[key].append(exp_id)

    prompt_ids = sorted(list(set(row[0] for row in rows)))
    split_idx = len(prompt_ids) // 2
    calib_prompts = set(prompt_ids[:split_idx])
    eval_prompts = set(prompt_ids[split_idx:])

    # 1. Train Transition Predictor on Calibration Set
    transition_counts = np.zeros((NL, NE, NE))
    layer_expert_counts = np.zeros((NL, NE))

    for key, experts in activations_map.items():
        p_id, t_pos, layer = key
        if p_id not in calib_prompts:
            continue

        for exp_id in experts:
            if exp_id < NE:
                layer_expert_counts[layer, exp_id] += 1

        if layer > 0:
            prev_key = (p_id, t_pos, layer - 1)
            if prev_key in activations_map:
                prev_top1 = activations_map[prev_key][0]
                if prev_top1 < NE:
                    for exp_id in experts:
                        if exp_id < NE:
                            transition_counts[layer, prev_top1, exp_id] += 1

    # Normalize probabilities
    transition_probs = np.zeros_like(transition_counts)
    for l in range(NL):
        for e in range(NE):
            row_sum = transition_counts[l, e].sum()
            if row_sum > 0:
                transition_probs[l, e] = transition_counts[l, e] / row_sum
            else:
                layer_sum = layer_expert_counts[l].sum()
                if layer_sum > 0:
                    transition_probs[l, e] = layer_expert_counts[l] / layer_sum
                else:
                    transition_probs[l, e] = 1.0 / NE

    layer_0_most_frequent = int(np.argmax(layer_expert_counts[0]))

    # 2. Evaluate Predictor on Test Set — with PER-LAYER tracking
    per_layer_stats = {l: {"total": 0, "top1_correct": 0, "top3_correct": 0, "topk_correct": 0} for l in range(NL)}
    total_predictions = 0
    top1_correct = 0
    top3_correct = 0
    topk_correct = 0

    for key, experts in activations_map.items():
        p_id, t_pos, layer = key
        if p_id not in eval_prompts:
            continue

        ground_truth = set(experts)
        if not ground_truth:
            continue

        # Predict
        if layer == 0:
            pred_exp = layer_0_most_frequent
            sorted_predictions = np.argsort(layer_expert_counts[layer])[::-1]
        else:
            prev_key = (p_id, t_pos, layer - 1)
            if prev_key in activations_map:
                prev_top1 = activations_map[prev_key][0]
                if prev_top1 < NE:
                    probs = transition_probs[layer, prev_top1]
                    pred_exp = int(np.argmax(probs))
                    sorted_predictions = np.argsort(probs)[::-1]
                else:
                    pred_exp = 0
                    sorted_predictions = np.argsort(layer_expert_counts[layer])[::-1]
            else:
                pred_exp = 0
                sorted_predictions = np.argsort(layer_expert_counts[layer])[::-1]

        pred_top1 = pred_exp
        pred_top3 = set(sorted_predictions[:3].tolist())
        pred_topk = set(sorted_predictions[:top_k].tolist())

        total_predictions += 1
        per_layer_stats[layer]["total"] += 1

        if pred_top1 in ground_truth:
            top1_correct += 1
            per_layer_stats[layer]["top1_correct"] += 1
        if not ground_truth.isdisjoint(pred_top3):
            top3_correct += 1
            per_layer_stats[layer]["top3_correct"] += 1
        if not ground_truth.isdisjoint(pred_topk):
            topk_correct += 1
            per_layer_stats[layer]["topk_correct"] += 1

    top1_acc = top1_correct / total_predictions if total_predictions > 0 else 0.0
    top3_acc = top3_correct / total_predictions if total_predictions > 0 else 0.0
    topk_acc = topk_correct / total_predictions if total_predictions > 0 else 0.0

    # Random baseline
    random_top1 = 1.0 / NE
    random_topk = min(1.0, top_k / NE)

    # Per-layer breakdown
    per_layer_breakdown = {}
    for l in range(NL):
        s = per_layer_stats[l]
        if s["total"] > 0:
            per_layer_breakdown[str(l)] = {
                "total": s["total"],
                "top1_accuracy": s["top1_correct"] / s["total"],
                "top3_accuracy": s["top3_correct"] / s["total"],
                f"top{top_k}_accuracy": s["topk_correct"] / s["total"]
            }

    result = {
        "model_name": model_name,
        "total_predictions": total_predictions,
        "top1_accuracy": top1_acc,
        "top3_accuracy": top3_acc,
        f"top{top_k}_accuracy": topk_acc,
        "random_baseline_top1": random_top1,
        f"random_baseline_top{top_k}": random_topk,
        "improvement_over_random_top1": top1_acc / random_top1 if random_top1 > 0 else 0.0,
        "calibration_prompts": len(calib_prompts),
        "evaluation_prompts": len(eval_prompts),
        "per_layer_breakdown": per_layer_breakdown
    }

    print(f"  Top-1: {top1_acc*100:.2f}% | Top-3: {top3_acc*100:.2f}% | Top-{top_k}: {topk_acc*100:.2f}%")
    print(f"  Random baseline Top-1: {random_top1*100:.2f}% | Improvement: {top1_acc/random_top1:.1f}x")
    print(f"  Calib prompts: {len(calib_prompts)} | Eval prompts: {len(eval_prompts)}")

    # Print per-layer summary (every 8th layer)
    print(f"\n  Per-Layer Top-1 Accuracy (sampled):")
    for l in range(0, NL, max(1, NL // 8)):
        if str(l) in per_layer_breakdown:
            acc = per_layer_breakdown[str(l)]["top1_accuracy"]
            print(f"    Layer {l:3d}: {acc*100:.1f}% ({per_layer_breakdown[str(l)]['total']} samples)")

    out_dir = f"/home/palakm/MoEServingSim/evaluation/results/e02_router/{model_name}"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "router_accuracy.json"), "w") as f:
        json.dump(result, f, indent=4)

def main():
    for name, spec in MODELS.items():
        evaluate_router_accuracy(name, spec)

if __name__ == "__main__":
    main()
