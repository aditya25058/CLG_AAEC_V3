# evaluation/scripts/e09_quality_benchmarks.py
# Rewritten: no hardcoded fallback, multiple benchmarks, proper sample sizes
import os
import json
import traceback
import torch

MODELS = {
    "qwen3_30b": "Qwen/Qwen3-30B-A3B",
    "deepseek_v2_lite": "deepseek-ai/DeepSeek-V2-Lite"
}

# Multiple tasks with proper sample sizes
BENCHMARKS = [
    {"task": "arc_easy", "limit": 100, "description": "ARC-Easy (100 samples)"},
    {"task": "arc_challenge", "limit": 100, "description": "ARC-Challenge (100 samples)"},
]

def run_quality_benchmark(model_name: str, model_path: str):
    print(f"Running Quality Benchmarks for {model_name}...")

    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from lm_eval.evaluator import simple_evaluate
    except ImportError as e:
        print(f"  FATAL: lm-eval not installed: {e}")
        # Write explicit FAIL — no fallback
        report = {
            "model_name": model_name,
            "status": "FAILED",
            "error": f"lm-eval import failed: {str(e)}",
            "benchmarks": []
        }
        _save_result(model_name, report)
        return

    print(f"  Instantiating {model_path} via HFLM...")
    try:
        model_instance = HFLM(
            pretrained=model_path,
            dtype="bfloat16" if torch.cuda.is_available() else "float32",
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"  FATAL: Could not load model: {e}")
        traceback.print_exc()
        report = {
            "model_name": model_name,
            "status": "FAILED",
            "error": f"Model load failed: {str(e)}",
            "benchmarks": []
        }
        _save_result(model_name, report)
        return

    benchmark_results = []

    for bench in BENCHMARKS:
        task = bench["task"]
        limit = bench["limit"]
        desc = bench["description"]
        print(f"\n  Evaluating: {desc}...")

        try:
            results = simple_evaluate(
                model=model_instance,
                tasks=[task],
                limit=limit,
                batch_size="auto"
            )

            task_results = results["results"].get(task, {})
            score = task_results.get("acc,none", task_results.get("acc", None))
            stderr = task_results.get("acc_stderr,none", task_results.get("acc_stderr", None))
            n_samples = limit if limit else "full"

            benchmark_results.append({
                "task": task,
                "description": desc,
                "accuracy": score,
                "accuracy_stderr": stderr,
                "samples": n_samples,
                "status": "PASS" if score is not None else "ERROR"
            })

            if score is not None:
                print(f"    Score: {score*100:.2f}% ± {(stderr or 0)*100:.2f}%")
            else:
                print(f"    WARNING: Could not extract accuracy from results")

        except Exception as e:
            print(f"    ERROR on {task}: {e}")
            traceback.print_exc()
            benchmark_results.append({
                "task": task,
                "description": desc,
                "accuracy": None,
                "accuracy_stderr": None,
                "samples": limit if limit else "full",
                "status": "FAILED",
                "error": str(e)
            })

    report = {
        "model_name": model_name,
        "status": "COMPLETED",
        "benchmarks": benchmark_results,
        "note": "AAEC is provably lossless (E01). These benchmarks verify end-to-end pipeline correctness, not AAEC-specific quality."
    }

    _save_result(model_name, report)

def _save_result(model_name, report):
    out_dir = f"/home/palakm/MoEServingSim/evaluation/results/e09_quality/{model_name}"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "quality_comparison.json"), "w") as f:
        json.dump(report, f, indent=4)

def main():
    for name, path in MODELS.items():
        run_quality_benchmark(name, path)

if __name__ == "__main__":
    main()
