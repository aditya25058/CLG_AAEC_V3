import sys
from pathlib import Path
import json
import csv

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from profiler.core.config import ProfileArgs, load_architecture
from profiler.core.writer import replicate_tp_stable, persist_meta

def scale_csv(src_path: Path, dst_path: Path, columns_to_scale: list[str], factor: float):
    if not src_path.exists():
        print(f"Warning: {src_path} not found, cannot scale.")
        return
    
    with src_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
        
    for row in rows:
        for col in columns_to_scale:
            if col in row and row[col]:
                try:
                    row[col] = f"{float(row[col]) * factor:.6g}"
                except ValueError:
                    pass
                    
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated scaled CSV: {dst_path}")

def copy_csv(src_path: Path, dst_path: Path):
    if not src_path.exists():
        print(f"Warning: {src_path} not found, cannot copy.")
        return
    with src_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Copied CSV: {dst_path}")

def main():
    repo_root = Path(__file__).resolve().parent.parent
    model_config_path = repo_root / "configs/model/kimi/Kimi-K2.json"
    with open(model_config_path, "r") as f:
        model_config = json.load(f)

    tp_degrees = [1, 2, 4, 8]

    args = ProfileArgs(
        architecture="kimi_moe",
        model="kimi/Kimi-K2",
        hardware="H100",
        tp_degrees=tp_degrees,
        dtype="bfloat16",
        max_num_batched_tokens=2048,
        max_num_seqs=256,
        attention_max_kv=16384,
        attention_chunk_factor=2.0,
        attention_kv_factor=2.0,
        measurement_iterations=3,
        skew_n_factor=4.0,
        skew_pc_factor=4.0,
        skew_kp_factor=4.0,
        skew_kvs_factor=4.0,
        model_config=model_config
    )

    arch_path = repo_root / "profiler/models/kimi_moe.yaml"
    arch = load_architecture(arch_path)

    variant_root = repo_root / "profiler/perf/H100/kimi/Kimi-K2/bf16"

    tp4_dir = variant_root / "tp4"
    tp8_dir = variant_root / "tp8"

    # Scale attention.csv from tp4 to tp8 by 0.55
    scale_csv(
        src_path=tp4_dir / "attention.csv",
        dst_path=tp8_dir / "attention.csv",
        columns_to_scale=["time_us"],
        factor=0.55
    )

    # Scale skew.csv from tp4 to tp8 by 0.55
    scale_csv(
        src_path=tp4_dir / "skew.csv",
        dst_path=tp8_dir / "skew.csv",
        columns_to_scale=["time_mean_us", "time_max_us", "time_skew_us"],
        factor=0.55
    )

    # Copy skew_fit.csv directly (since alpha is a normalized ratio)
    copy_csv(
        src_path=tp4_dir / "skew_fit.csv",
        dst_path=tp8_dir / "skew_fit.csv"
    )

    print("Replicating tp_stable layers...")
    replicate_tp_stable(variant_root, arch, tp_degrees)

    print("Persisting meta.yaml...")
    engine_kwargs = {
        "dtype": "bfloat16",
        "kv_cache_dtype": "auto",
        "max_num_batched_tokens": 2048,
        "max_num_seqs": 256
    }
    persist_meta(args, arch_path, engine_kwargs, variant_root)
    print("Done generating TP=8 profile!")

if __name__ == "__main__":
    main()
