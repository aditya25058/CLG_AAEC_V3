# evaluation/scripts/e03_energy_concentration.py
import os
import json
import sqlite3
import numpy as np

MODELS = {
    "qwen3_30b": {
        "db_path": "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/qwen3_30b_real_v2.db",
        "intermediate_dim": 768
    },
    "deepseek_v2_lite": {
        "db_path": "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/deepseek_lite_real.db",
        "intermediate_dim": 1408
    }
}

def analyze_energy_concentration(model_name: str, spec: dict):
    db_path = spec["db_path"]
    int_dim = spec["intermediate_dim"]
    
    if not os.path.exists(db_path):
        print(f"Skipping {model_name} (database not found)")
        return
        
    print(f"Analyzing energy concentration for {model_name}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT energy_k_50, energy_k_70, energy_k_80, energy_k_90, energy_k_95, energy_k_99
        FROM activations
    """)
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        print("  No activation records found.")
        return
        
    data = np.array(rows)
    thresholds = [50, 70, 80, 90, 95, 99]
    results = {}
    
    print(f"  {'Threshold (%)':<15} | {'Mean Columns':<15} | {'Std Dev':<10} | {'Fraction of FFN (%)':<20}")
    print("  " + "-"*68)
    
    for idx, t in enumerate(thresholds):
        col_data = data[:, idx]
        mean_cols = float(np.mean(col_data))
        std_cols = float(np.std(col_data))
        fraction = (mean_cols / int_dim) * 100.0
        
        results[f"k_{t}"] = {
            "mean_columns": mean_cols,
            "std_columns": std_cols,
            "fraction_pct": fraction
        }
        print(f"  {t:<15d} | {mean_cols:<15.2f} | {std_cols:<10.2f} | {fraction:<19.2f}%")
        
    out_dir = f"/home/palakm/MoEServingSim/evaluation/results/e03_energy/{model_name}"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "energy_concentration.json"), "w") as f:
        json.dump(results, f, indent=4)

def main():
    for name, spec in MODELS.items():
        analyze_energy_concentration(name, spec)

if __name__ == "__main__":
    main()
