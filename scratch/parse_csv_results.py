import os
import json
import pandas as pd
import numpy as np
import ast

def parse_csv(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} does not exist.")
        return None
        
    df = pd.read_csv(csv_path)
    
    # Calculate total latency in seconds
    max_end_time = df['end_time'].max()
    total_latency_s = max_end_time / 1e9
    
    # Calculate input and output tokens
    total_prompt = df['input'].sum()
    total_gen = df['output'].sum()
    
    # Throughputs
    prompt_thru_tok_s = total_prompt / total_latency_s if total_latency_s > 0 else 0.0
    gen_thru_tok_s = total_gen / total_latency_s if total_latency_s > 0 else 0.0
    token_thru_tok_s = (total_prompt + total_gen) / total_latency_s if total_latency_s > 0 else 0.0
    
    # Latencies (convert from ns to ms)
    avg_ttft_ms = (df['TTFT'].mean()) / 1e6
    median_ttft_ms = (df['TTFT'].median()) / 1e6
    
    avg_tpot_ms = (df['TPOT'].mean()) / 1e6
    median_tpot_ms = (df['TPOT'].median()) / 1e6
    
    # Flatten all ITL lists to get overall ITL mean
    all_itls = []
    for itl_str in df['ITL']:
        try:
            itl_list = ast.literal_eval(itl_str)
            if isinstance(itl_list, list):
                all_itls.extend(itl_list)
        except Exception as e:
            print(f"Error parsing ITL list: {e}")
            
    avg_itl_ms = (np.mean(all_itls) / 1e6) if all_itls else 0.0
    
    return {
        "total_latency_s": round(total_latency_s, 3),
        "prompt_thru_tok_s": round(prompt_thru_tok_s, 2),
        "gen_thru_tok_s": round(gen_thru_tok_s, 2),
        "token_thru_tok_s": round(token_thru_tok_s, 2),
        "avg_ttft_ms": round(avg_ttft_ms, 2),
        "median_ttft_ms": round(median_ttft_ms, 2),
        "avg_tpot_ms": round(avg_tpot_ms, 2),
        "median_tpot_ms": round(median_tpot_ms, 2),
        "avg_itl_ms": round(avg_itl_ms, 2)
    }

def main():
    models = ["qwen3", "deepseek", "llama4"]
    policies = ["fifo", "sab", "sab_aae", "sab_cooldown", "sab_thresh"]
    
    results = {m: {} for m in models}
    
    for model in models:
        for pol in policies:
            csv_path = f"outputs/phase1/{model}_{pol}.csv"
            print(f"Parsing {csv_path}...")
            metrics = parse_csv(csv_path)
            if metrics:
                results[model][pol] = metrics
            else:
                results[model][pol] = {
                    "total_latency_s": 0.0,
                    "prompt_thru_tok_s": 0.0,
                    "gen_thru_tok_s": 0.0,
                    "token_thru_tok_s": 0.0,
                    "avg_ttft_ms": 0.0,
                    "median_ttft_ms": 0.0,
                    "avg_tpot_ms": 0.0,
                    "median_tpot_ms": 0.0,
                    "avg_itl_ms": 0.0
                }
                
    with open("outputs/phase1/summary.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("Regenerated outputs/phase1/summary.json successfully.")

if __name__ == "__main__":
    main()
