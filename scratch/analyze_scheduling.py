import pandas as pd
import sys

def analyze_file(csv_path):
    print(f"\nAnalysis for: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df.sort_values(by="TTFT")
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df[['request id', 'input', 'output', 'arrival', 'queuing_delay', 'TTFT', 'latency']])

def main():
    if len(sys.argv) > 1:
        analyze_file(sys.argv[1])
    else:
        # Default analysis
        analyze_file("outputs/phase1/qwen3_sab.csv")
        analyze_file("outputs/phase1/qwen3_sab_thresh.csv")
        analyze_file("outputs/phase1/llama4_sab.csv")
        analyze_file("outputs/phase1/llama4_sab_thresh.csv")

if __name__ == "__main__":
    main()
