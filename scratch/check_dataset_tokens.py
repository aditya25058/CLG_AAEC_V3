import json
import sys

def print_stats(filepath):
    print(f"Stats for {filepath}:")
    with open(filepath, "r") as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            req_id = data["request_id"]
            arrival = data["arrival_time_ns"] / 1e9  # ns to sec
            input_toks = data["input_toks"]
            output_toks = data["output_toks"]
            source = data["source_trace"]
            print(f"  Req {req_id}: arrival={arrival:.1f}s, input={input_toks}, output={output_toks}, source={source}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print_stats(sys.argv[1])
    else:
        print_stats("datasets/qwen3_livecodebench_10req.jsonl")
