import sys
import os
from collections import defaultdict

# Path to the chakra library within astra-sim
chakra_lib = "/home/palakm/MoEServingSim/astra-sim/extern/graph_frontend/chakra/build/lib"
if chakra_lib not in sys.path:
    sys.path.insert(0, chakra_lib)

from chakra.schema.protobuf.et_def_pb2 import Node as ChakraNode, GlobalMetadata
from chakra.src.third_party.utils.protolib import decodeMessage as decode_message
from chakra.src.third_party.utils.protolib import openFileRd as open_file_rd

def check_cycles(et_path):
    print(f"Reading ET trace: {et_path}")
    if not os.path.exists(et_path):
        print("File does not exist.")
        return

    execution_trace = open_file_rd(et_path)
    global_metadata = GlobalMetadata()
    decode_message(execution_trace, global_metadata)

    nodes = {}
    adj = defaultdict(list)
    in_degree = defaultdict(int)

    node = ChakraNode()
    while decode_message(execution_trace, node):
        node_id = node.id
        nodes[node_id] = {
            "name": node.name,
            "type": node.type,
        }
        all_parents = list(node.data_deps) + list(node.ctrl_deps)
        for parent_id in all_parents:
            adj[parent_id].append(node_id)
            in_degree[node_id] += 1
        # Clear node for the next decode
        node = ChakraNode()
    execution_trace.close()

    print(f"Loaded {len(nodes)} nodes.")

    # Detect cycle using DFS
    visited = {} # id -> state (0 = unvisited, 1 = visiting, 2 = visited)
    cycle = []
    
    def dfs(u):
        visited[u] = 1 # visiting
        for v in adj[u]:
            if visited.get(v, 0) == 1:
                # Cycle detected!
                cycle.append((u, v))
                return True
            elif visited.get(v, 0) == 0:
                if dfs(v):
                    cycle.append((u, v))
                    return True
        visited[u] = 2 # visited
        return False

    has_cycle = False
    for u in nodes:
        if visited.get(u, 0) == 0:
            if dfs(u):
                has_cycle = True
                break

    if has_cycle:
        print("CYCLE DETECTED!")
        print("Path of cycle (reversed):", cycle)
        for u, v in reversed(cycle):
            print(f"  Node {u} ({nodes[u]['name']}) -> Node {v} ({nodes[v]['name']})")
    else:
        print("No cycle detected. Graph is a valid DAG.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        et_file = "/home/palakm/MoEServingSim/astra-sim/inputs/workload/H100/Qwen/Qwen3-235B-A22B/dp_A_batch0/llm.0.et"
    else:
        et_file = sys.argv[1]
    check_cycles(et_file)
