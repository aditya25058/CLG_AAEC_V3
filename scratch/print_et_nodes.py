import sys
import os

# Path to the chakra library within astra-sim
chakra_lib = "/home/palakm/MoEServingSim/astra-sim/extern/graph_frontend/chakra/build/lib"
if chakra_lib not in sys.path:
    sys.path.insert(0, chakra_lib)

from chakra.schema.protobuf.et_def_pb2 import Node as ChakraNode, GlobalMetadata
from chakra.src.third_party.utils.protolib import decodeMessage as decode_message
from chakra.src.third_party.utils.protolib import openFileRd as open_file_rd

def print_et(et_path, count=50):
    execution_trace = open_file_rd(et_path)
    global_metadata = GlobalMetadata()
    decode_message(execution_trace, global_metadata)

    node = ChakraNode()
    i = 0
    while decode_message(execution_trace, node) and i < count:
        print(f"Node {node.id}: name={node.name}, type={node.type}")
        print(f"  data_deps={list(node.data_deps)}, ctrl_deps={list(node.ctrl_deps)}")
        for attr in node.attr:
            if attr.name == "comm_type":
                print(f"  comm_type={attr.int64_val}")
            elif attr.name == "comm_size":
                print(f"  comm_size={attr.int64_val}")
            elif attr.name == "involved_dim":
                print(f"  involved_dim={list(attr.bool_list.values)}")
        i += 1
    execution_trace.close()

if __name__ == "__main__":
    print_et("/home/palakm/MoEServingSim/astra-sim/inputs/workload/H100/Qwen/Qwen3-235B-A22B/dp_A_batch0/llm.0.et")
