import sys
import os

chakra_lib = "/home/palakm/MoEServingSim/astra-sim/extern/graph_frontend/chakra/build/lib"
if chakra_lib not in sys.path:
    sys.path.insert(0, chakra_lib)

from chakra.schema.protobuf.et_def_pb2 import Node as ChakraNode, GlobalMetadata
from chakra.src.third_party.utils.protolib import decodeMessage as decode_message
from chakra.src.third_party.utils.protolib import openFileRd as open_file_rd

def debug_et(path):
    f = open(path, "rb")
    print(f"File size: {os.path.getsize(path)} bytes")
    
    # Read global metadata first
    execution_trace = open_file_rd(path)
    global_metadata = GlobalMetadata()
    decode_message(execution_trace, global_metadata)
    
    i = 0
    while True:
        node = ChakraNode()
        # Get position before decoding
        pos = f.tell() # Wait, execution_trace is a different file object, but let's see.
        success = decode_message(execution_trace, node)
        if not success:
            print(f"decode_message returned False at index {i}")
            break
        print(f"Node {i}: id={node.id}, name={node.name}, deps={list(node.data_deps)}")
        i += 1
    execution_trace.close()

if __name__ == "__main__":
    debug_et("/home/palakm/MoEServingSim/astra-sim/inputs/workload/H100/Qwen/Qwen3-235B-A22B/dp_A_batch0/llm.0.et")
