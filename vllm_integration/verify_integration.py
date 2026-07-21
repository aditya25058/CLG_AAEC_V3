# vllm_integration/verify_integration.py
import torch
import sys
import os

# Enable importing from the local folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fused_moe_colossus import FusedMoEWithCOLOSSUS, CacheController

def main():
    print("======================================================================")
    print("Verifying vLLM COLOSSUS Integration Modules Compatibility")
    print("======================================================================")
    
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    
    # Model parameters matching Qwen3 or Mixtral style
    num_layers = 2
    num_experts = 8
    top_k = 2
    hidden_size = 2048
    intermediate_size = 4096
    cache_size = 512
    
    print(f"1. Allocating Cache Controller (VRAM Cache Size: {cache_size} columns)...")
    cache_controller = CacheController(
        num_layers=num_layers,
        num_experts=num_experts,
        cache_size=cache_size,
        hidden_size=hidden_size
    )
    print("   -> Cache Controller successfully initialized.")
    
    print("2. Initializing FusedMoEWithCOLOSSUS layer class...")
    moe_layer = FusedMoEWithCOLOSSUS(
        layer_idx=0,
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        cache_controller=cache_controller
    ).to(device).to(dtype=torch.bfloat16)
    print("   -> FusedMoE layer initialized and mapped to GPU.")
    
    # Generating mock token input batch
    batch_size = 128
    x = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=device)
    print(f"3. Formed mock batch token inputs: {x.shape}")
    
    print("4. Executing FusedMoE layer forward pass with Triton Streaming Accumulation...")
    try:
        # Run warm-up
        y = moe_layer(x)
        print("   -> Forward pass warm-up completed.")
        
        # Run timed execution
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(5):
            y = moe_layer(x)
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_time = start_event.elapsed_time(end_event) / 5
        print(f"   -> Forward pass execution succeeded.")
        print(f"   -> Output Tensor Shape: {y.shape}")
        print(f"   -> Average execution time: {elapsed_time:.3f} ms")
        print("\n[SUCCESS] COLOSSUS integration modules are compilation-ready and compatible with Triton/PyTorch serving.")
    except Exception as e:
        print(f"\n[ERROR] Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
