import torch
import json
import numpy as np
from transformers import AutoModelForCausalLM

def main():
    print("==================================================================")
    print("Extracting Real weight energy concentration from Qwen3-30B-A3B...")
    print("==================================================================")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Loading model on device:", device)
    
    # Load model in bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-30B-A3B",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    print("Model loaded successfully!")
    
    # Layers we want to profile to get a comprehensive view
    layers_to_profile = [0, 12, 24, 36, 47]
    d_model = 2048
    ffn_dim = 768
    num_experts = 128
    
    # Cumulative energy benchmarks
    pct_1_idx = int(0.01 * d_model)
    pct_5_idx = int(0.05 * d_model)
    pct_10_idx = int(0.10 * d_model)
    pct_20_idx = int(0.20 * d_model)
    
    all_layer_stats = {}
    
    # For global average
    global_energy_curves = []
    
    for layer_idx in layers_to_profile:
        print(f"\nProfiling Layer {layer_idx}...")
        # Access down_proj weight shape: [128, 2048, 768]
        # down_proj represents [num_experts, out_features (hidden_size), in_features (intermediate_size)]
        down_proj_weight = model.model.layers[layer_idx].mlp.experts.down_proj.detach().float()
        
        layer_curves = []
        
        # Profile a sample of experts to speed up calculation, e.g. 8 experts per layer
        sampled_experts = [0, 16, 32, 48, 64, 80, 96, 112]
        
        for exp_idx in sampled_experts:
            expert_weight = down_proj_weight[exp_idx] # [2048, 768]
            
            # For each neuron (column of the expert_weight matrix)
            for neuron_idx in range(ffn_dim):
                neuron_contrib = expert_weight[:, neuron_idx] # size [2048]
                abs_contrib = torch.abs(neuron_contrib).cpu().numpy()
                
                # Sort descending
                sorted_contrib = np.sort(abs_contrib)[::-1]
                total_energy = np.sum(sorted_contrib)
                if total_energy > 0:
                    cum_energy = np.cumsum(sorted_contrib) / total_energy
                    layer_curves.append(cum_energy)
                    global_energy_curves.append(cum_energy)
                    
        layer_curves = np.mean(layer_curves, axis=0)
        
        layer_1 = layer_curves[pct_1_idx] * 100
        layer_5 = layer_curves[pct_5_idx] * 100
        layer_10 = layer_curves[pct_10_idx] * 100
        layer_20 = layer_curves[pct_20_idx] * 100
        
        print(f"Layer {layer_idx} Results:")
        print(f" - Top 1% channels:  {layer_1:.2f}% energy")
        print(f" - Top 5% channels:  {layer_5:.2f}% energy")
        print(f" - Top 10% channels: {layer_10:.2f}% energy")
        print(f" - Top 20% channels: {layer_20:.2f}% energy")
        
        all_layer_stats[str(layer_idx)] = {
            "top_1_pct_energy": float(layer_1),
            "top_5_pct_energy": float(layer_5),
            "top_10_pct_energy": float(layer_10),
            "top_20_pct_energy": float(layer_20)
        }
        
    global_curves = np.mean(global_energy_curves, axis=0)
    g_1 = global_curves[pct_1_idx] * 100
    g_5 = global_curves[pct_5_idx] * 100
    g_10 = global_curves[pct_10_idx] * 100
    g_20 = global_curves[pct_20_idx] * 100
    
    print("\n=======================================================")
    print("GLOBAL AVERAGED WEIGHT ENERGY CONCENTRATION:")
    print(f" - Top 1% channels:  {g_1:.2f}% energy")
    print(f" - Top 5% channels:  {g_5:.2f}% energy")
    print(f" - Top 10% channels: {g_10:.2f}% energy")
    print(f" - Top 20% channels: {g_20:.2f}% energy")
    print("=======================================================")
    
    results = {
        "global_average": {
            "top_1_pct_energy": float(g_1),
            "top_5_pct_energy": float(g_5),
            "top_10_pct_energy": float(g_10),
            "top_20_pct_energy": float(g_20)
        },
        "layer_wise": all_layer_stats
    }
    
    output_path = "/home/palakm/MoEServingSim/qwen3_30b_plots/real_neuron_energy_distribution.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Successfully saved real Qwen3 weight energy metrics to: {output_path}")

if __name__ == "__main__":
    main()
