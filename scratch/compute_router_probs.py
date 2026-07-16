import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    model_id = "Qwen/Qwen3-30B-A3B"
    print("Loading model on GPU to inspect router...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto", 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Inspect gate module structure
    layer = model.model.layers[10]
    gate_module = layer.mlp.gate
    print("Gate module class:", gate_module.__class__.__name__)
    print("Gate module structure:")
    print(gate_module)
    
    # We will hook gate_module itself, and inside the hook we can compute raw softmax if we find the weight parameter
    # Let's find any linear layer or weight parameter inside gate_module
    weight_param = None
    for name, param in gate_module.named_parameters():
        print(f"  Parameter found: {name}, shape: {param.shape}")
        if "weight" in name:
            weight_param = param
            
    if weight_param is None:
        print("No weight parameter found in gate module.")
        return
        
    # Hook the gate module to compute raw softmax on inputs
    raw_softmax_list = []
    
    def gate_hook(module, args, output):
        # args[0] is hidden_states of shape [seq_len, hidden_dim] or [batch, seq_len, hidden_dim]
        h = args[0].detach().float()
        # Compute logits manually: logits = h @ weight.T
        logits = torch.matmul(h, weight_param.detach().float().t())
        probs = torch.softmax(logits, dim=-1).cpu()
        raw_softmax_list.append(probs)
        
    h = gate_module.register_forward_hook(gate_hook)
    
    text = "Explain the connection between gradient descent and convex optimization."
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        _ = model(**inputs)
        
    h.remove()
    
    all_raw_probs = torch.cat(raw_softmax_list, dim=0) # [seq_len, num_experts]
    all_raw_probs = all_raw_probs.view(-1, all_raw_probs.size(-1)).numpy()
    
    # Sort raw softmax probabilities descending per token
    sorted_raw = np.sort(all_raw_probs, axis=-1)[:, ::-1]
    
    raw_t1 = sorted_raw[:, 0]
    raw_t4_cum = np.sum(sorted_raw[:, :4], axis=-1)
    raw_t8_cum = np.sum(sorted_raw[:, :8], axis=-1)
    
    print("\n--- Router Softmax Probability Distribution ---")
    print(f"  Top-1 Probability:")
    print(f"    Mean:   {np.mean(raw_t1):.4f}")
    print(f"    Std:    {np.std(raw_t1):.4f}")
    print(f"    Median: {np.median(raw_t1):.4f}")
    
    print(f"  Top-4 Cumulative Probability:")
    print(f"    Mean:   {np.mean(raw_t4_cum):.4f}")
    print(f"    Std:    {np.std(raw_t4_cum):.4f}")
    print(f"    Median: {np.median(raw_t4_cum):.4f}")
    
    print(f"  Top-8 Cumulative Probability:")
    print(f"    Mean:   {np.mean(raw_t8_cum):.4f}")
    print(f"    Std:    {np.std(raw_t8_cum):.4f}")
    print(f"    Median: {np.median(raw_t8_cum):.4f}")

if __name__ == "__main__":
    main()
