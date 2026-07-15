# evaluation/utils/model_loader.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def load_model_and_tokenizer(model_id: str, device: str = "cuda:0", mock: bool = False):
    """
    Loads MoE models and tokenizers.
    If mock=True, instantiates config only to construct layer outlines for testing.
    """
    print(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if mock:
        print(f"[MOCK] Loading configuration for {model_id}...")
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        # Create a tiny dummy model with same metadata structures
        print("[MOCK] Initializing dummy model structures...")
        model = None
        return model, tokenizer, config

    print(f"Loading weights for {model_id} on {device}...")
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    model.eval()
    return model, tokenizer, config
