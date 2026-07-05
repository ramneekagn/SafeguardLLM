from typing import Tuple, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
def get_model_tokenizer():
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-0.6B", padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model,tokenizer