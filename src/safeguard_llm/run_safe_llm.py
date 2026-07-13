from safeguard_llm.safety_harness import SafeLLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from safeguard_llm.dataset.dataset_provider import DatasetProvider
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from importlib import resources
import os


def run_safe_llm(dataset_name, dataset_size, prompt_col, label_col,subset=None):

    provider = DatasetProvider(dataset_name, source = "hf")
    ds = provider.get_dataset(split="default", seed=40, subset=subset, size=dataset_size)
    dataloader = DataLoader(ds, batch_size=1)
    outputs = []
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-1.7B", padding_side="left"
    )
    safe_model = SafeLLM(
        model, tokenizer, max_gen_len=32, config_path="src/safeguard_llm/config/safe_llm_config.yaml"
    )
    all_labels = []
    for batch in tqdm(dataloader): 
        inputs = batch[prompt_col]
        labels = batch[label_col]
        all_labels.extend(labels)
        output = safe_model.generate(inputs)
        outputs.extend(output)
    
    return outputs,all_labels
