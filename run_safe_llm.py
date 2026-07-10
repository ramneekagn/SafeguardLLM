from src.safety_harness import SafeLLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.dataset.dataset_provider import DatasetProvider
from torch.utils.data import DataLoader
from evaluator import SafetyEvaluator
from tqdm.auto import tqdm

def run_safe_llm(dataset_size):
    outputs = []
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-0.6B", padding_side="left"
    )
    safe_model = SafeLLM(
        model, tokenizer, max_gen_len=64, config_path=r"src/config/safe_llm_config.yaml"
    )
    provider = DatasetProvider(dataset_name = "allenai/wildjailbreak", source = "hf")
    ds = provider.get_dataset(split="eval", seed=40, size=dataset_size)
    dataloader = DataLoader(ds, batch_size=16)
    all_labels = []
    for batch in tqdm(dataloader): 
        inputs = batch["adversarial"]
        all_labels.extend(batch["label"])
        output = safe_model.generate(inputs)
        outputs.extend(output)
    
    return outputs,all_labels
