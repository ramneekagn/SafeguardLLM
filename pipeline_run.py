from pathlib import Path
from safeguard_llm.evaluator import SafetyEvaluator
from safeguard_llm.run_safe_llm import run_safe_llm
from safeguard_llm.utils.save_results import save_results_as_json
from transformers import AutoModelForCausalLM, AutoTokenizer
from safeguard_llm.safety_harness import SafeLLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from safeguard_llm.dataset.dataset_provider import DatasetProvider
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from importlib import resources
from datasets import load_dataset
from torch.utils.data import DataLoader
import shutil
from pathlib import Path
from datasets import load_dataset
dataset = load_dataset("allenai/wildjailbreak", "eval", delimiter="\t", keep_default_na=False)
ds = dataset["train"]
print(ds)
dataloader = DataLoader(ds, batch_size=16)
outputs = []
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B", device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-1.7B", padding_side="left"
)

safe_model = SafeLLM(
    model, tokenizer, max_gen_len=64, config_path="src/safeguard_llm/config/safe_llm_config.yaml"
)
all_labels = []
for batch in tqdm(dataloader): 
    inputs = batch["adversarial"]
    labels = batch["label"]
    all_labels.extend(labels)
    output = safe_model.generate(inputs)
    outputs.extend(output)

test_file = Path("results_jb_first_run.json_")
save_results_as_json(outputs, test_file)
