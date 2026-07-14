from pathlib import Path
from safeguard_llm.run_safe_llm import run_safe_llm
from safeguard_llm.utils.save_results import save_results_as_json
from transformers import AutoModelForCausalLM, AutoTokenizer
from safeguard_llm.safety_harness import SafeLLM
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from pathlib import Path
from datasets import load_dataset,concatenate_datasets
from safeguard_llm.utils.output_judge import Benchmark_Eval
from dotenv import load_dotenv
import asyncio

dataset = load_dataset("allenai/wildjailbreak", "eval", delimiter="\t", keep_default_na=False)
print(dataset)
ds_true = dataset["train"].filter(lambda elm: elm["label"] == 1)
ds_false = dataset["train"].filter(lambda elm: elm["label"] == 0)
ds_true = ds_true.select(range(210))
ds_false = ds_false.select(range(210))
dataset = concatenate_datasets([ds_true, ds_false])

dataloader = DataLoader(dataset, batch_size=16)
outputs = []
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B", device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-1.7B", padding_side="left"
)

safe_model = SafeLLM(
    model, tokenizer, max_gen_len=256, config_path="src/safeguard_llm/config/safe_llm_config.yaml"
)
all_labels = []
for batch in tqdm(dataloader): 
    inputs = batch["adversarial"]
    labels = batch["label"]
    all_labels.extend(labels)
    output = safe_model.generate(inputs)
    outputs.extend(output)

for i, res in enumerate(outputs): 
    val = all_labels[i]
    val = val.item()
    if val == 0: 
        res.output_label_gold = False
    res.prompt_label_gold = bool(val)

test_file = Path("results_jb_first_run.json")
save_results_as_json(outputs, test_file)
load_dotenv()
json_config = Path("results_jb_first_run.json")
#json_fix = Path("results_jb_first_run_judged.json")
#response_to_label(json_fix)
be = Benchmark_Eval(json_config)
asyncio.run(be.execute_judgement())