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
import json

def reclassify(results, rule_func):

    for entry in results:
        entry["overall_disapproval"] = rule_func(entry)
    return results


def implication_rule(entry):

    input_bert = entry["input_disapprovals"]["InputRobertaJBDetector"]["disapproved"]
    input_lp = entry["internal_disapprovals"]["LPInternalDetector1"]["disapproved"]
    conditional_lp = entry["internal_disapprovals"]["LPInternalDetectorConditonal"]["disapproved"]
    output_bert = entry["output_disapprovals"]["OutputRobertaJBDetector"]["disapproved"]

    is_input_harmful = input_bert or input_lp

    #  (A => B) equivalent to (not A or B)
    implication = (not is_input_harmful) or conditional_lp

    return implication and output_bert

out_path = Path("results_jb_reclassified.json")
json_path = Path("results_jb_experiment_1_judged.json")

with open(json_path, "r") as f:
    results = json.load(f)
results = reclassify(results, implication_rule)

with open(out_path, "w") as f:
            json.dump(results, f, indent=4)       