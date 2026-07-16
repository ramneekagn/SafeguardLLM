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
    """ Reclassifies the results list using the specified rule function """
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

def and_rule(entry):

    input_bert = entry["input_disapprovals"]["InputRobertaJBDetector"]["disapproved"]
    input_lp = entry["internal_disapprovals"]["LPInternalDetector1"]["disapproved"]
    output_bert = entry["output_disapprovals"]["OutputRobertaJBDetector"]["disapproved"]
    return input_bert and input_lp and output_bert

if __name__ == "__main__":
    json_path = Path("results_thres_0.9_jb_preliminary_dataset_judged.json")
    save_path = Path("results_thres_0.9_jb_preliminary_dataset_judged_reclassified_and_rule.json")

    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    results = reclassify(results, and_rule)
    print(f"Saving reclassified results to: {save_path}")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)