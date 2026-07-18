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
"""
def implication_rule(entry):
    input_bert = entry["input_disapprovals"]["InputDetector1"]["disapproved"]
    input_lp = entry["internal_disapprovals"]["InternalDetector1"]["disapproved"]
    conditional_lp = entry["internal_disapprovals"]["LPInternalDetectorConditonal"]["disapproved"]
    output_bert = entry["output_disapprovals"]["OutputRobertaJBDetector"]["disapproved"]

    is_input_harmful = input_bert or input_lp

    #  (A => B) equivalent to (not A or B)
    implication = (not is_input_harmful) or conditional_lp
"""
def and_rule(entry: dict) -> bool:
    disapproval_results = []
    categories = ["input_disapprovals", "internal_disapprovals", "output_disapprovals"]
    for category in categories:
        category_dict = entry.get(category, {})
        for detector_name, detector_data in category_dict.items():
            disapproved = detector_data["disapproved"]
            disapproval_results.append(disapproved)
    if not disapproval_results:
        return False
    return all(disapproval_results)

def or_rule(entry: dict) -> bool:
    disapproval_results = []
    categories = ["input_disapprovals", "internal_disapprovals", "output_disapprovals"]
    for category in categories:
        category_dict = entry.get(category, {})
        for detector_name, detector_data in category_dict.items():
            disapproved = detector_data["disapproved"]
            disapproval_results.append(disapproved)
    return any(disapproval_results)

def majority_vote(entry: dict) -> bool:
    disapproval_results = []
    categories = ["input_disapprovals", "internal_disapprovals", "output_disapprovals"]
    for category in categories:
        category_dict = entry.get(category, {})
        for detector_name, detector_data in category_dict.items():
            disapproved = detector_data["disapproved"]
            disapproval_results.append(disapproved)
    total_detectors = len(disapproval_results)
    votes_for_disapproval = sum(disapproval_results)
    majority_threshold = (total_detectors // 2) + 1
    return votes_for_disapproval >= majority_threshold

if __name__ == "__main__":
    json_path = Path("results/safe_llm_config_base_full_run_copy/50_50_easy/raw_judged.json")
    save_path = Path("results/safe_llm_config_base_full_run_copy/50_50_easy/judged_or_test.json")

    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    results = reclassify(results, or_rule)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    save_path = Path("results/safe_llm_config_base_full_run_copy/50_50_easy/judged_majority_test.json")

    results = reclassify(results, majority_vote)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)