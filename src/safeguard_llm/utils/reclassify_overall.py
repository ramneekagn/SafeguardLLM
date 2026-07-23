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
