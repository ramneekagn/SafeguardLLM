import json
from pathlib import Path
from run_and_judge import run_and_judge, _prepare_output_file
from reclassify_overall import reclassify, implication_rule, and_rule
from run_evaluation_reworked import log_eval
from dotenv import load_dotenv
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

def run_pipeline(dataset_name, output_filename="results_jb.json", rule_func=implication_rule, sample_size=100, batch_size=8, log_dir="logs"):
    run_and_judge(
        dataset_name=dataset_name,
        output_filename=output_filename,
        sample_size=sample_size,
        batch_size=batch_size,
        willJudge=True
    )

    base_results_path = _prepare_output_file(output_filename, dataset_name)
    results_path = base_results_path.with_stem(f"{base_results_path.stem}_judged")
    ensemble_path = results_path.with_stem(f"{results_path.stem}_reclassified_{rule_func.__name__}")

    print(f"Reading judged results from: {results_path}")
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    reclassified_results = reclassify(results, rule_func)

    print(f"Saving reclassified results to: {ensemble_path}")
    with open(ensemble_path, "w", encoding="utf-8") as f:
        json.dump(reclassified_results, f, indent=4)
    log_eval(results_path, ensemble_path, log_dir=log_dir)

load_dotenv()
dataset_name = "preliminary_dataset"

run_pipeline(
    dataset_name=dataset_name,
    output_filename="results_thres_0.95inp_0.95int_0.5out_jb.json",
    rule_func=and_rule,
    sample_size=100,
    batch_size=8
)