import json
import sys
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

from safeguard_llm.evaluator import SafetyEvaluator
from tabulate import tabulate
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from datasets import load_dataset

#AI generated logger prompt: "generate me log file that runs the evaluations and logs the results in a text file"
def log_eval(name,sample_size, ensemble_results_dir,  log_dir="logs"):
    """
    Executes the standard and ensemble evaluations, capturing all console 
    outputs and writing them into a timestamped log file.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"eval_log_{name}.txt"
    
    print(f"Initiating evaluation. Writing logs to {log_file}...")
    
    with open(log_file, "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            header_data = [
                ["Evaluation Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                ["Ensemble Results Source", ensemble_results_dir],
                ["Sample size", sample_size ]
            ]
            print(tabulate(header_data, headers=["Metadata", "Value"], tablefmt="fancy_grid"))
            print("\n" + "=" * 80 + "\n")
            
            print(">>> RUNNING STANDARD EVALUATION <<<\n")
            run_evalulation(ensemble_results_dir)
            print("\n" + "=" * 80 + "\n")
            
            print(">>> RUNNING ENSEMBLE EVALUATION <<<\n")
            run_evalulation_ensemble(ensemble_results_dir)
            print("\n" + "=" * 80)
            print("Evaluation completed successfully.")
            
    print("Logging complete.")


def run_evalulation_ensemble(results_dir): 
    json_path = Path(results_dir)
    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    print("ensemble")
    evaluator = SafetyEvaluator(results, truth_rule=lambda x, y: x and y)
    rate_metrics = evaluator.get_ensemble_rate_metrics()
    class_report = evaluator.get_ensemble_classification_report()

    table_data = list(rate_metrics.items())
    table_data.append(("Ideal Refusal Rate", evaluator.get_ideal_refusal()))
    
    print(tabulate(table_data, headers=["Ensemble Metric", "Value"]))
    print("\nClassification Report:")
    print(tabulate(class_report.items(), headers=["Input Metric", "Value"]))

def run_evalulation(results_dir): 
    json_path = Path(results_dir)
    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    eval_input_internal = SafetyEvaluator(results, truth_rule=lambda x, y: x)
    all_detectors= eval_input_internal._get_detector_names()

    for category in ["input_disapprovals", "internal_disapprovals"]:
        for detector_name in all_detectors.get(category, []):
            print(f">>> {category.upper()}: {detector_name} <<<")
            try:
                metrics = eval_input_internal.get_rate_metrics(category, detector_name)
                table_data = list(metrics.items())
                table_data.append(("Ideal Refusal Rate", eval_input_internal.get_ideal_refusal()))
                print(tabulate(table_data, headers=["Metric", "Value"], tablefmt="github"))
            except Exception as e:
                print(f"Error evaluating {detector_name}: {e}")
            print("\n" + "-" * 50 + "\n")

    eval_output = SafetyEvaluator(results, truth_rule=lambda x, y: y)
    for detector_name in all_detectors.get("output_disapprovals", []):
        print(f">>> OUTPUT DISAPPROVALS: {detector_name} <<<")
        try:
            metrics = eval_output.get_rate_metrics("output_disapprovals", detector_name)
            table_data = list(metrics.items())
            table_data.append(("Ideal Refusal Rate", eval_output.get_ideal_refusal()))
            print(tabulate(table_data, headers=["Metric", "Value"], tablefmt="github"))
        except Exception as e:
            print(f"Error evaluating {detector_name}: {e}")
        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    run_evalulation("results/safe_llm_config_base_multiple_smoke_test_2/50_50_hard_jbb-benign_jbb-harmful/judged_and_rule.json")
    run_evalulation_ensemble("results/safe_llm_config_base_multiple_smoke_test_2/50_50_hard_jbb-benign_jbb-harmful/judged_and_rule.json")
