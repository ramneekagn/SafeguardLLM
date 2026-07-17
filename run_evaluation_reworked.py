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
def log_eval(name, ensemble_results_dir, log_dir="logs"):
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
                ["Ensemble Results Source", ensemble_results_dir]
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
    
    # Input/Internal Evaluator
    evaluator = SafetyEvaluator(results, truth_rule=lambda x, y: x)

    print(">>> INPUT DETECTOR <<<")
    input_class_metrics = evaluator.get_rate_metrics("input_disapprovals", "InputDetector1")
    table_data = list(input_class_metrics.items())
    table_data.append(("Ideal Refusal Rate", evaluator.get_ideal_refusal()))
    print(tabulate(table_data, headers=["Input Metric", "Value"]))
    print("\n" + "-" * 50 + "\n")

    print(">>> INTERNAL DETECTOR <<<")
    internal_class_metrics = evaluator.get_rate_metrics("internal_disapprovals", "InternalDetector1")
    table_data = list(internal_class_metrics.items())
    table_data.append(("Ideal Refusal Rate", evaluator.get_ideal_refusal()))
    print(tabulate(table_data, headers=["Metric", "Value"]))
    print("\n" + "-" * 50 + "\n")

    # Output Evaluator
    evaluator = SafetyEvaluator(results, truth_rule=lambda x, y: y)
    
    print(">>> OUTPUT DETECTOR <<<")
    output_class_metrics = evaluator.get_rate_metrics("output_disapprovals", "OutputDetector1")
    table_data = list(output_class_metrics.items())
    table_data.append(("Ideal Refusal Rate", evaluator.get_ideal_refusal()))
    print(tabulate(table_data, headers=["Metric", "Value"]))


if __name__ == "__main__":
    run_evalulation("results/safe_llm_config_base/50_50_easy/judged_and_rule.json")
    run_evalulation_ensemble("results/safe_llm_config_base/50_50_easy/judged_and_rule.json")
