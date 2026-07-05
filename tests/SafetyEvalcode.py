from pathlib import Path

import yaml

from src.llm_safety_eval import SafetyEvaluator

if __name__ == "__main__":
    cur_dir = Path(__file__).resolve()
    yaml_path = cur_dir.parent / "results.yaml"
    with open(yaml_path, "r") as f:
        results = yaml.load(f, Loader=yaml.UnsafeLoader)

    eval = SafetyEvaluator(results, [True, False])

    #cm = eval.get_confusion_matrix("input_approvals", "SimpleBERT1_input")
    #print(cm)
    eval.display_confusion_matrix("input_approvals", "SimpleBERT1_input")
    print(eval.get_latency("input_approvals", "SimpleBERT1_input"))
    #print(eval.get_rate_metrics("input_approvals", "SimpleBERT1_input"))
    print(eval.get_all_rate_metrics())
    print(eval.get_all_classifcation_reports())
    print(eval.get_all_confusion_matrices())