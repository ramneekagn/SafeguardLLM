import json
import sys
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

from safeguard_llm.evaluator import SafetyEvaluator
from tabulate import tabulate


def run_evalulation(results_dir): 
    json_path = Path(results_dir)
    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    evaluator = SafetyEvaluator(results, truth_rule= lambda x,y: x )

    input_class_metrics = evaluator.get_rate_metrics("input_disapprovals", "InputRobertaJBDetector")
    print(input_class_metrics)
    internal_class_metrics = evaluator.get_rate_metrics("internal_disapprovals", "LPInternalDetector1")
    print(internal_class_metrics )
    evaluator = SafetyEvaluator(results, truth_rule= lambda x,y: y )
    output_class_metrics = evaluator.get_rate_metrics("output_disapprovals", "OutputRobertaJBDetector")
    print(output_class_metrics)



run_evalulation("results_jb_preliminary_dataset_judged.json")
