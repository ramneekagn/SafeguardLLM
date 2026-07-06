from pathlib import Path

import yaml
import numpy as np

from src.llm_safety_eval import SafetyEvaluator


def load_result_yaml():
    with open(Path(__file__).parent / "results.yaml") as f:
        return yaml.load(f, Loader=yaml.UnsafeLoader)


def get_detector(results):
    detector_type = "input_approvals"
    detector_name = next(iter(results[0].input_approvals))
    return detector_type, detector_name


def test_evaluator_runs_without_error():
    results = load_result_yaml()

    evaluator = SafetyEvaluator(results, [True, False])

    evaluator.get_all_confusion_matrices()
    evaluator.get_all_classification_reports()
    evaluator.get_all_latency_metrics()
    evaluator.get_all_rate_metrics()

def test_print_functions():
    results = load_result_yaml()

    evaluator = SafetyEvaluator(results, [True, False])

    evaluator.print_all_latency_metrics(
        evaluator.get_all_latency_metrics()
    )

    evaluator.print_all_rates_metrics(
        evaluator.get_all_rate_metrics()
    )

    evaluator.print_all_classification_reports(
        evaluator.get_all_classification_reports()
    )


if __name__ == "__main__":
    test_evaluator_runs_without_error()
    test_print_functions()