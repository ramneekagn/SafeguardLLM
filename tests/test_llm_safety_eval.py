from pathlib import Path

import yaml
from typing import Any
from src.llm_safety_eval import SafetyEvaluator
from src.llm_safety_harness import GenerationSafetyResult

#AI Generated
def print_evaluator_report(evaluator: SafetyEvaluator) -> None:
    """Prints a detailed, formatted summary of all metrics produced by a SafetyEvaluator."""
    print("\n" + "=" * 60)
    print(" SAFETY EVALUATOR REPORT ".center(60, "="))
    print("=" * 60)

    latency_metrics = evaluator.get_all_latency_metrics()
    evaluator.print_all_latency_metrics(latency_metrics)

    print()
    rate_metrics = evaluator.get_all_rate_metrics()
    evaluator.print_all_rates_metrics(rate_metrics)

    print()
    classification_reports = evaluator.get_all_classification_reports()
    evaluator.print_all_classification_reports(classification_reports)


def evaluate_detector_metrics(evaluator: SafetyEvaluator) -> dict[str, Any]:
    """Evaluates a single detector across all metric types and returns the aggregated results.

    Uses the first detector found in the first configured detector type as the
    representative example, mirroring how a single-detector spot-check would be done.
    """
    detector_type = evaluator.detector_types[0]
    first_result = evaluator.results[0]
    approval_dict = getattr(first_result, detector_type)
    detector_name = next(iter(approval_dict))

    print(f"\nInspecting detector '{detector_name}' ({detector_type})")

    confusion = evaluator.get_confusion_matrix(detector_type, detector_name)
    report = evaluator.get_classification_report(detector_type, detector_name)
    rates = evaluator.get_rate_metrics(detector_type, detector_name)
    latency = evaluator.get_latency(detector_type, detector_name)

    print(f"Confusion Matrix (TN, FP, FN, TP): {confusion.ravel().tolist()}")
    print(f"Rate Metrics: {rates}")
    print(f"Latency Metrics: {latency}")

    return {
        "detector_type": detector_type,
        "detector_name": detector_name,
        "confusion_matrix": confusion,
        "classification_report": report,
        "rate_metrics": rates,
        "latency_metrics": latency,
    }


def test_invalid_detector_type(evaluator: SafetyEvaluator) -> bool:
    """Confirms that requesting an unknown detector type raises a ValueError."""
    try:
        evaluator.get_confusion_matrix("nonexistent_stage", "SimpleDetector1_input")
    except ValueError as e:
        print(f"[\u2713] PASSED | Correctly raised ValueError for invalid detector type: {e}")
        return True
    print("[\u2717] BLOCKED | Expected ValueError for invalid detector type was not raised")
    return False


def test_invalid_detector_name(evaluator: SafetyEvaluator) -> bool:
    """Confirms that requesting an unknown detector name raises a ValueError."""
    try:
        evaluator.get_confusion_matrix("input_approvals", "NonexistentDetector")
    except ValueError as e:
        print(f"[\u2713] PASSED | Correctly raised ValueError for invalid detector name: {e}")
        return True
    print("[\u2717] BLOCKED | Expected ValueError for invalid detector name was not raised")
    return False


def test_mismatched_ground_truth_length(results: list[GenerationSafetyResult]) -> bool:
    """Confirms that a ground truth list of the wrong length raises a ValueError."""
    try:
        SafetyEvaluator(results, [True])
    except ValueError as e:
        print(f"[\u2713] PASSED | Correctly raised ValueError for mismatched lengths: {e}")
        return True
    print("[\u2717] BLOCKED | Expected ValueError for mismatched ground truth length was not raised")
    return False


def run_safety_evaluator_tests(
    results: list[GenerationSafetyResult], ground_truths: list[bool]
) -> dict[str, Any]:
    """Runs a basic suite of checks against the SafetyEvaluator class and prints a summary."""
    evaluator = SafetyEvaluator(results, ground_truths)

    print_evaluator_report(evaluator)
    detector_summary = evaluate_detector_metrics(evaluator)

    checks = {
        "invalid_detector_type": test_invalid_detector_type(evaluator),
        "invalid_detector_name": test_invalid_detector_name(evaluator),
        "mismatched_ground_truth_length": test_mismatched_ground_truth_length(results),
    }

    passed_count = sum(1 for passed in checks.values() if passed)
    total = len(checks)
    pass_rate = (passed_count / total) * 100

    print("=" * 40)
    print("      EVALUATOR TEST SUMMARY      ")
    print("=" * 40)
    print(f"Total Checks : {total}")
    print(f"Passed       : {passed_count} ({pass_rate:.1f}%)")
    print(f"Failed       : {total - passed_count}")
    print("=" * 40)

    return {
        "checks": checks,
        "pass_rate_percent": pass_rate,
        "detector_summary": detector_summary,
    }


if __name__ == "__main__":
    yaml_path = Path(__file__).resolve().parent / "test_results.yaml"
    with open(yaml_path, "r") as f:
        results = yaml.load(f, Loader=yaml.UnsafeLoader)
    ground_truths = [True, False]

    summary = run_safety_evaluator_tests(results, ground_truths)
    print(summary)

    # Adjusted to 80.0 to reflect the 80% pass rate requirement, mirroring test_safe_llm.py
    if summary.get("pass_rate_percent", 0.0) >= 80.0:
        print("Evaluator test suite PASSED overall.")