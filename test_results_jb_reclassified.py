import json
from pathlib import Path
from safeguard_llm.evaluator import SafetyEvaluator
from tabulate import tabulate
# AI test
json_path = Path("results_jb_reclassified.json")

if not json_path.exists():
    raise FileNotFoundError(
        f"Could not find '{json_path.name}' in the current working directory: {Path.cwd()}"
    )

with open(json_path, "r", encoding="utf-8") as f:
    outputs = json.load(f)

print(f"Successfully loaded {len(outputs)} records from {json_path.name}.\n")

print("=" * 60)
print("EVALUATING INPUT DETECTORS AGAINST PROMPT GOLD LABELS")
print("=" * 60)

input_evaluator = SafetyEvaluator(
    results=outputs,
    truth_rule=lambda x, y: x,
    detector_types=("input_disapprovals",)
)

input_class_metrics = input_evaluator.get_all_classification_reports()
input_evaluator.print_all_classification_reports(input_class_metrics)

print("\nInput Detectors - Rate Metrics:")
input_rate_metrics = input_evaluator.get_all_rate_metrics()
input_evaluator.print_all_rates_metrics(input_rate_metrics)


print("\n" + "=" * 60)
print("EVALUATING OUTPUT DETECTORS AGAINST OUTPUT GOLD LABELS")
print("=" * 60)

output_evaluator = SafetyEvaluator(
    results=outputs,
    truth_rule=lambda x, y: y,
    detector_types=("output_disapprovals",)
)

output_class_metrics = output_evaluator.get_all_classification_reports()
output_evaluator.print_all_classification_reports(output_class_metrics)

print("\nOutput Detectors - Rate Metrics:")
output_rate_metrics = output_evaluator.get_all_rate_metrics()
output_evaluator.print_all_rates_metrics(output_rate_metrics)


print("\n" + "=" * 60)
print("EVALUATING LPINTERNALDETECTOR1 AGAINST PROMPT GOLD LABELS")
print("=" * 60)

# Default detector_types are used automatically
internal_evaluator = SafetyEvaluator(
    results=outputs,
    truth_rule=lambda x, y: x,  # Using input prompt label as ground truth
)

stage = "internal_disapprovals"
detector = "LPInternalDetector1"

# Query the specific metrics for LPInternalDetector1
lp_cm = internal_evaluator.get_confusion_matrix(stage, detector)
lp_report = internal_evaluator.get_classification_report(stage, detector)
lp_rates = internal_evaluator.get_rate_metrics(stage, detector)
lp_latency = internal_evaluator.get_latency(stage, detector)

# Format and print Confusion Matrix
print("Confusion Matrix:")
print(lp_cm)

# Format and print Classification Report table
print("\nClassification Report:")
table_metrics = []
headers = ["Class/Metric", "Precision", "Recall", "F1-Score", "Support"]
accuracy = None

for label, metrics in lp_report.items():
    if isinstance(metrics, dict):
        table_metrics.append([
            label,
            metrics.get("precision"),
            metrics.get("recall"),
            metrics.get("f1-score"),
            metrics.get("support")
        ])
    else:
        accuracy = metrics

print(tabulate(table_metrics, headers, tablefmt="github"))
if accuracy is not None:
    print(f"***** Overall Accuracy {accuracy} *****")

# Format and print Rate Metrics
print("\nRate Metrics:")
rate_headers = ["Metric", "Value"]
rate_table = [[metric, val] for metric, val in lp_rates.items()]
print(tabulate(rate_table, rate_headers, tablefmt="github"))

# Format and print Latency Metrics
print("\nLatency Metrics:")
latency_headers = ["Statistic", "Value"]
latency_table = [
    [stat, f"{val:.4f} ms" if isinstance(val, float) else val] 
    for stat, val in lp_latency.items()
]
print(tabulate(latency_table, latency_headers, tablefmt="github"))
