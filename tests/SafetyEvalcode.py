from pathlib import Path

import yaml
from tabulate import tabulate

from src.llm_safety_eval import SafetyEvaluator

if __name__ == "__main__":
    cur_dir = Path(__file__).resolve()
    yaml_path = cur_dir.parent / "results.yaml"
    with open(yaml_path, "r") as f:
        results = yaml.load(f, Loader=yaml.UnsafeLoader)

    eval = SafetyEvaluator(results, [True, False])

    eval = SafetyEvaluator(results, [True, False])
    metrics = eval.get_all_rate_metrics()

    approval_stage = list(iter(metrics))
    first_approval_stage = approval_stage[0]
    first_detector = next(iter(metrics[first_approval_stage]))
    metrics_headers = list(iter(metrics[first_approval_stage][first_detector]))

    print(metrics_headers)
    print("=" * 40)

    rates = eval.get_all_rate_metrics()
    cr = eval.get_all_classification_reports()
    latencies = eval.get_all_latency_metrics()
    eval.print_metrics_in_table(rates)
    eval.print_metrics_in_table(cr)
    eval.print_metrics_in_table(latencies)


    def catshit(rates):
        first_approval_type = next(iter(rates))  # key = input_approvals
        first_detector = next(iter(rates[first_approval_type]))  # SimpleBERT1_input
        headers = list(rates[first_approval_type][first_detector].keys())
        print(headers)

        rows = [
            list(metrics.values()) for metrics in rates[first_approval_type].values()
        ]
        print(rows)

        print(tabulate(rows, headers))

        for i in iter(rates):
            print(i)

        approval_stage = list(iter(rates))
        detectors = list(iter(rates[approval_stage[0]]))
        headers = list(iter(rates[approval_stage[0]][detectors[0]]))
        headers = ["Detectors"] + headers
        for s in approval_stage:
            print("=" * len(s))
            print(s.upper())
            print("=" * len(s))

            table_metrics = []

            detectors = list(iter(rates[s]))

            for d in detectors:
                metrics = rates[s][d].values()
                row = [d] + list(metrics)
                table_metrics.append(row)

            print(tabulate(table_metrics, headers, tablefmt="github"))



    """
    #cm = eval.get_confusion_matrix("input_approvals", "SimpleBERT1_input")
    #print(cm)
    eval.display_confusion_matrix("input_approvals", "SimpleBERT1_input")
    print(eval.get_latency("input_approvals", "SimpleBERT1_input"))
    #print(eval.get_rate_metrics("input_approvals", "SimpleBERT1_input"))
    print(eval.get_all_rate_metrics())
    print(eval.get_all_classifcation_reports())
    print(eval.get_all_confusion_matrices())
    """