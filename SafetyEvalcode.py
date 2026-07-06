"""
safety_eval_tables.py

Helper functions to turn the nested dict output of SafetyEvaluator
(get_all_rate_metrics, get_all_classifcation_reports, get_all_confusion_matrices)
into flat, readable pandas DataFrames / printable tables.

Usage:
    from safety_eval_tables import (
        rate_metrics_to_df,
        classification_reports_to_df,
        confusion_matrices_to_df,
        print_all_tables,
    )

    rate_df = rate_metrics_to_df(eval.get_all_rate_metrics())
    cls_df = classification_reports_to_df(eval.get_all_classifcation_reports())
    cm_df = confusion_matrices_to_df(eval.get_all_confusion_matrices(), labels=[True, False])

    print_all_tables(eval)
"""

from __future__ import annotations
import pandas as pd
import numpy as np


def _unwrap(x):
    """Convert numpy scalars to plain python floats/ints for clean display."""
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


# ---------------------------------------------------------------------------
# 1. Rate metrics (TPR, FNR, FPR, TNR, Refusal)
# ---------------------------------------------------------------------------
def rate_metrics_to_df(rate_metrics: dict) -> pd.DataFrame:
    """
    rate_metrics looks like:
        {
          "input_approvals": {
              "SimpleBERT1_input": {"TPR": ..., "FNR": ..., ...},
              ...
          },
          "internal_approvals": {...},
          "output_approvals": {...},
        }

    Returns a tidy DataFrame with columns:
        stage | detector | TPR | FNR | FPR | TNR | Refusal
    """
    rows = []
    for stage, detectors in rate_metrics.items():
        for detector_name, metrics in detectors.items():
            row = {"stage": stage, "detector": detector_name}
            row.update({k: _unwrap(v) for k, v in metrics.items()})
            rows.append(row)

    df = pd.DataFrame(rows)
    # nice column order if all expected metrics are present
    preferred = ["stage", "detector", "TPR", "FNR", "FPR", "TNR", "Refusal"]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[cols].sort_values(["stage", "detector"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Classification reports (precision/recall/f1 per class + accuracy)
# ---------------------------------------------------------------------------
def classification_reports_to_df(classification_reports: dict) -> pd.DataFrame:
    """
    classification_reports looks like sklearn's classification_report(output_dict=True)
    nested under stage -> detector, e.g.:
        {
          "input_approvals": {
              "SimpleBERT1_input": {
                  "False": {"precision":..., "recall":..., "f1-score":..., "support":...},
                  "True":  {...},
                  "accuracy": 0.5,
                  "macro avg": {...},
                  "weighted avg": {...},
              },
              ...
          },
          ...
        }

    Returns a tidy long-format DataFrame with columns:
        stage | detector | class | precision | recall | f1-score | support
    "accuracy" is folded in as its own row with class="accuracy" (precision/recall/f1 = the accuracy value).
    """
    rows = []
    for stage, detectors in classification_reports.items():
        for detector_name, report in detectors.items():
            for class_name, metrics in report.items():
                if class_name == "accuracy":
                    rows.append({
                        "stage": stage,
                        "detector": detector_name,
                        "class": "accuracy",
                        "precision": _unwrap(metrics),
                        "recall": _unwrap(metrics),
                        "f1-score": _unwrap(metrics),
                        "support": np.nan,
                    })
                else:
                    row = {"stage": stage, "detector": detector_name, "class": class_name}
                    row.update({k: _unwrap(v) for k, v in metrics.items()})
                    rows.append(row)

    df = pd.DataFrame(rows)
    preferred = ["stage", "detector", "class", "precision", "recall", "f1-score", "support"]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[cols].sort_values(["stage", "detector", "class"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Confusion matrices
# ---------------------------------------------------------------------------
def confusion_matrices_to_df(confusion_matrices: dict, labels=None) -> dict[str, pd.DataFrame]:
    """
    confusion_matrices looks like:
        {
          "input_approvals": {
              "SimpleBERT1_input": np.array([[0, 1], [0, 1]]),
              ...
          },
          ...
        }

    Returns a dict keyed by "stage/detector" -> labeled 2x2 (or NxN) DataFrame,
    e.g. rows = actual labels, cols = predicted labels.
    """
    if labels is None:
        labels = ["False", "True"]
    label_strs = [str(l) for l in labels]

    result = {}
    for stage, detectors in confusion_matrices.items():
        for detector_name, cm in detectors.items():
            cm = np.array(cm)
            df = pd.DataFrame(
                cm,
                index=[f"actual_{l}" for l in label_strs],
                columns=[f"pred_{l}" for l in label_strs],
            )
            result[f"{stage}/{detector_name}"] = df
    return result


# ---------------------------------------------------------------------------
# 4. Convenience: pretty-print everything at once
# ---------------------------------------------------------------------------
def print_all_tables(evaluator, labels=None) -> None:
    """
    Given a SafetyEvaluator instance, pulls all metrics and prints clean tables.
    """
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")

    print("=" * 80)
    print("RATE METRICS")
    print("=" * 80)
    rate_df = rate_metrics_to_df(evaluator.get_all_rate_metrics())
    print(rate_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("CLASSIFICATION REPORTS")
    print("=" * 80)
    cls_df = classification_reports_to_df(evaluator.get_all_classifcation_reports())
    print(cls_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("CONFUSION MATRICES")
    print("=" * 80)
    cm_dict = confusion_matrices_to_df(evaluator.get_all_confusion_matrices(), labels=labels)
    for key, df in cm_dict.items():
        print(f"\n--- {key} ---")
        print(df.to_string())

    return rate_df, cls_df, cm_dict


if __name__ == "__main__":
    # Example standalone usage against your existing script:
    from pathlib import Path
    import yaml
    from src.llm_safety_eval import SafetyEvaluator

    cur_dir = Path(__file__).resolve()
    yaml_path = cur_dir.parent / "results.yaml"
    with open(yaml_path, "r") as f:
        results = yaml.load(f, Loader=yaml.UnsafeLoader)

    evaluator = SafetyEvaluator(results, [True, False])
    print_all_tables(evaluator, labels=[True, False])