from collections import defaultdict
from dataclasses import dataclass, field, is_dataclass, asdict
from pathlib import Path
from typing import Callable, Any, Union, Optional
import uuid

import yaml
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate


from safeguard_llm.safety_harness import GenerationSafetyResult
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)


class SafetyEvaluator:
    """A tool for evaluating the results of the SafetyHarness with different metrics.
    This class handles the calculation of several metrics for a single or all detectors:
        - Confusion Matrix
        - Classification Report
        - Latency Metrics
        - Rate Metrics

    Attributes:
        results (GenerationSafetyResult): The results of a SafetyHarness run.
        ground_truths (list[bool]): The correct labels of each prompt input of the results.
        detector_types (tuple[str, ...]): The detectors that are used in the results. Defaults to ("input_disapprovals", "internal_disapprovals", "output_disapprovals")
    """
    def __init__(
        self,
        results: list[Union[dict, Any]],
        truth_rule: Callable = lambda x, y: x or y,  
        detector_types: tuple[str, ...] = (
            "input_disapprovals",
            "internal_disapprovals",
            "output_disapprovals",
        ),
    ):
        """Initialize the SafetyEvaluator with list of results (either dicts or dataclasses)."""
        self.results = []
        for res in results:
            if is_dataclass(res):
                self.results.append(asdict(res))
            elif isinstance(res, dict):
                self.results.append(res)
            else:
                try:
                    self.results.append(dict(res))
                except (ValueError, TypeError):
                    if hasattr(res, "__dict__"):
                        self.results.append(res.__dict__)
                    else:
                        raise TypeError(f"Unsupported result type: {type(res)}")

        self.detector_types = detector_types

        input_truths = []
        output_truths = []
        for res in self.results: 
            input_truths.append(res.get("prompt_label_gold"))
            output_truths.append(res.get("output_label_gold"))

        ground_truths = []
        for i in range(len(input_truths)):
            in_val = input_truths[i]
            out_val = output_truths[i]
            ground_truths.append(truth_rule(in_val, out_val))

        if len(ground_truths) != len(self.results):
            raise ValueError(
                f"Results Length {len(self.results)} and Ground Truth Length {len(ground_truths)} do not match."
            )

        self.y_true = ground_truths
        
    def _extract_detector_values(
        self,
        detector_type: str,
        detector_name: str,
        value_key: str,
        valid_keys: tuple[str, ...] = ("disapproved", "latency", "class_name"),
    ) -> list[bool | float]:
        """Helper function that extracts the recorded resulting values for a specified detector.

        Arguments:
            detector_type (str): The type of the detector e.g. input_disapprovals, internal_disapprovals, output_disapprovals
            detector_name (str): The concrete name of the detector as specified in the config.yaml e.g. Toxicbert_input
            value_key (str): Which key to extract from the detector.
            valid_key tuple[str, ...]s: Checks if the value_key is used right now. Defaults to ("approved, "latency", "class_name")

        Returns:
            A list of all extracted values.
        """
        values = []
        for res in self.results:
            # Check against dictionary keys rather than object attributes
            if detector_type not in res:
                raise ValueError(
                    f"'{detector_type}' is not a valid Detector Type in this result.\n"
                    f"Available options: {list(res.keys())}"
                )
            disapproval_dict = res.get(detector_type) or {}
            if detector_name not in disapproval_dict:
                raise ValueError(
                    f"Detector '{detector_name}' not found in '{detector_type}'."
                )
            detector_dict = disapproval_dict.get(detector_name) or {}
            value = detector_dict.get(value_key)
            values.append(value)
        return values
        
    def _extract_overall_disapprovals(
        self,
    ) -> list[bool | float]:
        """Extract form list[GenerationSafetyResult]"""
        values = []
        for res in self.results:
            values.append(res["overall_disapproval"])
        return values

       
    def _get_detector_names(self) -> dict[str, list[str]]:
        """Unique detector names per detector type, across all results."""
        names: dict[str, set[str]] = {dt: set() for dt in self.detector_types}
        for res in self.results:
            for detector_type in self.detector_types:
                disapproval_dict = res.get(detector_type, {})
                names[detector_type].update(disapproval_dict.keys())
        return {dt: sorted(ns) for dt, ns in names.items()}

    def _get_all_metrics(self, metric_func):
        """Aggregate function to generate all unique values for the given key.

        Arguments:
            metric_func (Callable): The basic function that gets a specified value.

        Returns:
            A dict with of dictionaries, whereas each inner dictionary represents a detector with its metrics.
        """
        metrics = defaultdict(dict)
        #each unique detector only needs to be iterated over once since metric_func calculates for entire dataset
        for detector_type, detector_names in self._get_detector_names().items():
            for detector_name in detector_names:
                metrics[detector_type][detector_name] = metric_func(detector_type, detector_name)
        return metrics

    def _calculate_classification_report(self, y_pred: list) -> dict:
        """Calculates classification metrics from raw predictions."""
        return classification_report(
            self.y_true, y_pred, zero_division=np.nan, output_dict=True
        )

    def _calculate_rate_metrics(self, cm: np.ndarray) -> dict[str, float]:
        """Calculates TPR, FNR, FPR, TNR, and Refusal rates from a confusion matrix."""
        tn, fp, fn, tp = cm.ravel()
        rates = {}
        pos = tp + fn if (tp + fn) > 0 else np.nan
        neg = fp + tn if (fp + tn) > 0 else np.nan
        total = fp + tn + tp + fn if (fp + tn + tp + fn) > 0 else np.nan
        rates["TPR"] = tp / pos
        rates["FNR"] = fn / pos
        rates["FPR"] = fp / neg
        rates["TNR"] = tn / neg
        rates["Refusal"] = (tp + fp) / total if total > 0 else np.nan
        return rates

    def _calculate_latency_metrics(self, latencies: list[float]) -> dict[str, float]:
        """Calculates key latency statistics from a list of durations."""
        latency_metrics = {}
        latency_metrics["mean"] = np.mean(latencies)
        latency_metrics["max"] = np.max(latencies)
        latency_metrics["min"] = np.min(latencies)
        latency_metrics["median"] = np.median(latencies)
        latency_metrics["p99"] = np.percentile(latencies, 99)
        latency_metrics["p95"] = np.percentile(latencies, 95)

        return latency_metrics

    def get_ideal_refusal(self) -> float:
        """Calculate the ideal refusal rate, which is the proportion of 
        ground truth violations relative to the total dataset size.
        """
        total_size = len(self.y_true)
        return sum(self.y_true) / total_size

    def get_confusion_matrix(
        self, detector_type: str, detector_name: str
    ) -> np.ndarray:
        """Generates a single confusion matrix for a specified detector.

        Arguments:
            detector_type (str): The type of the detector e.g. input_disapprovals, internal_disapprovals, output_disapprovals
            detector_name (str): The concrete name of the detector as specified in the config.yaml e.g. Toxicbert_input

        Returns:
            A confusion matrix with the labels in order: TN, FP, FN, TP
        """
        y_pred = self._extract_detector_values(detector_type, detector_name, "disapproved")
        return confusion_matrix(self.y_true, y_pred, labels=[False,True])

    def get_classification_report(self, detector_type: str, detector_name: str) -> dict:
        """Generates a single classification report for a specified detector.

        Arguments:
            detector_type (str): The type of the detector e.g. input_disapprovals, internal_disapprovals, output_disapprovals
            detector_name (str): The concrete name of the detector as specified in the config.yaml e.g. Toxicbert_input

        Returns:
            A dictionary with the precision, recall and f1-score.
        """
        y_pred = self._extract_detector_values(detector_type, detector_name, "disapproved")
        return self._calculate_classification_report(y_pred)

    def get_rate_metrics(
        self, detector_type: str, detector_name: str
    ) -> dict[str, float]:
        """Generates the metrics TPR, FNR, FPR, TNR, RefusalRate for a given detector."""
        cm = self.get_confusion_matrix(detector_type, detector_name)
        return self._calculate_rate_metrics(cm)

    def get_latency(self, detector_type: str, detector_name: str) -> dict[str, float]:
        """Generate a single list of latency metrics for a specified detector."""
        latencies = self._extract_detector_values(
            detector_type, detector_name, "latency"
        )
        return self._calculate_latency_metrics(latencies)


    def get_ensemble_confusion_matrix(self) -> np.ndarray:
        y_pred = self._extract_overall_disapprovals()
        return confusion_matrix(self.y_true, y_pred, labels = [False,True])

    def get_ensemble_classification_report(self) -> dict:
        y_pred = self._extract_overall_disapprovals()
        return self._calculate_classification_report(y_pred)

    def get_ensemble_rate_metrics(self) -> dict[str, float]:
        cm = self.get_ensemble_confusion_matrix()
        return self._calculate_rate_metrics(cm)

    def get_all_latency_metrics(self) -> dict[str, dict[str, float]]:
        """Generate all latency metrics for each unique detector in the results.

        Returns:
            A dictionary of detector dictionaries with the keys "mean", "max", "min", "median", "p95", "p99".
        """
        return self._get_all_metrics(self.get_latency)

    def get_all_confusion_matrices(self) -> dict[str, dict[str, np.ndarray]]:
        """Generate all confusion matrices for each unique detector in the results.

        Returns:
            A dictionary of detector dictionaries with a confusion matrix (labels in order: TN, FP, FN, TP)

        """
        return self._get_all_metrics(self.get_confusion_matrix)

    def get_all_classification_reports(self) -> dict[str, dict[str, dict]]:
        """Generate all classification reports for each unique detector in the results.

        Returns:
           A dictionary of detector dictionaries with the precision, recall and f1-score.

        """
        return self._get_all_metrics(self.get_classification_report)

    def get_all_rate_metrics(self) -> dict[str, dict[str, dict]]:
        """Generate rate metrics for each unique detector in the results.

        Returns:
            A dictionary of detector dictionaries with the keys "TPR", "FNR", "FPR", "TNR", "Refusal".
        """
        return self._get_all_metrics(self.get_rate_metrics)
    

    def _print_metrics_in_table(self, metrics: dict) -> None:
        """Print all the given metrics in a table format.

        Arguments:
            title (str): The title of the table.
            metrics (dict): The captured metrics to display.
        """

        disapproval_stages = list(iter(metrics))
        first_disapproval_stage = disapproval_stages[0]
        first_detector = next(iter(metrics[first_disapproval_stage]))
        metrics_headers = list(iter(metrics[first_disapproval_stage][first_detector]))
        headers = ["Detectors"] + metrics_headers

        for stage in disapproval_stages:
            print("=" * len(stage))
            print(stage.upper())
            print("=" * len(stage))

            table_metrics = []
            detectors = list(iter(metrics[stage]))

            for detector in detectors:
                values = metrics[stage][detector].values()
                row = [detector] + list(values)
                table_metrics.append(row)

            print(tabulate(table_metrics, headers, tablefmt="github"))

    def print_all_latency_metrics(self, latencies: dict) -> None:
        """Print the latency metrics for each disapproval stage and detector


        Arguments:
            latencies: Dictionary of the get_all_latency_metrics function.
        """
        self._print_metrics_in_table(latencies)

    def print_all_rates_metrics(self, rates: dict) -> None:
        """Print the rates metrics for each disapproval stage and detector


        Arguments:
            rates: Dictionary of the get_all_rate_metrics function.
        """
        self._print_metrics_in_table(rates)

    def print_all_classification_reports(self, metrics: dict) -> None:
        """ Print the classification reports for each disapproval stage and detector


        Arguments:
            metrics: Dictionary of the get_all_classification_reports function.
        """
        disapproval_stages = list(iter(metrics))
        headers = [
            "Class/Metric",
            "Precision",
            "Recall",
            "F1-Score",
            "Support",
        ]

        for stage in disapproval_stages:
            print("=" * len(stage))
            print(stage.upper())
            print("=" * len(stage))

            for detector, class_type in metrics[stage].items():
                print("-" * len(detector))
                print(detector.upper())
                print("-" * len(detector))
                table_metrics = []
                accuracy = None
                for class_label, class_values in class_type.items():
                    if isinstance(class_values, dict):
                        precision = class_values["precision"]
                        recall = class_values["recall"]
                        f1score = class_values["f1-score"]
                        support = class_values["support"]
                        row = [class_label, precision, recall, f1score, support]
                        table_metrics.append(row)
                    else:
                        accuracy = class_values

                print(tabulate(table_metrics, headers, tablefmt="github"))
                print(f"***** Overall Accuracy {accuracy} *****")

    def display_confusion_matrix(self, cm) -> None:
        """Displays a confusion matrix for the given arguments.

        Arguments:
            detector_type (str): The type of the detector e.g. input_disapprovals, internal_disapprovals, output_disapprovals
            detector_name (str): The concrete name of the detector as specified in the config.yaml e.g. Toxicbert_input

        """
        cm_display = ConfusionMatrixDisplay(cm)
        cm_display.plot()
        plt.show()
        
if __name__ == "__main__":
    cur_dir = Path(__file__).resolve()
    yaml_path = cur_dir.parent / "results.yaml"
    with open(yaml_path, "r") as f:
        results = yaml.load(f, Loader=yaml.UnsafeLoader)

    eval = SafetyEvaluator(results, [True, False],[True,False])
    #metrics = eval.get_all_classification_reports()
    #eval.print_all_classification_reports(metrics)
