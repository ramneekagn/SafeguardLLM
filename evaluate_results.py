import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Callable
from dotenv import load_dotenv
from tabulate import tabulate
from sklearn.metrics import ConfusionMatrixDisplay

# Import your updated, dictionary-safe SafetyEvaluator
from safeguard_llm.evaluator import SafetyEvaluator

# ==========================================
# 1. Reclassification Pipeline & Strategies
# ==========================================
def reclassify(results: list[dict], rule_func: Callable[[dict], bool]) -> list[dict]:
    """Iterates through the results list and updates overall_disapproval in-place."""
    for entry in results:
        entry["overall_disapproval"] = rule_func(entry)
    return results

# Strategy 1: Implication Rule (Your custom logic)
def implication_rule(entry: dict) -> bool:
    input_bert = entry["input_disapprovals"]["InputRobertaJBDetector"]["disapproved"]
    input_lp = entry["internal_disapprovals"]["LPInternalDetector1"]["disapproved"]
    conditional_lp = entry["internal_disapprovals"]["LPInternalDetectorConditonal"]["disapproved"]
    output_bert = entry["output_disapprovals"]["OutputRobertaJBDetector"]["disapproved"]
    is_input_harmful = input_bert or input_lp
    implication = (not is_input_harmful) or conditional_lp
    return implication and output_bert

# Strategy 2: Strict OR (Conservative - blocks if anything triggers)
def strict_or_rule(entry: dict) -> bool:
    input_bert = entry["input_disapprovals"]["InputRobertaJBDetector"]["disapproved"]
    input_lp = entry["internal_disapprovals"]["LPInternalDetector1"]["disapproved"]
    conditional_lp = entry["internal_disapprovals"]["LPInternalDetectorConditonal"]["disapproved"]
    output_bert = entry["output_disapprovals"]["OutputRobertaJBDetector"]["disapproved"]
    return input_bert or input_lp or conditional_lp or output_bert

# Strategy 3: Standard AND (Blocks only if input AND output are flagged)
def standard_and_rule(entry: dict) -> bool:
    input_bert = entry["input_disapprovals"]["InputRobertaJBDetector"]["disapproved"]
    input_lp = entry["internal_disapprovals"]["LPInternalDetector1"]["disapproved"]
    output_bert = entry["output_disapprovals"]["OutputRobertaJBDetector"]["disapproved"]
    return (input_bert or input_lp) and output_bert

# Strategy 4: Output Only (Trusts the output safety filter completely)
def output_only_rule(entry: dict) -> bool:
    return entry["output_disapprovals"]["OutputRobertaJBDetector"]["disapproved"]

# Strategy 5: Input Only (Blocks based strictly on input prompts)
def input_only_rule(entry: dict) -> bool:
    input_bert = entry["input_disapprovals"]["InputRobertaJBDetector"]["disapproved"]
    input_lp = entry["internal_disapprovals"]["LPInternalDetector1"]["disapproved"]
    return input_bert or input_lp


# ==========================================
# 2. Helpers for Metric Extraction
# ==========================================
def extract_binary_classification_metrics(report: dict) -> tuple[float, float, float]:
    """
    Safely extracts Precision, Recall, and F1-Score for the positive/harmful class.
    Includes robust fallbacks for all combinations of boolean, string, or integer keys.
    """
    # 1. Search common keys representing the positive class
    pos_keys = ["True", "true", "1", True, 1]
    for key in pos_keys:
        if key in report:
            metrics_dict = report[key]
            return (
                metrics_dict.get("precision", 0.0),
                metrics_dict.get("recall", 0.0),
                metrics_dict.get("f1-score", 0.0),
            )
            
    # 2. Hard fallback: Extract from the first non-average key in report
    for key, val in report.items():
        if key not in ["accuracy", "macro avg", "weighted avg"] and isinstance(val, dict):
            return (
                val.get("precision", 0.0),
                val.get("recall", 0.0),
                val.get("f1-score", 0.0),
            )
            
    return 0.0, 0.0, 0.0


def format_percentage(value: float) -> str:
    """Formats numeric values safely as percentages, handling NaN and other exceptions."""
    try:
        if value is None or np.isnan(value):
            return "N/A"
        return f"{value:.1%}"
    except Exception:
        return "N/A"


# ==========================================
# 3. Main Execution Loop
# ==========================================
if __name__ == "__main__":
    load_dotenv()
    
    # Define paths
    json_path = Path("results_jb_experiment_1_judged.json")
    out_path = Path("results_jb_reclassified.json")

    if not json_path.exists():
        raise FileNotFoundError(f"Missing evaluation file: {json_path}")

    # Load evaluated results
    with open(json_path, "r") as f:
        original_results = json.load(f)

    # Define all comparison strategies
    strategies = {
        "Implication Rule": implication_rule,
        "Strict OR (Block All)": strict_or_rule,
        "Standard AND": standard_and_rule,
        "Output Only": output_only_rule,
        "Input Only": input_only_rule,
    }

    # Define the 3 different ground truth perspectives
    ground_truth_perspectives = {
        "Input Ground Truth (Is Prompt Malicious?)": lambda x, y: x,
        "Output Ground Truth (Is Response Harmful?)": lambda x, y: y,
        "Joint Policy (Is Prompt Malicious OR Response Harmful?)": lambda x, y: x or y
    }

    headers = [
        "Strategy Name", 
        "Accuracy (Acc)", 
        "F1-Score (Harmful)", 
        "Precision (Pr)", 
        "Recall (Rec)", 
        "False Pos Rate (FPR)", 
        "False Neg Rate (FNR)", 
        "Refusal Rate"
    ]

    # Create a 3x5 layout grid of subplots (3 rows of perspectives x 5 columns of strategies)
    fig, axes = plt.subplots(
        len(ground_truth_perspectives), 
        len(strategies), 
        figsize=(4.2 * len(strategies), 4.0 * len(ground_truth_perspectives))
    )
    
    # Standardize axes to 2D shape for consistent indexing
    if len(ground_truth_perspectives) == 1 and len(strategies) == 1:
        axes = np.array([[axes]])
    elif len(ground_truth_perspectives) == 1:
        axes = axes[np.newaxis, :]
    elif len(strategies) == 1:
        axes = axes[:, np.newaxis]

    print("Executing evaluations across multiple ground truth targets...")
    print("=" * 110)

    # Run and plot each perspective-strategy pair
    for p_idx, (gt_name, gt_rule) in enumerate(ground_truth_perspectives.items()):
        print(f"\n>>> TARGET GROUND TRUTH: {gt_name.upper()}")
        print("-" * 110)
        
        comparison_table_rows = []

        for s_idx, (name, rule_func) in enumerate(strategies.items()):
            # Copy original data to avoid cross-strategy pollution
            temp_results = json.loads(json.dumps(original_results))
            
            # Reclassify
            reclassified = reclassify(temp_results, rule_func)
            
            # Evaluate against current ground truth perspective
            evaluator = SafetyEvaluator(reclassified, truth_rule=gt_rule)
            
            # Fetch raw metrics
            cm = evaluator.get_ensemble_confusion_matrix()
            report = evaluator.get_ensemble_classification_report()
            rates = evaluator.get_ensemble_rate_metrics()
            
            # Extract classification metrics safely
            accuracy = report.get("accuracy", 0.0)
            precision, recall, f1_score = extract_binary_classification_metrics(report)
            
            # Extract rate metrics
            fpr = rates.get("FPR", 0.0)
            fnr = rates.get("FNR", 0.0)
            refusal_rate = rates.get("Refusal", 0.0)
            
            # Append rows
            comparison_table_rows.append([
                name,
                format_percentage(accuracy),
                format_percentage(f1_score),
                format_percentage(precision),
                format_percentage(recall),
                format_percentage(fpr),
                format_percentage(fnr),
                format_percentage(refusal_rate)
            ])

            # Draw confusion matrix on its specific row/column axis subplot
            ax = axes[p_idx, s_idx]
            display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])
            display.plot(ax=ax, cmap="Blues", colorbar=False)
            
            # Cleanly label subplots to avoid clutter
            ax.set_title(f"{name}\n({gt_name.split()[0]} GT)", fontsize=8, fontweight="bold")

        # Print table for this perspective
        print(tabulate(comparison_table_rows, headers, tablefmt="github"))
        print("=" * 110)
    # Render confusion matrix visualizations
    plt.tight_layout()
    plt.show()