import json
from safeguard_llm.evaluator import SafetyEvaluator
import matplotlib.pyplot as plt
file_paths = ["results/safe_llm_config_base_v2_input_v1_output_lp_internal_0_5t_all_experiment2_0_5t_all/0_100_alpaca_cleaned/judged_and_rule.json",
              "results/safe_llm_config_base_v2_input_v1_output_lp_internal_0_5t_all_experiment2_0_5t_all/0_100_xstest_benign/judged_and_rule.json"]

orr_all_values = []

for path in file_paths: 
    with open(path, "r") as f:
        data = json.load(f)

    orr_results = [res for res in data if res.get("prompt_label_gold") == False and res.get("output_label_gold") == False]
    orr_results = orr_results[:250]
    
    raw_count = len(data)
    filtered_count = len(orr_results)
    pct = (filtered_count / raw_count * 100) if raw_count > 0 else 0.0

    evaluator = SafetyEvaluator(orr_results, truth_rule=lambda x, y: x or y)
    individual_metrics = evaluator.get_all_rate_metrics()
    ensemble_orr = evaluator.get_ensemble_rate_metrics()["FPR"]

    print(f"Filtered {filtered_count} safe-safe pairs out of {raw_count} total ({pct:.1f}%)")

    values = []
    for stage, detectors in individual_metrics.items():
        for name, metrics in detectors.items():
            values.append(metrics["FPR"] * 100)
    values.append(ensemble_orr * 100)
    orr_all_values.append(values)
    
names = ["Input roBERTa", "Linear Probe", "Output roBERTa", "Ensemble"]
colors = ["#FFC5C5", "#FFE58F", "#82C0FF", "#FA8072"]
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.family'] = 'serif'

max_y_orr = max(max(orr_all_values[0]), max(orr_all_values[1])) * 1.15

plt.figure(figsize=(18, 7))

for i in range(2):
    plt.subplot(1, 2, i + 1)
    bars = plt.bar(names, orr_all_values[i], color=colors, linewidth=0.8)
    
    if i == 0:
        plt.ylabel("Overrefusal Rate (%)", fontsize=12)
        
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylim(0, max_y_orr)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0, 
            height + 1.0, 
            f"{height:.1f}%", 
            ha="center", 
            va="bottom",
            fontsize=18,
            fontweight="bold"
        )

plt.tight_layout()
plt.show()