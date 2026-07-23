import json
import matplotlib.pyplot as plt
from safeguard_llm.evaluator import SafetyEvaluator

file_paths = ["results/safe_llm_config_base_v2_input_v1_output_lp_internal_0_5t_all_experiment2_0_5t_all/100_0_advbench_harmful/judged_and_rule.json",
              "results/safe_llm_config_base_v2_input_v1_output_lp_internal_0_5t_all_experiment2_0_5t_all/100_0_allen_wild_jb/judged_and_rule.json"]

all_values = []
all_titles = ["AdvBench", "WildJBB Adversarial"]

for path in file_paths: 
    with open(path, "r") as f:
        data = json.load(f)

    asr_results = [res for res in data if res.get("prompt_label_gold") == True and res.get("output_label_gold") == True]

    raw_count = len(data)
    filtered_count = len(asr_results)
    pct = (filtered_count / raw_count * 100)

    evaluator = SafetyEvaluator(asr_results, truth_rule=lambda x, y: x and y)
    individual_metrics = evaluator.get_all_rate_metrics()
    ensemble_asr = evaluator.get_ensemble_rate_metrics()["FNR"]
    
    print(f"Filtered {filtered_count} successful attack pairs out of {raw_count} total ({pct:.1f}%)")

    values = []
    for stage, detectors in individual_metrics.items():
        for name, metrics in detectors.items():
            values.append(metrics["FNR"] * 100)
    values.append(ensemble_asr * 100)
    all_values.append(values)

max_y = max(max(all_values[0]), max(all_values[1])) * 1.15

plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.family'] = 'serif'

names = ["Input roBERTa", "Linear Probe", "Output roBERTa", "Ensemble"]
colors = ["#FFC5C5", "#FFE58F", "#82C0FF", "#FA8072"]

plt.figure(figsize=(16, 7))

for i in range(2):
    plt.subplot(1, 2, i + 1)
    bars = plt.bar(names, all_values[i], color=colors, linewidth=0.8)
    
    if i == 0:
        plt.ylabel("Attack Success Rate (%)", fontsize=12)
        
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylim(0, max_y)

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