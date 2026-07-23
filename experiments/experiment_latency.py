import json
from safeguard_llm.evaluator import SafetyEvaluator

file_path = "results/safe_llm_config_base_v2_input_v1_output_lp_internal_0_5t_all_experiment2_0_5t_all/100_0_advbench_harmful/judged_and_rule.json"

with open(file_path, "r") as f:
    results = json.load(f)

evaluator = SafetyEvaluator(results)
latency_metrics = evaluator.get_all_latency_metrics()

names = []
means = []

for stage, detectors in latency_metrics.items():
    print(f"\n{stage}:")
for stage, detectors in latency_metrics.items():
    print(f"\n{stage}:")
    for name, metrics in detectors.items():
        mean_val = metrics["mean"]
        max_val = metrics["max"]
        min_val = metrics["min"]
        median_val = metrics["median"]
        print(f"  {name}: Mean={mean_val} | Median={median_val} | Min={min_val} | Max={max_val}")        
        names.append(name)
        means.append(mean_val)

