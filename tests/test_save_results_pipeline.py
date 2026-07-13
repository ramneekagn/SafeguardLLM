from pathlib import Path
from safeguard_llm.evaluator import SafetyEvaluator
from safeguard_llm.run_safe_llm import run_safe_llm
from safeguard_llm.utils.save_results import save_results_as_json

results, labels = run_safe_llm(
    dataset_name="deepset/prompt-injections", 
    dataset_size=10, 
    prompt_col="text", 
    label_col="label", 
    subset="train"
)

evaluator = SafetyEvaluator(results, labels)
test_file = Path("testv1_a2.json")
save_results_as_json(results, test_file)
