# %%
from safeguard_llm.evaluator import SafetyEvaluator
from safeguard_llm.run_safe_llm import run_safe_llm

outputs, labels = run_safe_llm(
    dataset_name="deepset/prompt-injections", 
    dataset_size=10, 
    prompt_col="text", 
    label_col="label", 
    subset="train"
)

evaluator = SafetyEvaluator(outputs, labels)
metrics = evaluator.get_all_classification_reports()
evaluator.print_all_classification_reports(metrics)
latency = evaluator.get_overall()
print(latency)

outputs,labels = run_safe_llm(dataset_name="SetFit/toxic_conversations", dataset_size=1000, prompt_col ="text", label_col ="label",subset="test")

evaluator = SafetyEvaluator(outputs, labels)
metrics = evaluator.get_all_classification_reports()
evaluator.print_all_classification_reports(metrics)

print(outputs)
evaluator.display_confusion_matrix(detector_type="internal_approvals",detector_name="LPInternalDetector1")


