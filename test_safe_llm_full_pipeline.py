# %%
from safeguard_llm.evaluator import SafetyEvaluator
from safeguard_llm.run_safe_llm import run_safe_llm

outputs,labels = run_safe_llm(dataset_name="textdetox/multilingual_toxicity_dataset", dataset_size=100, prompt_col ="text", label_col ="toxic",subset="en")


evaluator = SafetyEvaluator(outputs, labels)
metrics = evaluator.get_all_classification_reports()
evaluator.print_all_classification_reports(metrics)

evaluator.display_confusion_matrix(detector_type="internal_disapprovals",detector_name="LPInternalDetector1")
outputs,labels = run_safe_llm(dataset_name="SetFit/toxic_conversations", dataset_size=100, prompt_col ="text", label_col ="label",subset="test")

evaluator = SafetyEvaluator(outputs, labels)
metrics = evaluator.get_all_classification_reports()
evaluator.print_all_classification_reports(metrics)

print(outputs)
evaluator.display_confusion_matrix(detector_type="internal_approvals",detector_name="LPInternalDetector1")


