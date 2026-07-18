import json
from pathlib import Path
from run_and_judge import run_and_judge, _prepare_output_file
from reclassify_overall import reclassify, and_rule, or_rule, majority_vote
from run_evaluation_reworked import log_eval
from dotenv import load_dotenv
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

def run_pipeline(dataset_name, config_path, run_name, base_output_dir="results", config_name="safe_llm_config_base", sample_size=100, batch_size=8):
    name = f"{config_name}_{run_name}"
    dataset_dir = Path(base_output_dir) / name / dataset_name
    if dataset_dir.exists():
        raise FileExistsError(
            f"Target directory already exists: '{dataset_dir}'. "
        )
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    raw_filepath = dataset_dir / "raw.json"

    results = run_and_judge(
        dataset_name=dataset_name,
        output_filepath=raw_filepath,  
        sample_size=sample_size,
        batch_size=batch_size,
        config_path=config_path,
        willJudge=True
    )
    
    rules = [and_rule, or_rule, majority_vote]
    for rule in rules: 
        ensemble_path = dataset_dir / f"judged_{rule.__name__}.json"
        reclassified_results = reclassify(results, rule)
        
        print(f"Saving reclassified results to: {ensemble_path}")
        with open(ensemble_path, "w", encoding="utf-8") as f:
            json.dump(reclassified_results, f, indent=4)
        log_eval(rule.__name__, sample_size, ensemble_path, log_dir=dataset_dir)


load_dotenv()
run_name = "full_run3"
config_file_name = "safe_llm_config_mmbert32k"
config_path = Path(f"src/safeguard_llm/config/{config_file_name}.yaml")
#dataset_names = ["50_50_easy_jbb-harmful_alpaca_cleaned", "50_50_hard_jbb-benign_jbb-harmful",  "100_0_jbb_harmful", "0_100_alpaca_cleaned"]
dataset_names = ["50_50_easy_jbb-harmful_alpaca_cleaned"]
for dataset_name in dataset_names:
    run_pipeline(
        dataset_name=dataset_name,
        config_path = config_path,
        config_name=config_file_name,
        run_name = run_name,
        sample_size=100,
        batch_size=8
    )
