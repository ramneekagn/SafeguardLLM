from pathlib import Path
from safeguard_llm.run_safe_llm import run_safe_llm
from safeguard_llm.utils.save_results import save_results_as_json
from transformers import AutoModelForCausalLM, AutoTokenizer
from safeguard_llm.safety_harness import SafeLLM
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datasets import load_dataset,concatenate_datasets, Dataset
from safeguard_llm.utils.output_judge import Benchmark_Eval
from dotenv import load_dotenv
import asyncio
import json 
from dataclasses import asdict

SEED = 40 

# HANDLE DATASETS
def _check_sample_size(sample_size: int, ds_true: Dataset, ds_false: Dataset) -> int:
    """ Checks if sample size will lead to an IndexError and in this case adjusts it with the maximum possible index """
    if sample_size < 1:
        sample_size = 1
        print(f"Chosen sample size was too small. Reverting to {sample_size}")
    max_safe_sample_size = min(len(ds_true), len(ds_false))
    if sample_size > max_safe_sample_size:
        sample_size = max_safe_sample_size
        print(f"Choosen sample size was too large for the given dataset. Reverting to {max_safe_sample_size}")
    return sample_size

def _check_single_sample_size(sample_size: int, ds: Dataset):
    if sample_size < 1:
        sample_size = 1
        print(f"Chosen sample size was too small. Reverting to {sample_size}")
    if sample_size > len(ds):
        sample_size = len(ds)
        print(f"Choosen sample size was too large for the given dataset. Reverting to {sample_size}")
    return sample_size

def _prepare_output_file(output_filename: str, dataset_name: str) -> Path:
    test_file = Path(output_filename)
    test_file = test_file.with_stem(f"{test_file.stem}_{dataset_name}")
    return test_file

def _get_standardized_split(source: str, size: int, seed: int) -> Dataset:
    """Loads, standardizes, and returns a dataset split with columns: 'prompt' and 'label'."""
    if source == "alpaca-cleaned":
            ds = load_dataset("yahma/alpaca-cleaned", split="train")
            ds = ds.shuffle(seed=seed)
            ds = ds.select(range(size))
            return ds.map(
                lambda elm: {
                    "prompt": elm["instruction"],
                    "label": 0,
                },
                remove_columns=ds.column_names,
        )
    if source == "wildchat-benign":
            ds = load_dataset("allenai/WildChat-nontoxic", split="train")
            ds = ds.shuffle(seed=seed)
            buffer_size = size * 15
            #select first otherwise filtering takes too long 
            ds_subset = ds.select(range(buffer_size))
            ds_filtered = ds_subset.filter(lambda entry: entry["language"] == "English")
            ds = ds_filtered.select(range(size))
            return ds.map(
                lambda elm: {
                    "prompt": elm["conversation"][0]["content"],
                    "label": 0,
                },
                remove_columns=ds.column_names,
        )
    elif source == "jbb-harmful":
        ds = load_dataset(
            "JailbreakBench/JBB-Behaviors", "behaviors", split="harmful"
        )
        ds = ds.shuffle(seed=seed)
        size = _check_single_sample_size(size, ds)
        ds = ds.select(range(size))
        return ds.map(
            lambda elm: {"prompt": elm["Goal"], "label": 1},
            remove_columns=ds.column_names,
        )

    elif source == "jbb-benign":
        ds = load_dataset(
            "JailbreakBench/JBB-Behaviors", "behaviors", split="benign"
        )
        ds = ds.shuffle(seed=seed)
        size = _check_single_sample_size(size, ds)
        ds = ds.select(range(size))
        return ds.map(
            lambda elm: {"prompt": elm["Goal"], "label": 0},
            remove_columns=ds.column_names,
        )

    elif source == "xstest-benign":
        ds = load_dataset("walledai/XSTest", split="test")
        ds = ds.filter(lambda elm: elm["label"] == "safe")
        ds = ds.shuffle(seed=seed)
        size = _check_single_sample_size(size, ds)
        ds = ds.select(range(size))
        return ds.map(
            lambda elm: {"prompt": elm["prompt"], "label": 0},
            remove_columns=ds.column_names,
        )

    elif source == "xstest-harmful":
        ds = load_dataset("walledai/XSTest", split="test")
        ds = ds.filter(lambda elm: elm["label"] == "unsafe")
        ds = ds.shuffle(seed=seed)
        size = _check_single_sample_size(size, ds)
        ds = ds.select(range(size))
        return ds.map(
            lambda elm: {"prompt": elm["prompt"], "label": 1},
            remove_columns=ds.column_names,
        )

    else:
        raise ValueError(f"Unknown data source: {source}")
    
def prepare_dataset(
    dataset_name: str, sample_size: int = 200, seed: int = SEED
) -> Dataset:
    """Returns a Hugging Face dataset with standardized columns: 'prompt' and 'label'"""
    #high fp test 
    if dataset_name == "50_50_xstest":
        benign = _get_standardized_split("xstest-benign", sample_size, seed)
        harmful = _get_standardized_split("xstest-harmful", sample_size, seed)
        return concatenate_datasets([benign, harmful])

    elif dataset_name == "50_50_hard":
        benign = _get_standardized_split("jbb-benign", sample_size, seed)
        harmful = _get_standardized_split("jbb-harmful", sample_size, seed)
        return concatenate_datasets([benign, harmful])
    #100 harmless
    elif dataset_name == "0_100":
        return _get_standardized_split("alpaca-cleaned", sample_size, seed)

    #100 harmful
    elif dataset_name == "100_0":
        return _get_standardized_split("jbb-harmful", sample_size, seed)

    #easy
    elif dataset_name == "50_50_easy":
        benign = _get_standardized_split("alpaca-cleaned", sample_size, seed)
        harmful = _get_standardized_split("jbb-harmful", sample_size, seed)
        return concatenate_datasets([benign, harmful])

    #realistic
    elif dataset_name == "1_99":
        target_size = max(sample_size, 1000)
        size_benign = int(target_size * 0.99)
        size_harmful = int(target_size * 0.01)

        benign = _get_standardized_split("wildchat-benign", size_benign, seed)
        harmful = _get_standardized_split("jbb-harmful", size_harmful, seed)
        return concatenate_datasets([benign, harmful])

    else:
        raise ValueError(f"Dataset {dataset_name} not implemented.")
    
# EXECUTION

def run_and_judge(
        dataset_name: str,
        output_filepath: Path, 
        sample_size: int,
        batch_size: int,
        willJudge: bool,
        max_gen_len: int = 256,
        config_path: Path = Path("src/safeguard_llm/config/safe_llm_config.yaml"),
        model_name: str = "Qwen/Qwen3-1.7B"
):
    """ Runs the evaluation pipeline and optionally adds gold labels for the output by a LLM judge """
    # prepare ds
    print(f"Running {dataset_name}")
    dataset = prepare_dataset(dataset_name, sample_size=sample_size)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, padding_side="left"
    )

    safe_model = SafeLLM(
        model,
        tokenizer,
        max_gen_len=max_gen_len,
        config_path=config_path
    )

    results = []
    gold_labels = dataset["label"]

    # inference
    for batch in tqdm(dataloader):
        inputs = batch["prompt"]
        output = safe_model.generate(inputs)
        results.extend(output)

    # process gold labels 
    for i, res in enumerate(results):
        val = gold_labels[i]
        val = int(val)
        res.prompt_label_gold = bool(val)
    serialized_results = [asdict(res) for res in results]
    test_file = _prepare_output_file(output_filepath, dataset_name)
    # Write initial raw output
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(serialized_results, f, indent=4)

    # LLM judge
    if willJudge:
        print(f"Judging {dataset_name}")
        be = Benchmark_Eval(output_filepath)
        asyncio.run(be.execute_judgement())
        
        judged_filepath = output_filepath.with_stem(f"{output_filepath.stem}_judged")
        with open(judged_filepath, "r", encoding="utf-8") as f:
            results = json.load(f)
            
    return results

if __name__ == "__main__":
    load_dotenv()
    tests = ["50_50_easy", "50_50_hard", "50_50_xstest", "100_0", "0_100", "1_99"]
    for test in tests:
        print(test)
        dataset = prepare_dataset(test, 1)
        print(dataset)
