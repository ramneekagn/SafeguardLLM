from pathlib import Path
from safeguard_llm.run_safe_llm import run_safe_llm
from safeguard_llm.utils.save_results import save_results_as_json
from transformers import AutoModelForCausalLM, AutoTokenizer
from safeguard_llm.safety_harness import SafeLLM
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from pathlib import Path
from datasets import load_dataset,concatenate_datasets, Dataset
from safeguard_llm.utils.output_judge import Benchmark_Eval
from dotenv import load_dotenv
import asyncio
import json 
from safeguard_llm.evaluator import SafetyEvaluator


# HANDLE DATASETS
def _check_sample_size(sample_size: int, ds_true: Dataset, ds_false: Dataset) -> int:
    """ Checks if sample size will lead to an IndexError and in this case adjusts it with the maximum possible index """

    max_safe_sample_size = min(len(ds_true), len(ds_false))
    if sample_size > max_safe_sample_size:
        sample_size = max_safe_sample_size
        print(f"Choosen sample size was too large for the given dataset. Reverting to {max_safe_sample_size}")
    return sample_size

def _check_single_sample_size(sample_size: int, ds: Dataset):
    if sample_size > len(ds):
        sample_size = len(ds)
        print(f"Choosen sample size was too large for the given dataset. Reverting to {sample_size}")
    return sample_size

def _prepare_output_file(output_filename: str, dataset_name: str) -> Path:
    test_file = Path(output_filename)
    test_file = test_file.with_stem(f"{test_file.stem}_{dataset_name}")
    return test_file


def prepare_dataset(dataset_name: str, sample_size: int = 210):
    """ Returns a Hugging Face dataset with standardized columns: 'prompt' and 'label' """

    # === BALANCED DATASETS ===
    if dataset_name == "wildjailbreak":
        dataset = load_dataset("allenai/wildjailbreak", "eval", delimiter="\t", keep_default_na=False)
        ds_true = dataset["train"].filter(lambda elm: elm["label"] == 1)
        ds_false = dataset["train"].filter(lambda elm: elm["label"] == 0)

        sample_size = _check_sample_size(sample_size, ds_true, ds_false)
        ds_true = ds_true.select(range(sample_size))
        ds_false = ds_false.select(range(sample_size))
        dataset = concatenate_datasets([ds_true, ds_false])

        dataset = dataset.map(lambda elm: {
            "prompt": elm["adversarial"],
            "label": elm["label"]
        })
    if dataset_name == "wildjailbreak-vanilla":
            dataset = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False)
            ds_true = dataset["train"].filter(lambda elm: elm["data_type"] == "vanilla_harmful")
            ds_false = dataset["train"].filter(lambda elm: elm["data_type"] == "vanilla_benign")

            sample_size = _check_sample_size(sample_size, ds_true, ds_false)
            ds_true = ds_true.select(range(sample_size))
            ds_false = ds_false.select(range(sample_size))
            dataset = concatenate_datasets([ds_true, ds_false])

            dataset = dataset.map(lambda elm: {
                "prompt": elm["vanilla"],
                "label": 1 if elm["data_type"] == "vanilla_harmful" else 0
            })

    elif dataset_name == "preliminary_dataset": 
            ds_true = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
            ds_true = ds_true.map(
                lambda elm: {
                    "prompt": elm["Goal"],
                    "label": 1
                },
                remove_columns=ds_true.column_names
            )

            ds_false = load_dataset("tatsu-lab/alpaca", split="train")
            sample_size = _check_single_sample_size(sample_size, ds_false)
            ds_false = ds_false.select(range(sample_size))
            ds_false = ds_false.map(lambda elm: {
                "prompt": elm["instruction"],
                "label": 0
            }, remove_columns=ds_false.column_names)

            sample_size = _check_sample_size(sample_size, ds_true, ds_false)
            ds_true = ds_true.select(range(sample_size))
            ds_false = ds_false.select(range(sample_size))
            dataset = concatenate_datasets([ds_true, ds_false])


    elif dataset_name == "jailbreak-classification":
        dataset = load_dataset("jackhhao/jailbreak-classification", "default", delimiter="\t", keep_default_na=False)

        ds_true = dataset["test"].filter(lambda elm: elm["type"] == "jailbreak")
        ds_false = dataset["test"].filter(lambda elm: elm["type"] == "benign")

        sample_size = _check_sample_size(sample_size, ds_true, ds_false)

        ds_true = ds_true.select(range(sample_size))
        ds_false = ds_false.select(range(sample_size))

        dataset = concatenate_datasets([ds_true, ds_false])
        # dunno how the dataset looks like, just added prompt exolicitly to catch errors early
        dataset = dataset.map(lambda elm: {
            "prompt": elm["prompt"],
            "label": 1 if elm["type"] == "jailbreak" else 0})

    # === UNBALANCED DATASETS ===
    elif dataset_name == "jailbreakbench-benign":
        dataset = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="benign")
        sample_size = _check_single_sample_size(sample_size, dataset)
        dataset = dataset.select(range(sample_size))
        dataset = dataset.map(lambda elm: {
            "prompt": elm["Goal"],
            "label": 0
        }, remove_columns=dataset.column_names)


    elif dataset_name == "jailbreakbench-harmful":
        dataset = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
        sample_size = _check_single_sample_size(sample_size, dataset)
        dataset = dataset.select(range(sample_size))
        dataset = dataset.map(lambda elm: {
            "prompt": elm["Goal"],
            "label": 1
        }, remove_columns=dataset.column_names)


    elif dataset_name == "xstest-safe":
        dataset = load_dataset("walledai/XSTest", split="test")
        dataset = dataset.filter(lambda elm: elm["label"] == "safe")
        sample_size = _check_single_sample_size(sample_size, dataset)
        dataset = dataset.select(range(sample_size))
        dataset = dataset.map(lambda elm: {
            "prompt": elm["prompt"],
            "label": 0
        }, remove_columns=dataset.column_names)

    elif dataset_name == "xstest-unsafe":
        dataset = load_dataset("walledai/XSTest", split="test")
        dataset = dataset.filter(lambda elm: elm["label"] == "unsafe")
        sample_size = _check_single_sample_size(sample_size, dataset)
        dataset = dataset.select(range(sample_size))
        dataset = dataset.map(lambda elm: {
            "prompt": elm["prompt"],
            "label": 1
        }, remove_columns=dataset.column_names)

    elif dataset_name == "coconot":
        dataset = load_dataset("allenai/coconot", "contrast", split="test")
        sample_size = _check_single_sample_size(sample_size, dataset)
        dataset = dataset.select(range(sample_size))
        dataset = dataset.map(lambda elm: {
            "prompt": elm["prompt"],
            "label": 0
        }, remove_columns=dataset.column_names)


    elif dataset_name == "alpaca":
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
        sample_size = _check_single_sample_size(sample_size, dataset)
        dataset = dataset.select(range(sample_size))
        dataset = dataset.map(lambda elm: {
            "prompt": elm["instruction"],
            "label": 0
        }, remove_columns=dataset.column_names)

    else:
        raise ValueError(f"Dataset {dataset_name} not implemented.")

    return dataset

# EXECUTION

def run_and_judge(
        dataset_name: str,
        output_filename: str,
        sample_size: int, # mayeb default ?
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

    # init model + tokenizer
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

    outputs = []
    all_labels = []

    # inference
    for batch in tqdm(dataloader):
        inputs = batch["prompt"]
        labels = batch["label"]
        all_labels.extend(labels)
        output = safe_model.generate(inputs)
        outputs.extend(output)

    # process gold labels ?
    for i, res in enumerate(outputs):
        val = all_labels[i]
        val = int(val)
        if val == 0:
            res.output_label_gold = False
        res.prompt_label_gold = bool(val)

    test_file = _prepare_output_file(output_filename, dataset_name)
    save_results_as_json(outputs, test_file)

    # llm judge
    if willJudge:
        print(f"Judging {dataset_name}")
        be = Benchmark_Eval(test_file)
        asyncio.run(be.execute_judgement())
