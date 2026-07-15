import asyncio
from pathlib import Path
from tqdm.auto import tqdm
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from safeguard_llm.utils.save_results import save_results_as_json
from safeguard_llm.safety_harness import SafeLLM
from safeguard_llm.utils.output_judge import Benchmark_Eval

# HANDLE DATASETS
def _check_sample_size(sample_size: int, ds_true: Dataset, ds_false: Dataset) -> int:
    """ Checks if sample size will lead to an IndexError and in this case adjusts it with the maximum possible index """

    max_safe_sample_size = min(len(ds_true), len(ds_false))
    if sample_size > max_safe_sample_size:
        sample_size = max_safe_sample_size
        print(f"Choosen sample size was too large for the given dataset. Reverting to {max_safe_sample_size}")
    return sample_size

def _prepare_output_file(output_filename: str, dataset_name: str) -> Path:
    test_file = Path(output_filename)
    test_file = test_file.with_stem(f"{test_file.stem}_{dataset_name}")
    return test_file


def prepare_dataset(dataset_name: str, sample_size: int = 210):
    """ Returns a Hugging Face dataset with standardized columns: 'prompt' and 'label' """

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
        load_dotenv()
        be = Benchmark_Eval(test_file)
        asyncio.run(be.execute_judgement())

# RUN

if __name__ == "__main__":

    # === CONFIG ===
    dataset_name = "wildjailbreak"
    output_filename = "results_jb_first_run.json"
    sample_size = 210
    batch_size = 16
    willJudge = True

    """
    dataset_name = "jailbreak-classification"
    output_filename = "results_jb_first_run.json"
    sample_size = 100
    batch_size = 4
    willJudge = True
    """

    run_and_judge(
        dataset_name = dataset_name,
        output_filename = output_filename,
        sample_size = sample_size,
        batch_size = batch_size,
        willJudge = willJudge
    )