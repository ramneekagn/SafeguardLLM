import json
import numpy as np
import torch
from datasets import load_dataset, Dataset, concatenate_datasets
from dotenv import load_dotenv
from scipy.special import softmax
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, \
    average_precision_score, roc_auc_score, roc_curve
from transformers import TrainingArguments, AutoModelForSequenceClassification, Trainer, AutoTokenizer, \
    DataCollatorWithPadding
from pathlib import Path
from peft import PeftModel

# === CONFIG ===

BERTS = [
    "HanseeVee/roberta-base-input-jailbreak-classifier",
    "llm-semantic-router/mmbert32k-jailbreak-detector-merged",
    "llm-semantic-router/mmbert-safety-classifier-level1",
    "meta-llama/Llama-Prompt-Guard-2-86M",
    "meta-llama/Llama-Prompt-Guard-2-22M",
    "pmking27/jailbreak-detection",
    "hjackhhao/jailbreak-classifier",
    "lordofthejars/jailbreak-classifier"
]

DATASETS = ["5050_wild", "5050_both", "xs-test", "0199"]
SEED = 40

OUTPUT_PATH = Path("../scripts/results")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
load_dotenv()

# === FUNCTIONS ===

def prepare_dataset(dataset: str, size: int=250, seed: int=SEED) -> Dataset:
    """ Returns the dataset with the keys 'prompt' and 'label'

    Args:
        dataset: String name for the prepare_dataset if check
        size: How many samples. Defaults to 250. (Has no sanity check yet)
        seed: The seed for shuffling. Script initializes with SEED=40 as default.

    Returns:
        The named labelset with 'prompt' and 'label' as keys.
    """
    if dataset == "alpaca-cleaned":
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
    elif dataset == "xstest-benign":
        ds = load_dataset("walledai/XSTest", split="test")
        ds = ds.filter(lambda elm: elm["label"] == "safe")
        ds = ds.shuffle(seed=seed)
        ds = ds.select(range(size))
        return ds.map(
            lambda elm: {"prompt": elm["prompt"], "label": 0},
            remove_columns=ds.column_names,
        )

    elif dataset == "advbench":
        ds = load_dataset("walledai/AdvBench", split="train")
        ds = ds.shuffle(seed=seed)
        ds = ds.select(range(size))
        return ds.map(
            lambda elm: {"prompt": elm["prompt"], "label": 1},
            remove_columns=ds.column_names,
        )
    elif dataset == "wildjailbreak-adv":
        ds = load_dataset("walledai/WildJailbreak", split="train")
        ds = ds.filter(lambda elm: elm["label"] == "adversarial_harmful")
        ds = ds.shuffle(seed=seed)
        ds = ds.select(range(size))
        return ds.map(
            lambda elm: {"prompt": elm["prompt"], "label": 1},
            remove_columns=ds.column_names,
        )
    elif dataset == "wildjailbreak-benign":
        ds = load_dataset("walledai/WildJailbreak", split="train")
        ds = ds.filter(lambda elm: elm["label"] == "adversarial_benign")
        ds = ds.shuffle(seed=seed)
        ds = ds.select(range(size))
        return ds.map(
            lambda elm: {"prompt": elm["prompt"], "label": 0},
            remove_columns=ds.column_names,
        )
    elif dataset == "xstest-harmful":
        ds = load_dataset("walledai/XSTest", split="test")
        ds = ds.filter(lambda elm: elm["label"] == "unsafe")
        ds = ds.shuffle(seed=seed)
        ds = ds.select(range(size))
        return ds.map(
            lambda elm: {"prompt": elm["prompt"], "label": 1},
            remove_columns=ds.column_names,
        )
    # mixed datasets
    elif dataset == "xs-test":
        ds = load_dataset("walledai/XSTest", split="test")
        ds = ds.shuffle(seed=seed)
        return ds.map(
            lambda elm: {"prompt": elm["prompt"], "label": 1 if elm["label"] == "unsafe" else 0},
            remove_columns=ds.column_names,
        )

    elif dataset == "5050_easy":
        ds1 = prepare_dataset("alpaca-cleaned", 250)
        ds2 = prepare_dataset("advbench", 250)
        return concatenate_datasets([ds1, ds2])

    elif dataset == "5050_hard":
        ds1 = prepare_dataset("xstest-benign", 250)
        ds2 = prepare_dataset("wildjailbreak-adv", 250)
        return concatenate_datasets([ds1, ds2])

    elif dataset == "5050_both":
        ds1 = prepare_dataset("5050_easy")
        ds2 = prepare_dataset("5050_hard")
        return concatenate_datasets([ds1, ds2])

    elif dataset == "5050_wild":
        ds1 = prepare_dataset("wildjailbreak-adv", 210)
        ds2 = prepare_dataset("wildjailbreak-benign", 210)
        return concatenate_datasets([ds1, ds2])
    elif dataset == "0199":
        ds1 = prepare_dataset("alpaca-cleaned", 990)
        ds2 = prepare_dataset("wildjailbreak-adv", 10)
        return concatenate_datasets([ds1, ds2])

def compute_metrics_with_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    """ Computes the evaluation metrics at a specific threshold.

    Args:
        y_true: The gold label
        y_score: The prediction score
        threshold: At which threshold the prediction should be

    Returns:

    """
    y_preds = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_preds, labels=[0, 1]).ravel()

    fpr = fp / (tn + fp) if (tn + fp) > 0 else None
    fnr = fn / (fn + tp) if (fn + tp) > 0 else None
    tpr = tp / (fn + tp) if (fn + tp) > 0 else None
    tnr = tn / (tn + fp) if (tn + fp) > 0 else None

    # Not used for single label
    is_single_label = (len(set(y_true)) == 1)
    if not is_single_label:
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_preds, average="binary", zero_division=0.0)
        acc = accuracy_score(y_true, y_preds)

    metrics = {
    "threshold": float(threshold),
    "fpr": fpr, # ORR
    "fnr": fnr, # ASR
    "tpr": tpr, # JB Detection Rate
    "tnr": tnr, # Benign Detection Rate
    }

    if not is_single_label:
        metrics["precision"] = float(precision)
        metrics["recall"] = float(recall)
        metrics["f1_score"] = float(f1)
        metrics["acc"] = float(acc)

    return metrics

def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    """ Tokenizes the prompts in the dataset and returns them as a new dataset.

    Args:
        dataset: The dataset with a 'prompt' and 'label' key to tokenize the 'prompt's.
        tokenizer: AutoTokenizer from the model to evaluate on

    Returns:
        A Dataset object with the tokenized prompts now at 'prompt'
    """
    def _tokenize_text(text) -> AutoTokenizer:
        return tokenizer(
            text["prompt"],
            truncation=True,
            max_length=512,
        )

    tokenized_dataset = dataset.map(_tokenize_text, batched=True, remove_columns=["prompt"])

    return tokenized_dataset

def format_filename(dataset: str, model: str, ending: str="json") -> str:
    """ Returns the the model name, the dataset and the ending as a string. Also cleans / int _

    Args:
        dataset: the string name of the dataset
        model: the string name of the model
        ending: the filetype to return

    Returns:
        A combined string of model_dataset.ending

    """
    legal_dataset = dataset.replace("/", "_")
    legal_model = model.replace("/", "_")
    return f"{legal_model}__{legal_dataset}.{ending}"

def save_file(dataset: str, model:str, metrics: dict[str, float]) -> None:
    """ Saves the metrics at the global output_path with dataset and model as filename

    Args:
        dataset: the string name of the dataset
        model: the string name of the model
        metrics: A dict of metrics to save
    """
    filename = format_filename(dataset, model)
    out_file = OUTPUT_PATH / Path(filename)
    out_file.write_text(json.dumps(metrics))

# run through all the datasets with each model at each threshold and evaluate their performance, save all the metrics in jsonl ?
def run_eval(berts: list[str], datasets:list[str], threshold: float = 0.5) -> None:
    """ This function runs over all berts and runs inferece of each datasets generating the evaluation metrics\

    berts: A list of huggingface or local paths to the bert classifiers
    datasets: a list of dataset names matching prepare_dataset retrieval
    threshold: The evaluation threshold for the y_score. Defaults to 0.5

    """
    # LOAD MODEL
    for bert in berts:
        print(f">>> Loading BERT: {bert} <<<")
        try:
            tokenizer = AutoTokenizer.from_pretrained(bert)
            if "llm-semantic-router/mmbert-safety-classifier-level1" == bert:
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    "jhu-clsp/mmBERT-base",
                    num_labels=2,
                    torch_dtype=torch.float32
                )
                model = PeftModel.from_pretrained(base_model, "llm-semantic-router/mmbert-safety-classifier-level1")
            else:
                model = AutoModelForSequenceClassification.from_pretrained(
                    pretrained_model_name_or_path=bert,
                    num_labels=2, # auto ?
                )
        except Exception as e:
            print(f">>> Failed to load {bert} with exception {e} <<<")
            continue

        # LOAD DATASET
        for dataset in datasets:
            print(f">>> Loading Dataset: {dataset} <<<")
            ds = prepare_dataset(dataset, 250)
            tok_dataset = tokenize_dataset(ds, tokenizer)

            # Initialize Trainer
            eval_args = TrainingArguments(
                per_device_eval_batch_size=8,
            )

            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

            trainer = Trainer(
                model=model,
                args=eval_args,
                data_collator=data_collator
            )
            print(f">>> Evaluating {bert} on {dataset} <<<")
            predictions = trainer.predict(tok_dataset)
            logits = predictions.predictions # preds
            y_true = predictions.label_ids # labels

            y_score = softmax(logits, axis=-1)[:, 1] # only the positive class probs

           # Generate Metrics
            avg_precision = average_precision_score(y_true, y_score)
            roc_auc = roc_auc_score(y_true, y_score)
            fprs, tprs, thresholds = roc_curve(y_true, y_score)

            # budget of 5% fprs
            fpr_budget_idx = np.argmin(np.abs(fprs - 0.05))

            # get budget values
            fpr_threshold = thresholds[fpr_budget_idx]
            budget_fpr = fprs[fpr_budget_idx]
            fnr = 1.0 - tprs[fpr_budget_idx]

            # budget of 5% fnr = 95% tpr
            fnr_budget_idx = np.argmin(np.abs(tprs - 0.95))
            fnr_threshold = thresholds[fnr_budget_idx]
            budget_fnr = 1.0 - tprs[fnr_budget_idx]
            fpr = fprs[fnr_budget_idx]

            results = {
                "model": bert,
                "dataset": dataset,
                "sample_size": y_true.size,
                "roc_auc": round(roc_auc, 4),
                "pr-auc": round(avg_precision, 4),
                "05_fpr": {
                    "threshold": round(fpr_threshold, 4),
                    "fpr": round(budget_fpr, 4),
                    "fnr": round(fnr, 4)
                },
                "05_fnr": {
                    "threshold": round(fnr_threshold, 4),
                    "fnr": round(budget_fnr, 4),
                    "fpr": round(fpr, 4)
                },
            }

            print(">>> Saving Results <<<")
            save_file(dataset, bert, results)


if __name__ == "__main__":
    run_eval(BERTS, DATASETS)