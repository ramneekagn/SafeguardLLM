# %%
import torch
from torch import nn, Tensor 
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from safeguard_llm.detectors.internal.linear_probe import LinearProbe
from datasets import concatenate_datasets
import os

# %%
BATCH_SIZE = 2

ds = load_dataset("jackhhao/jailbreak-classification")

def label_to_tensor(example):
    example["type"] = 1 if example["type"] == "jailbreak" else 0
    return example

ds = ds["train"].map(label_to_tensor)
print(ds)

ds_yes = ds.filter(lambda example: example["type"] == 1 )
ds_no = ds.filter(lambda example: example["type"] == 0 )
print("ds_yes: ", ds_yes)
print("ds_no: ", ds_no)

ds = ds.shuffle(seed=40)

train_dataloader = DataLoader(ds, batch_size=BATCH_SIZE)

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B", device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-1.7B", padding_side="left"
)
if torch.cuda.is_available():
    device = "cuda"
else: 
    device = "cpu"

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# %%
import torch
from tqdm.auto import tqdm
import numpy as np 
from sklearn import metrics
import matplotlib.pyplot as plt  

# Global variable to store activation cache
activation_cache = None

def hook(module, input: Tensor, output: Tensor) -> None:
    global activation_cache 
    activation_cache = output[:, -1, :].detach().clone().to(device)


def extract_and_cache_activations(layer, dataloader, prompt_col, label_col, cache_path):
    """
    Runs a single forward pass over the dataset, extracts hidden states, 
    and saves them as a static PyTorch file on CPU.
    """
    forward_hook = model.model.layers[layer].register_forward_hook(hook)
    
    all_activations = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Extracting activations for caching", leave=True)
    
    for batch in pbar:
        inputs = batch[prompt_col]
        labels = batch[label_col].to(device).bfloat16()
        messages = [
            [{"role": "user", "content": prompt}]
            for prompt in inputs
        ]
        templated_inputs = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        tokenized_input = tokenizer(
            templated_inputs, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            _ = model(**tokenized_input)
            
        # Collect and move to CPU to conserve GPU VRAM
        all_activations.append(activation_cache.cpu())
        all_labels.append(labels.cpu())
        
    forward_hook.remove()
    
    # Concatenate lists into single tensors and save
    cached_activations = torch.cat(all_activations, dim=0)
    cached_labels = torch.cat(all_labels, dim=0)
    
    torch.save({"activations": cached_activations, "labels": cached_labels}, cache_path)
    print(f"Activations successfully saved to {cache_path}")


def train_model(layer, prompt_col, label_col):
    cache_path = f"cached_activations_layer{layer}.pt"
    
    # Check if cached activations already exist
    if not os.path.exists(cache_path):
        print(f"Cache file '{cache_path}' not found. Initiating extraction process...")
        extract_and_cache_activations(layer, train_dataloader, prompt_col, label_col, cache_path)
    else:
        print(f"Found cache file '{cache_path}'. Loading activations directly...")
        
    # Load activations and target labels
    cache = torch.load(cache_path)
    activations = cache["activations"].to(device).bfloat16()
    labels = cache["labels"].to(device).bfloat16()
    
    # Wrap in TensorDataset and use a larger batch size for ultra-fast training
    probe_dataset = TensorDataset(activations, labels)
    probe_dataloader = DataLoader(probe_dataset, batch_size=32, shuffle=True)
    
    linear_probe_model = LinearProbe(in_dim=2048).to(device).bfloat16()
    optimizer = torch.optim.Adam(linear_probe_model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    EPOCHS = 3
    linear_probe_model.train()
    for epoch in range(EPOCHS): 
        pbar = tqdm(probe_dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=True)
        
        for batch_activations, batch_labels in pbar:
            scores = linear_probe_model(batch_activations)
            optimizer.zero_grad() 
            loss = loss_fn(scores.squeeze(-1), batch_labels)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
    return linear_probe_model


def eval_accuracy(linear_probe_model, layer, data, threshold, prompt_col, label_col):
    forward_hook = model.model.layers[layer].register_forward_hook(hook)
    pbar = tqdm(data, leave=True)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    tp_list = []
    fp_list = []
    for batch in pbar:
        inputs = batch[prompt_col]
        labels = batch[label_col].to(device).bfloat16()
        messages = [
            [{"role": "user", "content": prompt}]
            for prompt in inputs
        ]
        templated_inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        tokenized_input = tokenizer(templated_inputs, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**tokenized_input)
            scores = linear_probe_model.predict(activation_cache, threshold=threshold)
            scores = scores.squeeze(-1)
            tp += torch.sum(torch.logical_and(scores == 1, labels == 1))
            tn += torch.sum(torch.logical_and(scores == 0, labels == 0))
            fp += torch.sum(torch.logical_and(scores == 1, labels == 0))
            fn += torch.sum(torch.logical_and(scores == 0, labels == 1))
    tp_list.append(tp)
    fp_list.append(fp)
    print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    print(fp)
    print(tn)
    print(fn)
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    print(accuracy)
    forward_hook.remove()


def roc(linear_probe_model, layer, data, prompt_col, label_col):
    forward_hook = model.model.layers[layer].register_forward_hook(hook)
    pbar = tqdm(data, leave=True)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    scores_full = []
    labels_full = []
    for batch in pbar:
        inputs = batch[prompt_col]
        labels = batch[label_col].to(device).bfloat16()
        messages = [
            [{"role": "user", "content": prompt}]
            for prompt in inputs
        ]
        templated_inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        tokenized_input = tokenizer(templated_inputs, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**tokenized_input)
            scores = linear_probe_model(activation_cache)
            scores_full.extend(scores.squeeze(-1).float().cpu())
            labels_full.extend(labels.float().cpu())
    fpr, tpr, thresholds = metrics.roc_curve(labels_full, scores_full)
    auroc_score = metrics.roc_auc_score(labels_full, scores_full)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                    name='example estimator')
    display.plot()
    plt.show()
    print("Auroc score: ", auroc_score)
    forward_hook.remove()

# %%
linear_probe_model = train_model(27, "prompt", "type")

torch.save(linear_probe_model.state_dict(), "model_linear_jailbreak_qwen3_1_7b_res_layer27_simple_jackhhao_jailbreakclassification_template.pt")


# %%
ds_eval = load_dataset("jackhhao/jailbreak-classification")

def label_to_tensor(example):
    example["type"] = 1 if example["type"] == "jailbreak" else 0
    return example

ds_eval = ds_eval["test"].map(label_to_tensor)

test_dataloader = DataLoader(ds_eval, batch_size=BATCH_SIZE)
roc(linear_probe_model, 27, test_dataloader, "prompt", "type")


# %%
ds_eval2 = load_dataset("dvilasuero/jailbreak-classification-gemma")
def label_to_tensor(example):
    example["gemma3-classification"] = 1 if example["gemma3-classification"] == "jailbreak" else 0
    return example

ds_eval2 = ds_eval2["train"].map(label_to_tensor)
print(ds_eval2)
test_dataloader = DataLoader(ds_eval2, batch_size=BATCH_SIZE)
roc(linear_probe_model, 27, test_dataloader, "prompt", "gemma3-classification")


# %%
ds_eval2 = load_dataset("allenai/wildjailbreak")
def label_to_tensor(example):
    example["label"] = 1 if example["label"] == "jailbreak" else 0
    return example

ds_eval2 = ds_eval2["train"].map(label_to_tensor)
print(ds_eval2)
test_dataloader = DataLoader(ds_eval2, batch_size=BATCH_SIZE)
roc(linear_probe_model, 27, test_dataloader, "adversarial", "label")


# %%
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader

ds = load_dataset("Necent/llm-jailbreak-prompt-injection-dataset")
split_name = "train" if "train" in ds else list(ds.keys())[0]
full_dataset = ds[split_name]

pos_ds = full_dataset.filter(lambda x: x["prompt_adversarial"] == 1)
neg_ds = full_dataset.filter(lambda x: x["prompt_adversarial"] == 0)
pos_subset = pos_ds.shuffle(seed=40).select(range(500))
neg_subset = neg_ds.shuffle(seed=40).select(range(500))
balanced_ds = concatenate_datasets([pos_subset, neg_subset]).shuffle(seed=40)
balanced_ds = balanced_ds.select_columns(["prompt", "prompt_adversarial"])

test_dataloader = DataLoader(balanced_ds, batch_size=BATCH_SIZE)

roc(
    linear_probe_model=linear_probe_model,
    layer=27,
    data=test_dataloader,
    prompt_col="prompt",
    label_col="prompt_adversarial"
)
