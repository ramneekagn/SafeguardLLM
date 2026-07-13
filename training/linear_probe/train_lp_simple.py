import torch
from torch import nn,Tensor 
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from safeguard_llm.detectors.internal.linear_probe import LinearProbe
from datasets import concatenate_datasets
import torch
from tqdm.auto import tqdm
import numpy as np 
from sklearn import metrics
import matplotlib.pyplot as plt  

BATCH_SIZE = 2

ds = load_dataset("jackhhao/jailbreak-classification")

def label_to_tensor(example):
    example["type"] = 1 if example["type"] == "jailbreak" else 0
    return example

ds = ds["train"].map(label_to_tensor)
print(ds)

ds_yes = ds.filter(lambda example: example["type"] == 1 )
ds_no= ds.filter(lambda example: example["type"] == 0 )
print("ds_yes: ", ds_yes)
print("ds_no: ", ds_no)

ds = ds.shuffle(seed=40)

train_dataloader = DataLoader(ds, batch_size=BATCH_SIZE)


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




def hook(module, input: Tensor, output: Tensor) -> None:
    global activation_cache 
    activation_cache = output[:, -1, :].detach().clone().to(device)

def train_model(layer, prompt_col, label_col):
    linear_probe_model = LinearProbe(in_dim=2048).to(device).bfloat16()

    forward_hook = model.model.layers[layer].register_forward_hook(hook)
    optimizer = torch.optim.Adam(linear_probe_model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    EPOCHS = 3
    linear_probe_model.train()
    for epoch in range(EPOCHS): 
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=True)
        
        for batch in pbar:
            inputs = batch[prompt_col]
            labels = batch[label_col].to(device).bfloat16()
            tokenized_input = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**tokenized_input)
            scores = linear_probe_model(activation_cache)
            optimizer.zero_grad() 
            loss = loss_fn(scores.squeeze(-1), labels)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    forward_hook.remove()
    return linear_probe_model

def eval_accuracy(linear_probe_model, layer,data,threshold, prompt_col, label_col):
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
        tokenized_input = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**tokenized_input)
            scores = linear_probe_model.predict(activation_cache,threshold=threshold)
            scores = scores.squeeze(-1)
            tp += torch.sum(torch.logical_and(scores == 1 ,labels == 1))
            tn += torch.sum(torch.logical_and(scores == 0,labels == 0))
            fp += torch.sum(torch.logical_and(scores == 1,labels == 0))
            fn += torch.sum(torch.logical_and(scores == 0,labels == 1))
    tp_list.append(tp)
    fp_list.append(fp)
    print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    print(fp)
    print(tn)
    print(fn)
    accuracy = (tp + tn) /( tp + fp + tn + fn)
    print(accuracy)
    forward_hook.remove()

def roc(linear_probe_model, layer,data, prompt_col, label_col):
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
        tokenized_input = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**tokenized_input)
            scores = linear_probe_model(activation_cache)
            scores_full.extend(scores.squeeze(-1).float().cpu())
            labels_full.extend(labels.float().cpu())
    fpr, tpr, thresholds = metrics.roc_curve(labels_full,scores_full)
    auroc_score = metrics.roc_auc_score(labels_full,scores_full)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                    name='example estimator')
    display.plot()
    plt.show()
    print("Auroc score: ", auroc_score)
    forward_hook.remove()

linear_probe_model = train_model(27, "prompt", "type")

torch.save(linear_probe_model.state_dict(), "model_linear_jailbreak_qwen3_1_7b_res_layer27_simple_jackhhao_jailbreakclassification.pt")
