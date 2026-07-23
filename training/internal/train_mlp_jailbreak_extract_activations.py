from datasets import load_dataset, concatenate_datasets
from safeguard_llm.detectors.internal.mlp_probe import MLPProbe
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from torch import optim, Tensor
import torch 
from tqdm.auto import tqdm

def prepare_dataset(target_task="content-harm", samples_per_group=5000):

    print(f"Loading and processing WildJailbreak for task: {target_task}...")
    full_ds = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False)["train"]
    
    v_benign = full_ds.filter(lambda x: x["data_type"] == "vanilla_benign").select(range(samples_per_group))
    a_benign = full_ds.filter(lambda x: x["data_type"] == "adversarial_benign").select(range(samples_per_group))
    v_harmful = full_ds.filter(lambda x: x["data_type"] == "vanilla_harmful").select(range(samples_per_group))
    a_harmful = full_ds.filter(lambda x: x["data_type"] == "adversarial_harmful").select(range(samples_per_group))
    
    def map_fn(elm):
        is_adversarial = "adversarial" in elm["data_type"]
        prompt = elm["adversarial"] if is_adversarial and elm["adversarial"] else elm["vanilla"]
        label = 1 if "harmful" in elm["data_type"] else 0
        return {"prompt": prompt, "label": label}
    
    v_benign = v_benign.map(map_fn, remove_columns=v_benign.column_names)
    a_benign = a_benign.map(map_fn, remove_columns=a_benign.column_names)
    v_harmful = v_harmful.map(map_fn, remove_columns=v_harmful.column_names)
    a_harmful = a_harmful.map(map_fn, remove_columns=a_harmful.column_names)
    
    mixed_ds = concatenate_datasets([v_benign, a_benign, v_harmful, a_harmful])
    mixed_ds = mixed_ds.shuffle(seed=40)

    return mixed_ds

ds_content = prepare_dataset(target_task="content-harm", samples_per_group=5000)

device = "cuda"
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B").to(device)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", padding_side = "left")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


cached_activations = []
def hook(module, input: Tensor, output:Tensor) -> None: 
    cached_activations.append(output[:,-1,:].detach().clone())

def cache_activations(model,tokenizer,layer,data, max_len):
    global cached_activations
    cached_activations = []
    model.eval()
    forward_hook = model.model.layers[layer].register_forward_hook(hook)
    data_loader = DataLoader(data, batch_size = 2)
    for batch in tqdm(data_loader): 
        inputs = batch["prompt"]
        messages = [[{"role": "user", "content": p}] for p in inputs]
        templated = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        tokenized = tokenizer(
            templated, padding=True, truncation=True, max_length=max_len,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**tokenized)
    forward_hook.remove()
    return_activations = cached_activations
    return return_activations 

raw_activations = cache_activations(model, tokenizer, 27, ds_content ,max_len= 512)
all_activations = torch.cat(raw_activations, dim=0)
all_labels = torch.tensor([sample["label"] for sample in ds_content])

torch.save({
    "activations": all_activations,
    "labels": all_labels
}, "cached_activations_adversarial.pt")
