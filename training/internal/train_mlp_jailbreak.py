from datasets import load_dataset, concatenate_datasets
from safeguard_llm.detectors.internal.mlp_probe import MLPProbe
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader

from torch import optim
import torch
from tqdm.auto import tqdm

device = "cuda"
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B").to(device)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", padding_side = "left")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token



def train_mlp(layer, tensor_dataset): 
    mlp_probe = MLPProbe(in_dim=2048).to(device).bfloat16()
    data_loader = DataLoader(tensor_dataset, batch_size=16)
    optimizer = torch.optim.Adam(mlp_probe.parameters(), lr=1e-3, weight_decay=1e-4)
    loss = torch.nn.BCEWithLogitsLoss() 
    mlp_probe.train()
    epochs = 3
    for epoch in range(epochs):
        for inputs, labels in tqdm(data_loader): 
            optimizer.zero_grad()
            labels = labels.float().unsqueeze(-1)
            inputs = inputs.to(device, dtype=torch.bfloat16)
            labels = labels.to(device, dtype=torch.bfloat16)
            prediction = mlp_probe.forward(inputs) 
            output = loss(prediction, labels)
            output.backward()
            optimizer.step()
    return mlp_probe


loaded_data = torch.load("cached_activations_adversarial.pt")
activations = loaded_data["activations"]
labels = loaded_data["labels"]
ds_with_activations = TensorDataset(activations, labels)
mlp = train_mlp(27, ds_with_activations)
torch.save(mlp.state_dict(), "model_linear_jailbreak_qwen3_1_7b_res_layer27_mlp_allanai_adv_vanilla_mix_jb.pt")

