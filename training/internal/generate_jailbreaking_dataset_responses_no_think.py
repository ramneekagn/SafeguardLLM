from datasets import load_dataset
from transformers import set_seed,AutoModelForCausalLM,AutoTokenizer
import torch 
from torch.utils.data import DataLoader
torch.cuda.is_available()
import pandas as pd
from datasets import Dataset
from tqdm.auto import tqdm  

jailbreak_dataset = load_dataset("sevdeawesome/jailbreak_success")
set_seed(40)
jb_dataset = jailbreak_dataset.shuffle(40)
jb_dataset = jb_dataset["train"]
jb_dataset = DataLoader(jb_dataset, batch_size=4)

print(jb_dataset)

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



prompts = []
generations = []
i = 0
#problem thinking is off, with thinking tokens, the generation is too expensive
#maybe we can limit the thinking to n tokens 
for batch in tqdm(jb_dataset, desc="Generating responses"):
    formatted_batch = []
    prompts.extend(batch["jailbreak_prompt_text"]) 
    batch_prompts = batch["jailbreak_prompt_text"]
    for prompt in batch_prompts:
        messages = [{"role": "user", "content": prompt}]
        formatted_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=False
        )
        formatted_batch.append(formatted_text)
        
    encoded = tokenizer(formatted_batch, return_tensors="pt", padding=True).to(device)
    outputs = model.generate(**encoded, max_new_tokens=100)
    input_len = encoded.input_ids.shape[1]
    outputs = outputs[:, input_len:]
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    generations.extend(decoded_outputs)
    i += 1

df = pd.DataFrame({
    "prompt": prompts,
    "generation": generations
})
hf_dataset = Dataset.from_pandas(df)

df.to_csv("jailbreak_generations.csv", index=False)
hf_dataset.save_to_disk("jailbreak_generations_dataset")





