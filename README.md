# LLM Safety Harness
Introduction
A harness is defined as a software scaffold around an LLM. Unlike agent harnesses, where the scaffolds are RAGs or connections to tool calls, our method connects LLMs to various detectors. We use BERT classifiers to detect input and output text, while simultaneously auditing the models' internals using linear probes.
We evaluated the models on various metrics. 
The final decision on whether an input or generation is flagged as a jailbreak is made by AND ensemble from all the classifiers. 

<img width="708" height="203" alt="image" src="https://github.com/user-attachments/assets/25d6fe62-6b5a-42b1-9b12-183c888210f7" />

# Installation

## 1. Navigate to the repository directory

```cd path/to/SafeGuardLLM```
 

## 2. Create a virtual environment


```
#Create Virtual Environment
python -m venv venv
# Activate in Linux/Mac
source venv/bin/activate
# Activate in Windows
venv\scritps\activate
```

## 3. Install the necessary requirements

```pip install -r requirements.txt```

## 4. Install as package
  pip install -e .

## uv workflow

```
# create the virtual environment
uv venv 
# install the base dependencies
uv sync
```

# Project Structure

## Directories

- `experiments/`: Scripts or our final experiments
- `results/`: The main results of our experiments
- `src/`: Python source files for whole Safety LLM Harness
- `training/`: Training scripts for the detectors

## Configuration & Documents

- `README.md`: Project introductions
- `requirements.txt`: Python environment requirements
- `pyproject.toml`: Project configuration and dependencies
- `uv.lock`: Lockfile for reproducible dependency management

# Detectors

This project contains different trained safety classifiers:
You can load the BERT models from the following repos (Huggingface):
- `HanseeVee/roberta-base-input-jailbreak-classifier"`
- `HanseeVee/roberta-base-output-jailbreak-classifier`

- `src/safeguard_llm/models/` contains the internal probes.

# Experiments

Contains the experiments from the paper

# Execution script: 
 run_judge_reclassify_eval is used to create the final data for the experiments, i.e the predictions of the different classifiers
