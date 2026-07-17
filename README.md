# LLM Safety Harness
Introduction
A harness is defined as a software scaffold around an LLM. Unlike agent harnesses, where the scaffolds are RAGs or connections to tool calls, our method connects LLMs to various detectors. We use BERT classifiers to detect input and output text, while simultaneously auditing the models' internals using linear probes.
We evaluated the models using the following metrics: F1 score, accuracy and latency, comparing different combinations of detectors. 
The final decision on whether an input or generation is flagged as a jailbreak is made by combining the majority vote from all the classifiers. 

View [Report](https://github.com/ramneekagn/SafeguardLLM/blob/report/report/report_rai.pdf) for more detail. 

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