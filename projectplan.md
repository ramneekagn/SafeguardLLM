# The problem	

- LLMs are complex black box systems, that are hard to control

- Reinforcement Learning with Human Feedback and Supervised Finetuning is limited in terms of control

- Attacks and vulnerabilities persist even in current SOTA systems 

# Methodology

See the LLM as a safety critical system, audit the input, internals and output in real time.

Potential solutions include:

Input: Safety Classifier to detect malicious prompts, or jailbreaking attempts, Prompt rewriting

Internals: Sparse Autoencoders, Linear Probes, Steering Vectors 

Output: Safety Classifiers 

 Goal: Mitigate safety risks of LLMs with different tools

# Research Questions	

RQ1: How robust are safety classifiers for LLMs? 

RQ2: Can we successfully implement internal methods during runtime without inhibiting the models capabilities? 

RQ3: What are the limitations of current LLM safeguards?  

# Milesstones

- Create a pipeline around the llm that you can use to flexibly bootstrap input, internals at a  certain layer and output
- Build a reliable safety classifier for the input
- Build a reliable safety classifier for the output
- Find an complementary inner safety mechanism 

# Project Timeline

1. Week: 
Specify research questions and scope 
Setup eval pipeline and test
Literature Search about tools
Start training classifier 
2. Week:
Safety Classifier testing and improvement 
Test LLM system with classifier 
Analysis of results 
First shot with other methods
3. Week: 
Compare other methods with safety classifier
Refine code base 
Finalize Results
Start report and presentation 
4. Week:
Write up the report 
Finish presentation

