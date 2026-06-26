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

# Decisions:

- What model? 
- What model size? 
- What problem? 
- What dataset/benchmark? 
- How does the red teaming approach look like? 
- What methods? 

# Milesstones

- Create a pipeline around the llm that you can use to flexibly bootstrap input, internals at a certain layer and output
- Choose model, and datasets
- Build a reliable safety classifier for the input
- Build a reliable safety classifier for the output
- Find an complementary inner safety mechanism
- Stress test the pipeline through red teaming and evals
- Finalize combination

# Project Timeline

1. Week: 
- Specify research questions and scope 
- Setup eval pipeline and test
- Literature Search about tools
- Implement input classifier (pretrained or finetune) 
- Evaluate Classifier
  
2. Week:
- Implement output classifier and evaluate 
- Implement inner method 
- Compare other methods with safety classifier in isolation
- Test in combination
- Analysis of results 

3. Week:
- Refine code base 
- Finalize Results
- First draft report and presentation 

4. Week:
- Finish presentation and report 
