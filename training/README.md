This section contains the training scripts for our models.

This includes: 
The output and input berts that are included in the paper

->The final linear probe included in the paper  (train_lp_jailbreak_simple.ipynb)

->An MLP probe training script, and an output linear probe, which predicts if an output harmful will be likely generated from the input. Both were outperformed by the LP and therefore not included in the paper.

-> Finally an LP script that is trained on a toxicity dataset, it contains the layer sweep which shows that the final layer has the strongest performance in terms of AU-ROC
