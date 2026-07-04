from abc import ABC, abstractmethod
from src.detectors.internal.internal_detector import InternalDetector
from torch import Tensor
import torch
from src.detectors.internal.linear_probe import LinearProbe
from pathlib import Path
import numpy as np
class LinearProbeDetector(InternalDetector): 

    def __init__(self,model_path: Path, device: str, target_layer_name: dict, threshold: float = 0.53):
        #for each token from 0 to seq_len
        self.target_layer = target_layer_name
        self.model_path = model_path
        self.device = device
        self.threshold = threshold
        self.predictions_gen = []
        self.probe = None 


    def hook(self, module, input: Tensor, output:Tensor) -> None: 
        input_dim = output.shape[2]
        if self.probe is None: 
            self.probe = LinearProbe(input_dim).to(self.device).bfloat16()
            self.probe.load_state_dict(torch.load(self.model_path,weights_only=True))
        activation = output[:,-1,:].detach().clone() 
        preds_batch = self.probe.predict(activation,threshold=self.threshold)
        self.predictions_gen.append(preds_batch.squeeze(-1).cpu().numpy())

    #once the output is finished we validate if any are above threshold
    def validate(self) -> list[bool]: 
        predictions_full_gen = np.array(self.predictions_gen)
        batch_predictions = np.max(predictions_full_gen, axis=0)
        return np.array(batch_predictions, dtype=bool).tolist()

            
    def reset(self) -> None: 
        self.predictions_gen = []
