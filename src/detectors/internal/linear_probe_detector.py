from abc import ABC, abstractmethod
from src.detectors.internal.internal_detector import InternalDetector
from torch import Tensor
import torch
from linear_probe import LinearProbe
from pathlib import Path

class LinearProbeDetector(InternalDetector): 

    def __init__(self,model_path: Path, device: str, target_layer_args: dict):
        #for each token from 0 to seq_len
        self.target_layer = target_layer_args["name"]
        self.model_path = model_path
        self.device = device
        self.activation_cache: list[Tensor] = []
        self.pos = 0
        self.threshold = 0

    # the hook gets the entire batch 
    def hook(self, module, input: Tensor, output:Tensor) -> None: 
        input_dim = output.shape[2]
        if self.probe is None: 
            self.probe = LinearProbe(input_dim).to(self.device)
            self.probe.load_state_dict(self.model_path)
        self.activation_cache.append(output[:,-1,:].detach().clone().to(self.device)) 
        self.pos += 1

    #once the output is finished we validate if any are above threshold
    def validate(self) -> list[bool]: 
        stacked = torch.stack(self.activation_cache, dim = 0)
        mean = torch.mean(stacked, dim=0)
        comparison_tensor = (mean > self.threshold).any(dim=-1)  
        return comparison_tensor.tolist()
        
    def reset(self) -> None: 
        self.activation_cache = []  
        self.pos = 0