from abc import ABC, abstractmethod
from safeguard_llm.detectors.internal.internal_detector import InternalDetector
from torch import Tensor
import torch
class SimpleInternalDetector(InternalDetector): 
    def __init__(self, target_layer: str):
        self.target_layer = target_layer
        #for each token from 0 to seq_len
        self.activation_cache: list[Tensor] = []
        self.pos = 0
        self.threshold = 0
    # the hook gets the entire batch 
    def hook(self, module, input: Tensor, output:Tensor) -> None: 
        self.activation_cache.append(output[:,-1,:].detach().clone().to(self.device)) 
        self.pos += 1

    #once the output is finished we validate if any are above threshold
    def is_unsafe(self) -> list[bool]: 
        stacked = torch.stack(self.activation_cache, dim = 0)
        mean = torch.mean(stacked, dim=0)
        comparison_tensor = (mean > self.threshold).any(dim=-1)  
        return comparison_tensor.tolist()
        
    def reset(self) -> None: 
        self.activation_cache = []  
        self.pos = 0