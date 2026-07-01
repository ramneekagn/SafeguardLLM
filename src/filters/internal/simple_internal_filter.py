from abc import ABC, abstractmethod
from src.filters.internal.internal_filter import InternalFilter
from torch import Tensor
import torch
class SimpleInternalFilter(InternalFilter): 
    def __init__(self, target_layer: str):
        self.target_layer = target_layer
        #for each token from 0 to seq_len
        self.activation_cache: list[Tensor] = []
        self.pos = 0
        self.threshold = 0
        if torch.cuda.is_available():
            torch.set_default_device('cuda:0') 
    # the hook gets the entire batch 
    def hook(self, module, input: Tensor, output:Tensor) -> None: 
        #output dim is [batch_size,seq_len,dim]
        self.activation_cache.append(output[:,-1,:]) 
        self.pos += 1
        print(output.shape)

    #once the output is finished we validate
    def validate(self) -> list[bool]: 
        stacked = torch.stack(self.activation_cache, dim = 0)
        mean = torch.mean(stacked, dim=0) 
        comparison_tensor = (mean > self.threshold).any(dim=-1)  
        return comparison_tensor.tolist()
    
    def reset(self) -> None: 
        self.activation_cache = {}
        self.pos = 0