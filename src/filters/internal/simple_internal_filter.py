from abc import ABC, abstractmethod
from src.filters.internal.internal_filter import InternalFilter
from torch import Tensor

class SimpleInternalFilter(InternalFilter): 
    def __init__(self, target_layer: str):
        self.target_layer = target_layer
        #for each token from 0 to seq_len
        self.activation_cache: dict[int, Tensor] = {}
        self.pos = 0

    # the hook gets the entire batch 
    def hook(self, module, input: Tensor, output:Tensor) -> None: 
        #output dim is [batch_size,seq_len,dim]
        self.activation_cache[self.pos] =  output
        self.pos += 1
        print(output.shape)

    #once the output is finished we validate
    def validate(self) -> list[bool]: 
        return [True] * 4
    def reset(self) -> None: 
        self.activation_cache = {}
        self.pos = 0