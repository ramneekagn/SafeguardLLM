from abc import ABC, abstractmethod
from torch import Tensor
class InternalFilter(ABC): 
    def __init__(self):
        pass

    #analyse and approve the input string 
    @abstractmethod
    def approve(self, activation: Tensor) -> bool: 
        pass

    