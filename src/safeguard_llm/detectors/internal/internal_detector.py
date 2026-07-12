from abc import ABC, abstractmethod
from torch import Tensor
class InternalDetector(ABC): 
    def __init__(self, target_layer: str):
        self.target_layer = target_layer

    @abstractmethod
    def hook(self,module, input: Tensor,output:Tensor) -> None: 
        pass

    #analyse and approve the input string 
    @abstractmethod
    def is_unsafe(self) -> list[bool]: 
        pass

    @abstractmethod
    def reset(self) -> None: 
        pass