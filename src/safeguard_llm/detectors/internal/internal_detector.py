from abc import ABC, abstractmethod
from torch import Tensor
class InternalDetector(ABC):
    """ Abstract base class for internal safety detectors.

    Attaches PyTorch forward hooks to a specific layers to monitor the activations.

    Attributes:
        target_layer: The identifier to the model layer to monitor
    """
    def __init__(self, target_layer: str):
        """ Initializes the internal detector class with the target layer name.

        Args:
            target_layer: Name of the layer to hook into.
        """
        self.target_layer = target_layer

    @abstractmethod
    def hook(self, module , input: Tensor ,output:Tensor) -> None:
        """ Hook into the layer to capture activations

        Args:
            module:
            input: The input tensor into the layer
            output: The output tensor from the layer
        """
        pass

    #analyse and approve the input string 
    @abstractmethod
    def is_unsafe(self) -> list[bool]:
        """ Evaluates the layers for harmful activations

        Returns:
            list[bool]:
        """

        pass

    @abstractmethod
    def reset(self) -> None: 
        pass