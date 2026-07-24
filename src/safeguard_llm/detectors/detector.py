from abc import ABC, abstractmethod

class Detector(ABC):
    """ Abstract base class for bert safety detectors. """
    def __init__(self):
        """ Initializes the base safety detector """
        pass

    #analyse and approve the input string 
    @abstractmethod
    def is_unsafe(self,input: list[str]):
        """ Analyzes a batch of text for safety violations.

         Args:
             input: A list of text strings to evaluate
         """
        pass

    