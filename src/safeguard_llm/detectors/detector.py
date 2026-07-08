from abc import ABC, abstractmethod

class Detector(ABC): 
    def __init__(self):
        pass

    #analyse and approve the input string 
    @abstractmethod
    def is_unsafe(self,input: list[str]): 
        pass

    