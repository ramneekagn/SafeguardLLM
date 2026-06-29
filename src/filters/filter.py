from abc import ABC, abstractmethod

class Filter(ABC): 
    def __init__(self):
        pass

    #analyse and approve the input string 
    @abstractmethod
    def validate(self,input: list[str]): 
        pass

    