from abc import ABC, abstractmethod

class IOFilter(ABC): 
    def __init__(self):
        pass

    #analyse and approve the input string 
    @abstractmethod
    def approve(input: str) -> bool: 
        pass

    