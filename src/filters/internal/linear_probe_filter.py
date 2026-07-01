from src.filters.internal.internal_filter import InternalFilter
from typing import Any
class LinearProbe(InternalFilter): 
    def __init__(self):
        super().__init__()

    def validate(self, inputs: str, model: Any) -> list[bool]: 
        harmful_list = ["system instructions", "attack", "exploit", "harmful"]
        response = [True]*len(inputs)
        for i, input in enumerate(inputs): 
            if any(elm in input for elm in harmful_list):
                response[i] = False
        return response

