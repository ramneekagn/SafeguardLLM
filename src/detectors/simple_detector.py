from src.detectors.detector import Detector  

class SimpleDetector(Detector): 
    def __init__(self):
        super().__init__()

    def validate(self, inputs: list[str]) -> list[bool]: 
        harmful_list = ["system instructions", "attack", "exploit", "harmful"]
        response = [True]*len(inputs)
        for i, input in enumerate(inputs): 
            if any(elm in input for elm in harmful_list):
                response[i] = False
        return response

