from safeguard_llm.detectors.detector import Detector  

class SimpleDetector(Detector):
    """ A SimpleDetector class for development and testing """
    def __init__(self):
        """ Initialize the basic simple detector"""
        super().__init__()

    def is_unsafe(self, inputs: list[str]) -> list[bool]:
        """ Evaluates if input has system_instruction, attack, exploit or harmful in it for testing purposes

          Args:
              inputs: a list of text strings

          Returns:
              a list for each string and its classifcation evaluation
          """
        harmful_list = ["system instructions", "attack", "exploit", "harmful"]
        response = [True]*len(inputs)
        for i, input in enumerate(inputs): 
            if any(elm in input for elm in harmful_list):
                response[i] = False
        return response

