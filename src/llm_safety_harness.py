
class LLMSafetyHarness(): 
    def __init__(self, model, tokenizer, input_filters: list, output_filters: list, internal_filters: list):
        self.model = model
        self.tokenizer = tokenizer
        self.input_filters = input_filters
        self.internal_filters= internal_filters
        self.output_filters = output_filters
        self.apply_internal_filters()

    def apply_input_filters(self) -> str:
        pass

    def apply_output_filters(self) -> str:
        pass

    #register hooks for the model
    def apply_internal_filters(self) -> str: 
        pass

    def _generate(self,inputs: list[str]):
        #tokenize inputs 
        #per token batchwise generation, 
        #at each iteration check if internal filter has triggered, 
        #if so return None

    #allow for batched input
    def generate(self, inputs: list[str]) -> str:
        if self.input_filters:
            approval = self.apply_input_filters(input) 
        
        if not approval: 
            return None
        
        outputs = self._generate(inputs)

        if self.output_filters:
            approval = self.apply_output_filters(input)     
        if not approval:      
            return None
        return outputs