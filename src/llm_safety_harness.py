from dataclasses import dataclass
from src.filters.filter import Filter

#for one prompt, we want to allow multiple safety mechanisms 
@dataclass
class GenerationSafetyResult:
    prompt: str 
    output: str 
    #For each prompt every filter is tracked by filtertype name and if approved
    input_approvals: dict[str,bool] 
    internal_approvals: dict[str,bool]  
    output_approvals: dict[str,bool]   
    overall_approval: bool 
    

class SafeLLM(): 
    def __init__(self, model, tokenizer, 
                 input_filters: list[Filter] = None , 
                internal_filters: list[Filter] = None , 
                output_filters: list[Filter] = None , 
                safety_config: dict[str, any] = None):
        self.model = model
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.input_filters = input_filters
        self.internal_filters= internal_filters
        self.output_filters = output_filters
        self.safety_config = safety_config

    def apply_io_filters(self, elements: list[str],filters) -> list[dict[str,bool]]:
        input_approvals = [{} for _ in elements]
        for filter in filters: 
            #Filter processes entire batch 
            input_approval_per_batch: list[bool] = filter.validate(elements)
            for i, approval in enumerate(input_approval_per_batch): 
                input_approvals[i][filter.__class__.__name__] = approval
        return input_approvals 

    #register hooks for the model
    # not sure what it has to return yet
    def apply_internal_filters(self) -> dict[tuple[int, str], bool]: 
        pass

    def _generate(self,inputs: list[str]):
        #tokenize inputs 
        #per token batchwise generation, 
        #at each iteration check if internal filter has triggered, 
        #if so return None
        # placeholder for testing
        tokenized = self.tokenizer(inputs, return_tensors ="pt", padding=True, truncation=True).to(self.model.device)
        outputs = self.model.generate(**tokenized)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True), [{} for _ in range(len(inputs))]

    #allow for batched input
    def generate(self, inputs: list[str]) -> list[GenerationSafetyResult]:
        safety_result_list = []
        input_approvals = [{} for _ in range(len(inputs))]
        output_approvals = [{} for _ in range(len(inputs))]
        if self.input_filters:
            input_approvals = self.apply_io_filters(inputs,filters = self.input_filters) 
        outputs, internal_approvals = self._generate(inputs)
        if self.output_filters:
            output_approvals = self.apply_io_filters(outputs,filters = self.output_filters) 
        for i, input in enumerate(inputs): 
            overall_approval = (
                all(input_approvals[i].values()) and 
                all(internal_approvals[i].values()) and 
                all(output_approvals[i].values())
            )
            safety_result_list.append(GenerationSafetyResult(input, outputs[i], input_approvals[i], internal_approvals[i], output_approvals[i], overall_approval)) 
        return safety_result_list
