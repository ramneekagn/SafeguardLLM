from dataclasses import dataclass
from src.filters.filter import Filter
from src.filters.internal.internal_filter import InternalFilter
from pathlib import Path 
from src.utils.load_config import load_safety_config
#for one prompt, we want to allow multiple safety mechanisms 
@dataclass
class GenerationSafetyResult:
    prompt: str 
    output: str 
    #For each prompt every filter is tracked by filtertype name and if approved
    input_approvals: dict[str,bool] 
    internal_approvals: dict[str,bool]  
    output_approvals: dict[str,bool]   
    # when all filters approve
    overall_approval: bool 
    

class SafeLLM(): 
    def __init__(self, model, tokenizer, config_path: Path):
        config = load_safety_config(config_path)
        self.input_filters: list[tuple[Filter, str]] = config["input_filters"]  
        self.internal_filters: list[tuple[InternalFilter, str]] = config["internal_filters"] 
        self.output_filters: list[tuple[Filter, str]] = config["output_filters"] 
        self.safety_config: dict[str, any] = config["safety_config"]
        self.forward_hooks = []
        self.model = model
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def apply_io_filters(self, elements: list[str], filters: list[tuple[Filter, str]]) -> list[dict[str, bool]]:
        input_approvals = [{} for _ in elements]
        for filter_instance, filter_name in filters: 
            input_approval_per_batch: list[bool] = filter_instance.validate(elements)
            for i, approval in enumerate(input_approval_per_batch): 
                input_approvals[i][filter_name] = approval
                
        return input_approvals 

    #register hooks for the model
    def apply_internal_filters(self, filters: list[tuple[InternalFilter, str]]) -> None: 
        for filter_instance, _ in filters: 
                layer = filter_instance.target_layer
                submodule = self.model.get_submodule(layer)
                self.forward_hooks.append(submodule.register_forward_hook(filter_instance.hook))

    def remove_internal_filters(self) -> None:
        for fwd_hook in self.forward_hooks: 
            fwd_hook.remove()

    def _generate(self,inputs: list[str]) -> tuple[list[str],list[dict[str, bool]]]:
        internal_approvals = [{} for _ in inputs]
        for internal_filter, _ in self.internal_filters:
            internal_filter.approvals = []
        tokenized = self.tokenizer(inputs, return_tensors ="pt", padding=True, truncation=True).to(self.model.device)
        outputs = self.model.generate(**tokenized)
        #read from hooks 
        for internal_filter, filter_name in self.internal_filters:
            internal_approval_per_batch = internal_filter.validate() 
            for i, approval in enumerate(internal_approval_per_batch): 
                internal_approvals[i][filter_name] = approval
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True), internal_approvals

    #allow for batched input
    def generate(self, inputs: list[str]) -> list[GenerationSafetyResult]:
        safety_result_list = []
        input_approvals = [{} for _ in range(len(inputs))]
        output_approvals = [{} for _ in range(len(inputs))]
        if self.input_filters:
            input_approvals = self.apply_io_filters(inputs,filters = self.input_filters) 
        if self.internal_filters:
            self.apply_internal_filters(self.internal_filters) 

        outputs, internal_approvals = self._generate(inputs)
        self.remove_internal_filters()
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
 