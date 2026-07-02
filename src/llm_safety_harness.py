from dataclasses import dataclass
from src.detectors.detector import Detector
from src.detectors.internal.internal_detector import InternalDetector
from pathlib import Path 
from src.utils.load_config import load_safety_config
#for one prompt, we want to allow multiple safety mechanisms 
@dataclass
class GenerationSafetyResult:
    prompt: str 
    output: str 
    # map detector_name -> {"class_name": str, "approved": bool}
    input_approvals: dict[str, dict[str, any]] 
    internal_approvals: dict[str, dict[str, any]]  
    output_approvals: dict[str, dict[str, any]]       
    overall_approval: bool = True
    

class SafeLLM(): 
    def __init__(self, model, tokenizer, config_path: Path):
        config = load_safety_config(config_path)
        self.input_detectors: list[tuple[Detector, str]] = config["input_detectors"]  
        self.internal_detectors: list[tuple[InternalDetector, str]] = config["internal_detectors"] 
        self.output_detectors: list[tuple[Detector, str]] = config["output_detectors"] 
        self.safety_config: dict[str, any] = config["safety_config"]
        self.forward_hooks = []
        self.model = model
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        
    def apply_io_detectors(self, elements: list[str], detectors: list[tuple[Detector, str]]) -> list[dict[str, bool]]:
        input_approvals = [{} for _ in elements]
        for detector_instance, detector_name in detectors: 
            input_approval_per_batch: list[bool] = detector_instance.validate(elements)
            for i, approval in enumerate(input_approval_per_batch): 
                input_approvals[i][detector_name] = {
                    "class_name": detector_instance.__class__.__name__,
                    "approved": approval
                }
        return input_approvals 

    #register hooks for the model
    def apply_internal_detectors(self, detectors: list[tuple[InternalDetector, str]]) -> None: 
        for detector_instance, _ in detectors: 
                layer = detector_instance.target_layer
                submodule = self.model.get_submodule(layer)
                self.forward_hooks.append(submodule.register_forward_hook(detector_instance.hook))

    def remove_internal_detectors(self) -> None:
        for fwd_hook in self.forward_hooks: 
            fwd_hook.remove()

    def _generate(self,inputs: list[str]) -> tuple[list[str],list[dict[str, bool]]]:
        internal_approvals = [{} for _ in inputs]
        for internal_detector, _ in self.internal_detectors:
            internal_detector.approvals = []
        tokenized = self.tokenizer(inputs, return_tensors ="pt", padding=True, truncation=True).to(self.model.device)
        input_len = tokenized["input_ids"].shape[1]

        outputs = self.model.generate(**tokenized)
        outputs = outputs[:,input_len:]
        #read from hooks 
        for internal_detector, detector_name in self.internal_detectors:
                    internal_approval_per_batch = internal_detector.validate() 
                    for i, approval in enumerate(internal_approval_per_batch): 
                        internal_approvals[i][detector_name] = {
                            "class_name": internal_detector.__class__.__name__,
                            "approved": approval
                        }
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True), internal_approvals

    #allow for batched input
    def generate(self, inputs: list[str]) -> list[GenerationSafetyResult]:
        safety_result_list = []
        input_approvals = [{} for _ in range(len(inputs))]
        output_approvals = [{} for _ in range(len(inputs))]
        for internal_detector, _ in self.internal_detectors:
                internal_detector.reset()
        if self.input_detectors:
            input_approvals = self.apply_io_detectors(inputs,detectors = self.input_detectors) 
        if self.internal_detectors:
            self.apply_internal_detectors(self.internal_detectors) 

        outputs, internal_approvals = self._generate(inputs)
        self.remove_internal_detectors()
        if self.output_detectors:
            output_approvals = self.apply_io_detectors(outputs,detectors = self.output_detectors) 
        for i, input in enumerate(inputs): 
            overall_approval = (
                all(v["approved"] for v in input_approvals[i].values()) and 
                all(v["approved"] for v in internal_approvals[i].values()) and 
                all(v["approved"] for v in output_approvals[i].values())
            )
            safety_result_list.append(GenerationSafetyResult(input, outputs[i], input_approvals[i], internal_approvals[i], output_approvals[i], overall_approval)) 
        return safety_result_list
 