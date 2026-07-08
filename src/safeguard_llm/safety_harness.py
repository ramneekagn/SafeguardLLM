import time
from typing import Any
from dataclasses import dataclass
from safeguard_llm.detectors.detector import Detector
from safeguard_llm.detectors.internal.internal_detector import InternalDetector
from pathlib import Path 
from safeguard_llm.utils.load_config import load_safety_config
#for one prompt, we want to allow multiple safety mechanisms 
@dataclass
class GenerationSafetyResult:
    prompt: str 
    output: str 
    # map detector_name -> {"class_name": str, "approved": bool}
    input_disapprovals: dict[str, dict[str, Any]]
    internal_disapprovals: dict[str, dict[str, Any]]
    output_disapprovals: dict[str, dict[str, Any]]
    overall_disapproval: bool = True


class SafeLLM(): 
    def __init__(self, model, tokenizer, max_gen_len, config_path: Path):
        config = load_safety_config(config_path)
        self.input_detectors: list[tuple[Detector, str]] = config["input_detectors"]  
        self.internal_detectors: list[tuple[InternalDetector, str]] = config["internal_detectors"] 
        self.output_detectors: list[tuple[Detector, str]] = config["output_detectors"] 
        self.safety_config: dict[str, Any] = config["safety_config"]
        self.forward_hooks = []
        self.model = model
        self.tokenizer = tokenizer
        self.max_gen_len = max_gen_len
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        if self.internal_detectors:
            self.apply_internal_detectors(self.internal_detectors)

    def apply_io_detectors(self, elements: list[str], detectors: list[tuple[Detector, str]]) -> list[dict[str, bool]]:
        input_disapprovals = [{} for _ in elements]
        for detector_instance, detector_name in detectors:
            start_time = time.perf_counter()
            input_disapproval_per_batch: list[bool] = detector_instance.is_unsafe(elements)
            latency = (time.perf_counter() - start_time) * 1000 # convert into ms
            for i, disapproval in enumerate(input_disapproval_per_batch): 
                input_disapprovals[i][detector_name] = {
                    "class_name": detector_instance.__class__.__name__,
                    "disapproved": disapproval,
                    "latency": round(latency, 2)
                }
        return input_disapprovals 

    #register hooks for the model
    def apply_internal_detectors(self, detectors: list[tuple[InternalDetector, str]]) -> None: 
        for detector_instance, _ in detectors: 
                layer = detector_instance.target_layer
                submodule = self.model.get_submodule(layer)
                self.forward_hooks.append(submodule.register_forward_hook(detector_instance.hook))

    def remove_internal_detectors(self) -> None:
        for fwd_hook in self.forward_hooks: 
            fwd_hook.remove()
        self.forward_hooks.clear()  
    def _generate(self,inputs: list[str]) -> tuple[list[str],list[dict[str, bool]]]:
        internal_disapprovals = [{} for _ in inputs]
        for internal_detector, _ in self.internal_detectors:
            internal_detector.disapprovals = []
        tokenized = self.tokenizer(inputs, return_tensors ="pt", padding=True, truncation=True).to(self.model.device)
        input_len = tokenized["input_ids"].shape[1]
        #TODO pass generation args 
        outputs = self.model.generate(**tokenized, max_new_tokens = self.max_gen_len)
        outputs = outputs[:,input_len:]
        #read from hooks 
        for internal_detector, detector_name in self.internal_detectors:
                    start_time = time.perf_counter()
                    internal_disapproval_per_batch = internal_detector.is_unsafe()
                    latency = (time.perf_counter() - start_time) * 1000 # convert into ms
                    for i, disapproval in enumerate(internal_disapproval_per_batch): 
                        internal_disapprovals[i][detector_name] = {
                            "class_name": internal_detector.__class__.__name__,
                            "disapproved": disapproval,
                            "latency": latency
                        }
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True), internal_disapprovals

    #allow for batched input
    def generate(self, inputs: list[str]) -> list[GenerationSafetyResult]:
        safety_result_list = []
        input_disapprovals = [{} for _ in range(len(inputs))]
        output_disapprovals = [{} for _ in range(len(inputs))]
        for internal_detector, _ in self.internal_detectors:
                internal_detector.reset()
        if self.input_detectors:
            input_disapprovals = self.apply_io_detectors(inputs,detectors = self.input_detectors) 
        outputs, internal_disapprovals = self._generate(inputs)
        if self.output_detectors:
            output_disapprovals = self.apply_io_detectors(outputs,detectors = self.output_detectors) 
        for i, input in enumerate(inputs): 
            overall_disapproval = (
                all(v["disapproved"] for v in input_disapprovals[i].values()) and 
                all(v["disapproved"] for v in internal_disapprovals[i].values()) and 
                all(v["disapproved"] for v in output_disapprovals[i].values())
            )
            safety_result_list.append(GenerationSafetyResult(input, outputs[i], input_disapprovals[i], internal_disapprovals[i], output_disapprovals[i], overall_disapproval)) 
        return safety_result_list
 