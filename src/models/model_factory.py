from typing import Tuple, Any
class ModelFactory(): 
    def __init__(self, model_config: dict[str, Any]):
        self.model_config = model_config
        self.source = "hf"

    #load model_config with source
    def _load(self) -> Tuple[Any,Any]:
        pass
    #allow for batched input
    def get(self, inputs: list[str]) -> str:
        model,tokenizer = self._load()
        return model,tokenizer