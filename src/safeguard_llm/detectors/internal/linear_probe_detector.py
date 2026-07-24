from safeguard_llm.detectors.internal.internal_detector import InternalDetector
from torch import Tensor
import torch
from safeguard_llm.detectors.internal.linear_probe import LinearProbe
from pathlib import Path
import numpy as np
class LinearProbeDetector(InternalDetector): 
    """ A internal detector loading a pre-trained probe model to monitor layer activations

    Attributes:
        target_layer: Layer to attach the hook to
        model_path: Path to the saved linear probe weights
        device: Identifier for inference such as 'cuda', 'cpu' 'mps'
        threshold: Probability threshold for classifying the activations as unsafe
        input_eval_only: If True, evaluates only the input
    """
    def __init__(self,model_path: Path, device: str, target_layer_name: dict, input_eval_only: bool, threshold: float = 0.53) -> None:
        """Initializes the LinearProbeDetector and lloads the pre-trained probe model

        model_path: Path to the saved linear probe weights
        device: Identifier for inference such as 'cuda', 'cpu' 'mps'
        target_layer_name: Name of the layer to hook into
        input_eval_only: If true evaluates only the input
        threshold: Probability threshold for classifying the activations as unsafe
        """
        self.target_layer = target_layer_name
        self.model_path = model_path
        self.device = device
        self.threshold = threshold
        print("LP: ", self.threshold)
        self.input_eval_only = input_eval_only
        self.eval_flag = True
        self.predictions_gen = []
        self.probe = None 

    def hook(self, module, input: Tensor, output:Tensor) -> None:
        """Captures activations and runs probe prediction.

        Args:
            module:
            input: Tensor of layer input
            output: Tensor of layer output
        """

        input_dim = output.shape[2]
        if self.probe is None: 
            self.probe = LinearProbe(input_dim).to(self.device).bfloat16()
            self.probe.load_state_dict(torch.load(self.model_path,weights_only=True))
        if self.eval_flag: 
            activation = output[:,-1,:].detach().clone() 
            preds_batch = self.probe.predict(activation,threshold=self.threshold)
            self.predictions_gen.append(preds_batch.squeeze(-1).cpu().numpy())  
        #if only prompt activation is evaluated
        if self.input_eval_only: 
            self.eval_flag = False

    def is_unsafe(self) -> list[bool]:
        """ Evaluates if any generation produced an unsafe activation.

        Returns:
            List[bool]: A list of boolean that capture if the probe activated or not.
        """
        predictions_full_gen = np.array(self.predictions_gen)
        batch_predictions = np.max(predictions_full_gen, axis=0)
        batch_predictions = batch_predictions.astype(bool)
        return batch_predictions.tolist()

            
    def reset(self) -> None:
        """ Resets the prediction list and sets flag for next evaluation"""
        self.predictions_gen = []
        self.eval_flag = True
