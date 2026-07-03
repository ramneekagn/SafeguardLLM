from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from src.detectors.detector import Detector

class BERTdetector(Detector):
    """
    A simple Filter class to validate inputs/outputs with a Bert model.

    Attributes:
        model_path (Path): The file path to the BERT model.
        threshold (float): The minimum confidence score for the label to evaluate to true. Defaults to 0.5.
        label_pos (int): The index position of the target label. Defaults to 0.
    """
    def __init__(self, model_path: Path, device: str, threshold: float = 0.5, label_pos: int = 0, ):
        super().__init__()
        self.threshold = threshold
        self.label_pos = label_pos
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()


    def validate(self, inputs: list[str]) -> list[bool]:
        """
        Validates a text batch against the threshold of the BERT model.

        Args:
            inputs (list[str]): A batch of strings to be labeled.

        Returns:
            list[bool]: A list of booleans for each input regarding if the target label exceed the set threshold.
        """
        input = self.tokenizer(
            inputs,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**input)
        # we assume here a multilabel model and only check for the first label of the model
        probs = torch.sigmoid(outputs.logits)
        first_label_probs = probs[:, self.label_pos]
        return (first_label_probs > self.threshold).tolist()








