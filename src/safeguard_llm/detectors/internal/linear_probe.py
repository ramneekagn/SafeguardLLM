from torch import nn, Tensor


class LinearProbe(nn.Module):
    """ A class object of a linear probe to classify activations..

    Attributes:
        lin: The linear network of the probe
        sig: Sigmoid activation functions
    """
    def __init__(self,in_dim: int) -> None:
        """ Initializes the linear probe.

        Args:
            in_dim: Defines the input dimensions of the linear probe\activation vectors.
        """
        super().__init__()
        self.lin = nn.Linear(in_features= in_dim, out_features=1)
        self.sig = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        """ Computes raw logit scores

        Args:
            x: Input activation tensor

        Returns:
            Tensor of logits from the activation x

        """
        return self.lin(x)
    
    def predict_proba(self, x: Tensor) -> Tensor:
        """ Predicts the sigmoid activations of the probe classification

        Args:
            x: Input tensor to predict

        Returns:
            An output tensor where each element of x got sigmoid activates

        """
        x = self.lin(x)
        return self.sig(x)
    
    def predict(self,x: Tensor ,threshold: float):
        """ Binary classifies the input tensor x with a given threshold.

        Args:
            x: Input activation
            threshold: Decision threshold for classifcation

        Returns:
            

        """
        score = self.predict_proba(x) 
        return ((score > threshold)*1).int()

