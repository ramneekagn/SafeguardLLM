import torch
from torch import nn,Tensor 
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import tqdm 
class LinearProbe(nn.Module):
    def __init__(self,in_dim): 
        super().__init__()
        self.lin = nn.Linear(in_features= in_dim, out_features=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        return self.lin(x)
    
    def predict_proba(self,x):
        x = self.lin(x)
        return self.sig(x)
    
    def predict(self,x,threshold):
        score = self.predict_proba(x) 
        return ((score > threshold)*1).int()

