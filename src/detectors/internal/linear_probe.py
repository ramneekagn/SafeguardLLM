import torch
from torch import nn,Tensor 
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
class LinearProbe(nn.Module):
    def __init__(self,in_dim): 
        super().__init__()
        self.lin = nn.Linear(in_features= in_dim, out_features=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.lin(x)
        return self.sig(x)
    
    def predict(self,x,threshold):
        score = self.forward(x) # from 0 to 1 classic softmax output for binary 
        
        return ((score > threshold)*1).int()


def train_model(linear_probe_model,train_data): 
    #for linear probes it a bit different
    # our inputs are the activations, the dataset inputs are put into the llm, whose weights are frozen 
    # we have to pass the labels only to the linear model, the label is simple, 1 if harmless, 0 if harmful
    #for testing 
    #batch data ()
    #dataloader 
    #define optimizer

    optimizer = torch.optim.Adam(linear_probe_model.parameters(),lr=0.001)
    loss_fn = torch.nn.BCELoss()
    activations = 0 
    labels = 0 
    EPOCHS = 10
    linear_probe_model.train()
    for _ in range(EPOCHS): 
        for activation_batch,label_batch in train_data:
            optimizer.zero_grad() 
            score = linear_probe_model(activation_batch)
            loss = loss_fn(score,label_batch)
            loss.backward()
            optimizer.step()
    return linear_probe_model

def test_model_data():
    def test_model(linear_probe_model,test_data): 
        tp = 0
        tn = 0 
        n = 0 
        for activation_batch,label_batch in test_data:
            with torch.no_grad():
                prediction_batch = linear_probe_model.predict(activation_batch, 0.5)
            tp += torch.sum(torch.logical_and(prediction_batch == 1 , label_batch==1 ))
            tn += torch.sum(torch.logical_and(prediction_batch == 0, label_batch == 0))
            n += label_batch.shape[0]
        acc =  (tp+tn)/n
        return acc.item()

    if torch.cuda.is_available(): 
        device = "cuda:0"
    else: 
        device ="cpu"

    lp_model = LinearProbe(in_dim=4).to(device)
    #load 
    data = fetch_openml(name="banknote-authentication", as_frame=True, parser="auto")
    X = data.data.copy()
    y = data.target.astype('category').cat.codes
    X_numeric = X.select_dtypes(include=["number"])
    X_numeric = X_numeric.fillna(X_numeric.mean())
    X_tensor = torch.tensor(X_numeric.values, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1).to(device)

    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=0.2, random_state=42
    )
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset,batch_size=32)
    test_loader = DataLoader(test_dataset,batch_size=32)

    lp_model = train_model(lp_model,train_loader)

    acc = test_model(lp_model, test_loader)
    print(acc)