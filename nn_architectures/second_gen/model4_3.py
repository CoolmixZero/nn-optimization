import torch
from torch import nn


class Model4_3(nn.Module):
    def __init__(self, in_features: int):
        super(Model4_3, self).__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 8)
        self.fc7 = nn.Linear(8, 16)
        self.fc8 = nn.Linear(16, 32)
        self.fc9 = nn.Linear(32, 64)
        self.fc10 = nn.Linear(64, 128)
        self.fc11 = nn.Linear(128, 256)
        self.fc12 = nn.Linear(256, 1)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = torch.relu(self.fc1(X))
        X = torch.relu(self.fc2(X))
        X = torch.relu(self.fc3(X))
        X = torch.relu(self.fc4(X))
        X = torch.relu(self.fc5(X))
        X = torch.relu(self.fc6(X))
        X = torch.relu(self.fc7(X))
        X = torch.relu(self.fc8(X))
        X = torch.relu(self.fc9(X))
        X = torch.relu(self.fc10(X))
        X = torch.relu(self.fc11(X))
        return torch.sigmoid(self.fc12(X))