import torch
from torch import nn


class Model4(nn.Module):
    def __init__(self, in_features: int):
        super(Model4, self).__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 128)
        self.fc5 = nn.Linear(128, 256)
        self.fc6 = nn.Linear(256, 1)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = torch.relu(self.fc1(X))
        X = torch.relu(self.fc2(X))
        X = torch.relu(self.fc3(X))
        X = torch.relu(self.fc4(X))
        X = torch.relu(self.fc5(X))
        return torch.sigmoid(self.fc6(X))