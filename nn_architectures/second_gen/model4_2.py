import torch
from torch import nn


class Model4_2(nn.Module):
    def __init__(self, in_features: int):
        super(Model4_2, self).__init__()
        self.fc1 = nn.Linear(in_features, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 64)
        self.fc4 = nn.Linear(64, 512)
        self.fc5 = nn.Linear(512, 1024)
        self.fc6 = nn.Linear(1024, 1)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = torch.relu(self.fc1(X))
        X = torch.relu(self.fc2(X))
        X = torch.relu(self.fc3(X))
        X = torch.relu(self.fc4(X))
        X = torch.relu(self.fc5(X))
        return torch.sigmoid(self.fc6(X))