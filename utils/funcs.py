import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from typing import Tuple



def train_loop(model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module,
               epochs: int, train_data: DataLoader) -> list:

    train_timeline = []
    for epoch in range(epochs):
        total_loss = 0
        for features, labels in train_data:
            preds = model(features)
            loss = criterion(preds, labels)
            total_loss += loss.detach()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_timeline.append(total_loss)
        print(f'Epoch {epoch}: Batch Loss - {total_loss}')

    return train_timeline