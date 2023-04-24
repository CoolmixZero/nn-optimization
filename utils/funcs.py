import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from typing import Tuple



def train_loop(model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module,
               epochs: int, train_data: DataLoader) -> list:

    train_timeline = []
    for epoch in range(epochs):
        batch_loss = 0
        for features, labels in train_data:
            preds = model(features)
            loss = criterion(preds, labels)
            batch_loss += loss.detach()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_timeline.append(batch_loss)
        print(f'Epoch {epoch}: Batch Loss - {batch_loss}')
        
    return list