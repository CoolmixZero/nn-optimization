import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from typing import Tuple
import matplotlib.pyplot as plt


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

        print(f'Epoch {epoch}: Batch Loss - {total_loss}{" "*20}', end='\r')
    print('\n')
    return train_timeline


def get_preds(model: nn.Module, X: Tensor, rounded: bool = True) -> Tensor:
    with torch.inference_mode():
        preds = model(X)

    return torch.round(preds) if rounded else preds


def get_metrics(model: nn.Module, X: Tensor, y: Tensor) -> Tuple[float, ...]:
    acc = accuracy_score(y, get_preds(model, X))
    recall = recall_score(y, get_preds(model, X))
    precision = precision_score(y, get_preds(model, X))
    f1 = f1_score(y, get_preds(model, X))

    return acc, recall, precision, f1


def draw_learning_process(learning_timeline: list):
    fig, ax = plt.subplots()
    fig.set_figwidth(10)
    fig.set_figheight(8)
    ax.plot(learning_timeline, label='Learning Timeline')
    ax.legend()
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Total Loss')
    plt.show()


def evaluate_model(model: nn.Module, train_loader: DataLoader, lr: float,
                   X_test: Tensor, y_test: Tensor, epochs: int, filename: str):

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_timeline = train_loop(
        model, optimizer, criterion, epochs, train_loader)

    acc, recall, precision, f1 = get_metrics(model, X_test, y_test)
    print(
        f'Accuracy: {acc:.4f}\nRecall: {recall:.4f}\nPrecision: {precision:.4f}\nF1: {f1:.4f}')

    draw_learning_process(train_timeline)
    torch.save(model.state_dict(), f'models/{filename}.pt')
