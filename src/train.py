"""Train ProductCNN on Fashion-MNIST.

Usage:
    python -m src.train --epochs 5 --batch-size 128 --lr 1e-3
    python -m src.train --quick   # 1 epoch, smoke test
"""
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import DATA_DIR, MODEL_PATH, MODELS_DIR
from .model import ProductCNN


def get_loaders(batch_size: int):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),  # Fashion-MNIST stats
    ])
    train_ds = datasets.FashionMNIST(
        root=str(DATA_DIR), train=True, download=True, transform=transform
    )
    test_ds = datasets.FashionMNIST(
        root=str(DATA_DIR), train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


def train(epochs: int, batch_size: int, lr: float):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_loader, test_loader = get_loaders(batch_size)
    model = ProductCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"  epoch {epoch} batch {batch_idx} loss {running_loss / batch_idx:.4f}")

        acc = evaluate(model, test_loader, device)
        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch}/{epochs}  test_acc={acc:.4f}  time={elapsed:.1f}s")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--quick", action="store_true", help="1 epoch smoke test")
    args = parser.parse_args()

    if args.quick:
        args.epochs = 1

    train(args.epochs, args.batch_size, args.lr)


if __name__ == "__main__":
    main()
