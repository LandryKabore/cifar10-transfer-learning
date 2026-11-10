"""Fine-tune ResNet18 on CIFAR-10 with transfer learning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model import CIFAR10_CLASSES, build_model, get_device

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"


def make_loaders(batch_size: int = 64) -> tuple[DataLoader, DataLoader]:
    # ImageNet normalization works well with pretrained ResNet weights.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224, padding=8),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_ds = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False
    )
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def train(
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    pretrained: bool = True,
) -> Path:
    device = get_device()
    print(f"Using device: {device}")

    train_loader, test_loader = make_loaders(batch_size=batch_size)
    model = build_model(num_classes=len(CIFAR10_CLASSES), pretrained=pretrained).to(device)

    # Only optimize parameters that require gradients (layer4 + classifier).
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    history = []
    best_acc = 0.0
    best_path = CKPT_DIR / "resnet18_cifar10_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{100 * correct / total:.1f}%")

        scheduler.step()
        train_loss = running_loss / total
        train_acc = 100.0 * correct / total
        test_acc = evaluate(model, test_loader, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 2),
                "test_acc": round(test_acc, 2),
            }
        )
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.2f}% test_acc={test_acc:.2f}%"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": CIFAR10_CLASSES,
                    "test_acc": best_acc,
                    "epoch": epoch,
                },
                best_path,
            )
            print(f"  Saved best checkpoint -> {best_path} ({best_acc:.2f}%)")

    metrics_path = CKPT_DIR / "training_history.json"
    metrics_path.write_text(json.dumps({"best_test_acc": best_acc, "history": history}, indent=2))
    print(f"Best test accuracy: {best_acc:.2f}%")
    print(f"History written to {metrics_path}")
    return best_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune ResNet18 on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        pretrained=not args.no_pretrained,
    )
