import os
import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple
import argparse

import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

from datasets import get_train_val_loaders
from model import build_model
from utils import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="mlp", choices=["mlp", "cnn", "rescnn"])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_count += bs

        pbar.set_postfix(loss=loss.item())

    return total_loss / total_count, total_correct / total_count


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    pbar = tqdm(loader, desc="val  ", leave=False)
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_count += bs

        pbar.set_postfix(loss=loss.item())

    return total_loss / total_count, total_correct / total_count


def save_best_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    extra: Dict | None = None,
) -> None:
    ckpt = {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict()
    }
    if extra is not None:
        ckpt.update(extra)

    ckpt_dir = os.path.dirname(path)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    torch.save(ckpt, path)


def write_csv_metrics(path: str, rows: List[EpochMetrics]) -> None:
    metrics_dir = os.path.dirname(path)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for r in rows:
            w.writerow([r.epoch, r.train_loss, r.train_acc, r.val_loss, r.val_acc])


def plot_curves(out_dir: str, rows: List[EpochMetrics]) -> None:
    os.makedirs(out_dir, exist_ok=True)

    epochs = [r.epoch for r in rows]
    train_loss = [r.train_loss for r in rows]
    val_loss = [r.val_loss for r in rows]
    train_acc = [r.train_acc for r in rows]
    val_acc = [r.val_acc for r in rows]

    plt.figure()
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss.png"), dpi=150)
    plt.close()

    plt.figure()
    plt.plot(epochs, train_acc, label="train_acc")
    plt.plot(epochs, val_acc, label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "acc.png"), dpi=150)
    plt.close()


def fit(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    epochs: int = 10,
    ckpt_path: str = "artifacts/best.pt",
    metrics_path: str = "results/metrics.csv",
    plots_dir: str = "results",
    patience: int = 5,
    min_delta: float = 0.0
) -> List[EpochMetrics]:
    best_val_loss = float("inf")
    history: List[EpochMetrics] = []
    epochs_no_improve = 0
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history.append(EpochMetrics(epoch, train_loss, train_acc, val_loss, val_acc))

        print(
            f"Epoch {epoch:02d}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_epoch = epoch
            
            save_best_checkpoint(
                ckpt_path,
                model,
                optimizer,
                epoch,
                best_val_loss
            )
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Stopped at epoch {epoch}, best epoch was {best_epoch}")
            break

    write_csv_metrics(metrics_path, history)
    plot_curves(plots_dir, history)
    return history



def main():
    args = parse_args()
    set_seed(args.seed)

    train_loader, val_loader = get_train_val_loaders(
        root="data", 
        batch_size=64, 
        val_size=5000,
        seed=args.seed
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    ckpt_path = args.ckpt or f"artifacts/{args.model}_best.pt"
    model = build_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    fit(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        epochs=args.epochs,
        ckpt_path=ckpt_path,
        metrics_path=f"results/{args.model}/metrics.csv",
        plots_dir=f"results/{args.model}"
    )


if __name__ == "__main__":
    main()
