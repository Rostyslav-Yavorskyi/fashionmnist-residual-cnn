import os
from typing import Dict
from typing import List
import argparse

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

from model import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="mlp", choices=["mlp", "cnn", "rescnn"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--out-dir", type=str, default=None)
    return p.parse_args()


def load_checkpoint(path: str, model: nn.Module, device: str) -> Dict:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return ckpt


def save_confusion_matrix(y_true, y_pred, out_path: str, class_names: List[str] | None = None):
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(values_format="d", ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def save_classification_report(y_true, y_pred, out_path: str, class_names: List[str] | None = None):
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=False)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(str(report))

    
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    all_preds = []
    all_targets = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_count += bs

        all_preds.append(preds.cpu())
        all_targets.append(y.cpu())

    avg_loss = total_loss / total_count
    avg_acc = total_correct / total_count

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    
    return avg_loss, avg_acc, all_targets, all_preds


def main():
    args = parse_args()
    ckpt_path = args.ckpt or f"artifacts/{args.model}_best.pt"
    out_dir = args.out_dir or f"results/{args.model}"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    
    # Model
    model = build_model(args.model).to(device)

    # Load best checkpoint 
    ckpt = load_checkpoint(ckpt_path, model, device)
    print("Loaded checkpoint from epoch:", ckpt.get("epoch"), "best_val_loss:", ckpt.get("best_val_loss"))

    # Eval
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    print(f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")

    # Save artifacts
    class_names = test_dataset.classes

    save_confusion_matrix(y_true, y_pred, f"{out_dir}/confusion_matrix.png", class_names)
    save_classification_report(y_true, y_pred, f"{out_dir}/classification_report.txt", class_names)


if __name__ == "__main__":
    main()

