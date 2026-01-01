# FashionMNIST - CNN vs Residual CNN

This project compares three neural network architectures on the FashionMNIST dataset:
- MLP (baseline)
- CNN
- Residual CNN (ResNet-style)

The goal is to demonstrate:
- clean training / evaluation pipeline
- architectural comparison
- overfitting handling (early stopping)
- reproducible experiments via CLI

## Overview

FashionMNIST is a grayscale image classification task (28×28, 10 classes).
We benchmark increasingly expressive models to analyze accuracy, generalization
and error patterns.

This project is intended as a clean, reproducible baseline for image classification experiments using PyTorch.

## Setup

### Option 1: pip
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Option 2: uv (faster)
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Usage

### Train
```bash
uv run python train.py \
  --model rescnn \
  --epochs 20 \
  --batch-size 64 \
  --lr 1e-3
```

### Evaluate
```bash
uv run python eval.py \
  --model rescnn \
  --batch-size 64
```

### Reproducibility
Use `--seed` to make runs reproducible:

```bash
uv run python train.py --model rescnn --seed 42
```

CLI options are available via `--help`.

## Models

| Model       | Description                             |
|-------------|-----------------------------------------|
| MLP         | Fully-connected baseline                |
| CNN         | 2-layer convolutional network           |
| ResidualCNN | CNN with residual blocks (ResNet-style) |


## Data Split

  - Train: 60,000 samples
  - Validation: random split from training set (used for early stopping)
  - Test: 10,000 samples (never seen during training)

Early stopping is applied based on validation loss.

## Results

Results obtained with `--seed 42`, best checkpoint by validation loss.

| Model       | Test Accuracy |
|-------------|---------------|
| MLP         | 0.8835        |
| CNN         | 0.9150        |
| ResidualCNN | 0.9293        |


## Analysis

  - ResidualCNN achieves the best generalization performance
  - Largest confusion occurs between visually similar classes:
    - Shirt ↔ T-shirt / Pullover / Coat
  - Residual connections stabilize training but give only modest gains on small datasets

## Artifacts

Generated during training / evaluation:
```
artifacts/
  └── {model}_best.pt
results/
  └── {model}/ 
      ├── metrics.csv
      ├── loss.png
      ├── acc.png
      ├── confusion_matrix.png
      └── classification_report.txt
```

## Tech Stack
  - Python
  - PyTorch
  - Torchvision
  - Scikit-learn
  - Matplotlib
  - tqdm

## Next Steps

Potential improvements for future iterations:

- Add learning rate scheduler (Cosine / StepLR)
- Experiment with data augmentation (random crop, flip)
- Implement deeper ResidualCNN variants
- Add mixed precision training (AMP)
- Track experiments with TensorBoard or Weights & Biases
- Evaluate per-class accuracy and error cases

