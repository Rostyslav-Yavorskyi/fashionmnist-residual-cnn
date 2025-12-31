from typing import Tuple
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split

import typing

def get_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

def get_train_val_loaders(
    root: str,
    batch_size: int,
    val_size: int = 5000,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    transform = get_transforms()

    full_train = datasets.FashionMNIST(
        root=root,
        train=True,
        download=True,
        transform=transform
    )

    train_size = len(full_train) - val_size
    train_dataset, val_dataset = random_split(
        full_train, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    return train_loader, val_loader


def get_test_loader(
    root: str,
    batch_size: int,
    num_workers: int = 0
) -> Tuple[DataLoader, typing.List[str]]:
    transform = get_transforms()
    
    test_dataset = datasets.FashionMNIST(
        root=root,
        train=False,
        download=True,
        transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return test_loader, test_dataset.classes

