from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def _train_transform(augment: dict[str, Any]) -> transforms.Compose:
    ops: list[Any] = []
    if augment.get("enabled", False):
        degrees = float(augment.get("degrees", 10))
        translate = float(augment.get("translate", 0.1))
        scale = float(augment.get("scale", 0.1))
        ops.append(
            transforms.RandomAffine(
                degrees=degrees,
                translate=(translate, translate),
                scale=(1.0 - scale, 1.0 + scale),
            )
        )
    ops += [
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ]
    return transforms.Compose(ops)


def _test_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )


def build_dataloaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    root = Path(cfg["root"])
    root.mkdir(parents=True, exist_ok=True)

    batch_size = int(cfg["batch_size"])
    num_workers = int(cfg.get("num_workers", 0))
    pin_memory = bool(cfg.get("pin_memory", False))

    train_ds = datasets.MNIST(
        root=str(root),
        train=True,
        download=True,
        transform=_train_transform(cfg.get("augment", {})),
    )
    test_ds = datasets.MNIST(
        root=str(root),
        train=False,
        download=True,
        transform=_test_transform(),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, test_loader
