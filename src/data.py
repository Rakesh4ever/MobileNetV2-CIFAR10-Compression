"""CIFAR-10 dataset preparation, normalization, and data augmentation pipeline."""

from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Standard CIFAR-10 channel-wise statistics computed over the 50,000 training images.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def get_cifar10_transforms(use_random_erasing: bool = True) -> Tuple[transforms.Compose, transforms.Compose]:
    """Build training and evaluation transforms for CIFAR-10.

    Training pipeline:
      1. RandomCrop(32, padding=4, padding_mode='reflect') for translation invariance.
      2. RandomHorizontalFlip(p=0.5) for horizontal mirror invariance.
      3. ToTensor() scaling pixel intensities from [0, 255] to [0.0, 1.0].
      4. Normalize(CIFAR10_MEAN, CIFAR10_STD) for zero-mean, unit-variance inputs.
      5. Optional RandomErasing (Cutout-style) to prevent overfitting on small 32x32 crops.

    Evaluation pipeline:
      1. ToTensor()
      2. Normalize(CIFAR10_MEAN, CIFAR10_STD)
    """
    train_ops = [
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
    if use_random_erasing:
        train_ops.append(
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0.0)
        )

    test_ops = [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
    return transforms.Compose(train_ops), transforms.Compose(test_ops)


def get_cifar10_loaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    eval_batch_size: int = 256,
    num_workers: int = 2,
    use_random_erasing: bool = True,
    pin_memory: bool = True,
    seed: int = 42,
    calibration_samples: int = 1024,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, test_loader, calib_loader) for CIFAR-10.

    The calibration loader draws a deterministic subset of training images using
    evaluation (non-augmented) transforms for activation range calibration.
    """
    train_transform, test_transform = get_cifar10_transforms(
        use_random_erasing=use_random_erasing
    )

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    calib_base = datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=test_transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    # Deterministic generator for reproducible shuffling and calibration subset selection
    gen = torch.Generator().manual_seed(seed)
    calib_indices = torch.randperm(len(calib_base), generator=gen)[:calibration_samples].tolist()
    calib_dataset = Subset(calib_base, calib_indices)

    loader_gen = torch.Generator().manual_seed(seed)
    use_pin = pin_memory and torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin,
        drop_last=True,
        generator=loader_gen,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin,
    )
    calib_loader = DataLoader(
        calib_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin,
    )
    return train_loader, test_loader, calib_loader
