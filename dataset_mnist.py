# scripts/dataset_mnist.py
#
# Handles MNIST download, loading, and reference set generation.
# Import get_loaders() and get_reference_set() from this module.

import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

DATA_DIR = "data/"

def get_loaders(batch_size: int, data_dir: str = DATA_DIR):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = datasets.MNIST(
        data_dir, train=True,  download=True, transform=transform
    )
    test_set = datasets.MNIST(
        data_dir, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True,    # num_workers=4 for parallel data loading; pin_memory=True speeds up CPU→GPU transfer
                              persistent_workers=True)    # persistent_workers=True keeps workers alive across epochs for speed
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True,
                              persistent_workers=True)

    return train_loader, test_loader


def get_reference_set(n_samples: int = 1000,
                      data_dir: str = DATA_DIR,
                      seed: int = 0):
    """
    Return a fixed, frozen set of MNIST test images for activation extraction.
    The same reference set must be used across ALL subjects for comparability.

    Args:
        n_samples : number of reference images to draw
        data_dir  : root directory where MNIST is stored
        seed      : random seed for reproducible selection

    Returns:
        images : torch.Tensor of shape (n_samples, 1, 28, 28) — normalized
        labels : torch.Tensor of shape (n_samples,)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_set = datasets.MNIST(
        data_dir, train=False, download=True, transform=transform
    )

    # Fixed random selection — same across all calls with same seed
    rng     = np.random.default_rng(seed)
    indices = rng.choice(len(test_set), size=n_samples, replace=False)
    indices = sorted(indices.tolist())

    subset  = Subset(test_set, indices)
    loader  = DataLoader(subset, batch_size=n_samples, shuffle=False)

    images, labels = next(iter(loader))

    return images, labels


if __name__ == "__main__":
    print(f"Downloading MNIST to '{DATA_DIR}' ...")
    train_loader, test_loader = get_loaders(batch_size=64)
    print(f"Train batches : {len(train_loader)}")
    print(f"Test batches  : {len(test_loader)}")

    print(f"\nGenerating reference set ...")
    images, labels = get_reference_set(n_samples=1000)
    print(f"Reference set shape : {images.shape}")
    print(f"Labels distribution : { {i: (labels==i).sum().item() for i in range(10)} }")
    print("Done.")