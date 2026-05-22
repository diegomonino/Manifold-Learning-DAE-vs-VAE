"""
datasets.py
-----------
Unified dataset generation and loading interface.

Datasets:
    - SwissRoll:   2D manifold embedded in R^3 (generated locally)
    - LinearH:     d-dimensional manifold linearly embedded in R^D (generated locally)
    - MNIST:       784-dimensional image dataset (auto-downloaded via torchvision)

Usage:
    dataset = get_dataset("swiss_roll", n_samples=2000, noise=0.0)
    dataset = get_dataset("linear", n_samples=10000, intrinsic_dim=3, ambient_dim=30)
    dataset = get_dataset("mnist", n_samples=10000)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_swiss_roll
import torchvision
import torchvision.transforms as transforms
import os


class NumpyDataset(Dataset):
    """
    Wraps a numpy array into a PyTorch Dataset.
    Stores optional labels for visualization (coloring).
    """
    def __init__(self, data: np.ndarray, labels: np.ndarray = None):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = labels  # kept as numpy for matplotlib colormaps

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def make_swiss_roll_dataset(
    n_samples: int = 2000,
    noise: float = 0.0,
    random_state: int = 42
) -> NumpyDataset:
    """
    2D manifold embedded in R^3.
    Intrinsic dimension: 2, Ambient dimension: 3.
    """
    X, color = make_swiss_roll(
        n_samples=n_samples,
        noise=noise,
        random_state=random_state
    )
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return NumpyDataset(data=X, labels=color)


def make_linear_manifold_dataset(
    n_samples: int = 10000,
    intrinsic_dim: int = 3,
    ambient_dim: int = 30,
    random_state: int = 42
) -> NumpyDataset:
    """
    Uniform sampling from [0,1]^d, linearly embedded into R^D via
    a random orthonormal matrix. Replicates H_{d,D} from Ivanov et al. (2021).
    """
    assert ambient_dim >= intrinsic_dim, (
        f"ambient_dim ({ambient_dim}) must be >= intrinsic_dim ({intrinsic_dim})"
    )
    rng = np.random.RandomState(random_state)
    Z = rng.uniform(0, 1, size=(n_samples, intrinsic_dim))

    A_raw = rng.randn(ambient_dim, intrinsic_dim)
    A, _ = np.linalg.qr(A_raw)
    A = A[:, :intrinsic_dim]

    X = Z @ A.T
    return NumpyDataset(data=X, labels=Z[:, 0])


def make_mnist_dataset(
    n_samples: int = 10000,
    data_dir: str = "./data/raw",
    train: bool = True,
    flatten: bool = True
) -> NumpyDataset:
    """
    MNIST handwritten digits (auto-downloaded).
    Intrinsic dimension: ~10-12 (estimated). Ambient dimension: 784.
    """
    os.makedirs(data_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    mnist = torchvision.datasets.MNIST(
        root=data_dir, train=train, download=True, transform=transform
    )

    n_samples = min(n_samples, len(mnist))
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(mnist), generator=generator)[:n_samples]
    subset = torch.utils.data.Subset(mnist, indices)
    loader = DataLoader(subset, batch_size=n_samples, shuffle=False)
    X, y = next(iter(loader))

    if flatten:
        X = X.view(n_samples, -1).numpy()
    else:
        X = X.numpy()

    return NumpyDataset(data=X, labels=y.numpy())


def get_dataset(name: str, **kwargs) -> NumpyDataset:
    """Unified entry point. name: 'swiss_roll', 'linear', or 'mnist'."""
    registry = {
        "swiss_roll": make_swiss_roll_dataset,
        "linear": make_linear_manifold_dataset,
        "mnist": make_mnist_dataset,
    }
    if name not in registry:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(registry.keys())}")
    return registry[name](**kwargs)


def get_dataloader(name: str, batch_size: int = 256, shuffle: bool = True, **kwargs) -> DataLoader:
    """Convenience wrapper that returns a DataLoader directly."""
    dataset = get_dataset(name, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    print("== Swiss Roll ==")
    sr = get_dataset("swiss_roll", n_samples=2000)
    print(f"  Shape: {sr.data.shape}, Range: [{sr.data.min():.2f}, {sr.data.max():.2f}]")

    print("== Linear H_3,30 ==")
    h = get_dataset("linear", n_samples=10000, intrinsic_dim=3, ambient_dim=30)
    print(f"  Shape: {h.data.shape}, Range: [{h.data.min():.2f}, {h.data.max():.2f}]")

    print("== MNIST ==")
    mn = get_dataset("mnist", n_samples=1000)
    print(f"  Shape: {mn.data.shape}, Labels: {np.unique(mn.labels)}")
