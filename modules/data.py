import torch
import numpy as np
import torchvision
import torchvision.transforms
from typing import Type, Tuple
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import VisionDataset


def create_datasets(dataset: Type[VisionDataset], download_path: str, transform) -> Tuple[VisionDataset, VisionDataset, list]:
    """
    Creates training and test datasets, handling both 'split' and 'train' arguments.

    Args:
        dataset (Type[VisionDataset]): A torchvision dataset class (e.g., torchvision.datasets.Food101, torchvision.datasets.CIFAR10).
        download_path (str): Directory path to download/store the dataset.
        transform (torchvision.transforms.Compose): Transformations to apply to the dataset.

    Returns:
        tuple: (train_dataset, test_dataset, class_names)
    """
    try:
        train_data = dataset(root=download_path, split='train', download=True, transform=transform)
        test_data = dataset(root=download_path, split='test', download=True, transform=transform)
    except TypeError:
        train_data = dataset(root=download_path, train=True, download=True, transform=transform)
        test_data = dataset(root=download_path, train=False, download=True, transform=transform)

    class_names = train_data.classes if hasattr(train_data, 'classes') else list(set(train_data.targets))
    
    return train_data, test_data, class_names


def create_dataloaders(train_data, test_data, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """
    Wraps the datasets into PyTorch DataLoaders.

    Args:
        train_data (torch.utils.data.Dataset): Training dataset.
        test_data (torch.utils.data.Dataset): Testing dataset.
        batch_size (int): Number of samples per batch.

    Returns:
        tuple: (train_dataloader, test_dataloader)
    """
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    return train_loader, test_loader


def get_subset(dataset, fraction=0.1, seed=42):
    """
    Returns a random subset of a dataset.

    Args:
        dataset (torch.utils.data.Dataset): Full dataset.
        fraction (float): Fraction of dataset to sample (e.g., 0.1 for 10%).
        seed (int): Random seed for reproducibility.

    Returns:
        Subset: A subset of the original dataset.
    """
    np.random.seed(seed)
    subset_size = int(len(dataset) * fraction)
    indices = np.random.choice(len(dataset), size=subset_size, replace=False)
    return Subset(dataset, indices)
