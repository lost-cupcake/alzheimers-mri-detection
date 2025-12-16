
import os
from typing import Tuple
import torch
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms

def get_slice_dataloaders(
    data_root: str,
    batch_size: int = 32,
    val_split: float = 0.2
) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root=data_root, transform=transform)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def save_model(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
