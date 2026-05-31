import os
import re
from typing import Tuple
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import GroupShuffleSplit


def _patient_id(path: str) -> str:
    """Patient ID from the slice's folder, e.g.
    .../slices/0/OAS1_0001_MR1/slice_000.png -> 'OAS1_0001'."""
    patient_folder = os.path.basename(os.path.dirname(path))
    m = re.match(r"(OAS1_\d+)", patient_folder)
    return m.group(1) if m else patient_folder


def get_slice_dataloaders(
    data_root: str,
    batch_size: int = 32,
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
    ])
    val_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    train_full = datasets.ImageFolder(root=data_root, transform=train_tf)
    val_full = datasets.ImageFolder(root=data_root, transform=val_tf)

    groups = [_patient_id(path) for path, _ in train_full.samples]
    gss = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
    train_idx, val_idx = next(gss.split(range(len(train_full)), groups=groups))

    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(val_full, val_idx)

    train_patients = {groups[i] for i in train_idx}
    val_patients = {groups[i] for i in val_idx}
    overlap = train_patients & val_patients
    assert not overlap, f"Patient leakage! Overlapping patients: {overlap}"
    print(f"[split] train: {len(train_patients)} patients / {len(train_idx)} slices | "
          f"val: {len(val_patients)} patients / {len(val_idx)} slices | overlap: 0")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def save_model(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)