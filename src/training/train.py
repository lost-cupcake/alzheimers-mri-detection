import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from src.model.model_factory import get_model

# Fix Python path for VS Code
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)

from src.model.model_2d_cnn import Alzheimer2DCNN
from src.training.utils import get_slice_dataloaders, save_model

PROCESSED_SLICES_DIR = os.path.join(ROOT, "dataset", "OASIS1", "processed", "slices")
MODEL_PATH = os.path.join(ROOT, "saved_models", "best_2d_cnn.pth")

def train_2d_cnn(
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    train_loader, val_loader = get_slice_dataloaders(PROCESSED_SLICES_DIR, batch_size=batch_size)

    model = Alzheimer2DCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total if total > 0 else 0.0
        print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f} - Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, MODEL_PATH)
            print(f"[INFO] New best model saved with val acc {best_val_acc:.4f}")

    print("[DONE] Training complete.")

if __name__ == "__main__":
    train_2d_cnn()
