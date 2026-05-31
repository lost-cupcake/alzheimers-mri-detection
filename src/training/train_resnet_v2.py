import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from src.training.utils import get_slice_dataloaders, save_model

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PROCESSED_SLICES_DIR = os.path.join(ROOT, "dataset", "OASIS1", "processed", "slices")
MODEL_PATH = os.path.join(ROOT, "saved_models", "best_resnet18.pth")


def build_grayscale_resnet18(num_classes=2):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # adapt 3-channel pretrained conv to 1 channel, preserving learned features
    old_w = m.conv1.weight.data                       # [64, 3, 7, 7]
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.conv1.weight.data = old_w.mean(dim=1, keepdim=True)
    m.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(m.fc.in_features, num_classes))
    return m


def train_resnet(epochs=30, batch_size=32, lr=1e-4, weight_decay=1e-4,
                 patience=5, device="cuda" if torch.cuda.is_available() else "cpu"):
    train_loader, val_loader = get_slice_dataloaders(
        PROCESSED_SLICES_DIR, batch_size=batch_size, img_size=224)

    model = build_grayscale_resnet18(num_classes=2).to(device)

    class_weights = torch.tensor([2.2, 1.0], device=device)  # class 0 = Alzheimer's (minority)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc, no_improve = 0.0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)
        train_loss = running / len(train_loader.dataset)

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total if total else 0.0
        print(f"Epoch {epoch}/{epochs} - Loss: {train_loss:.4f} - Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc, no_improve = val_acc, 0
            save_model(model, MODEL_PATH)
            print(f"[INFO] New best ResNet saved: {best_val_acc:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[STOP] Early stopping after {patience} stale epochs.")
                break

    print(f"[DONE] Best val acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    train_resnet()