
import os
import torch

from model.model_2d_cnn import Alzheimer2DCNN
from training.utils import get_slice_dataloaders

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PROCESSED_SLICES_DIR = os.path.join(ROOT, "dataset", "OASIS1", "processed", "slices")
MODEL_PATH = os.path.join(ROOT, "saved_models", "best_2d_cnn.pth")

def evaluate_2d_cnn(device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    _, val_loader = get_slice_dataloaders(PROCESSED_SLICES_DIR, batch_size=32)
    model = Alzheimer2DCNN(num_classes=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
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

    acc = correct / total if total > 0 else 0.0
    print(f"Validation Accuracy: {acc:.4f}")
    return acc

if __name__ == "__main__":
    evaluate_2d_cnn()
