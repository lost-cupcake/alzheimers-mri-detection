import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from src.model.model_factory import get_model
from src.training.utils import get_slice_dataloaders

MODEL_NAME = "2d_cnn"
MODEL_PATH = r"D:\downloads\Alzheimer-Detection-full\saved_models\best_2d_cnn.pth"
DATA_DIR = "dataset/OASIS1/processed/slices"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = get_model(MODEL_NAME).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

_, val_loader = get_slice_dataloaders(DATA_DIR, batch_size=32)

y_true, y_pred, y_prob = [], [], []

with torch.no_grad():
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        prob = torch.softmax(out, dim=1)[:, 1]
        pred = torch.argmax(out, dim=1)

        y_true.extend(y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())
        y_prob.extend(prob.cpu().numpy())

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, digits=4))

print("\nROC-AUC:", roc_auc_score(y_true, y_prob))
