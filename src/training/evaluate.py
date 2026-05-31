import os
import sys
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)

from src.model.model_2d_cnn import Alzheimer2DCNN
from src.training.utils import get_slice_dataloaders

PROCESSED_SLICES_DIR = os.path.join(ROOT, "dataset", "OASIS1", "processed", "slices")
MODEL_PATH = os.path.join(ROOT, "saved_models", "best_2d_cnn.pth")

# folder "0" = Alzheimer's, folder "1" = Normal  (index 0 = Alzheimer's)
CLASS_NAMES = ["Alzheimer's", "Normal"]


def evaluate_2d_cnn(device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    _, val_loader = get_slice_dataloaders(PROCESSED_SLICES_DIR, batch_size=32)

    model = Alzheimer2DCNN(num_classes=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    y_true, y_pred, y_prob_alz = [], [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            prob_alz = torch.softmax(out, dim=1)[:, 0]   # P(class 0 = Alzheimer's)
            preds = out.argmax(1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob_alz.extend(prob_alz.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob_alz = np.array(y_prob_alz)

    acc = (y_pred == y_true).mean()
    cm = confusion_matrix(y_true, y_pred)   # rows/cols ordered [0,1] = [Alzheimer's, Normal]

    print("\n=== SLICE-LEVEL RESULTS ===")
    print(f"Slice accuracy: {acc:.4f}\n")
    print("Confusion matrix (rows=true, cols=pred):")
    print(f"             pred:{CLASS_NAMES[0]:>12} {CLASS_NAMES[1]:>10}")
    for i, row in enumerate(cm):
        print(f"true {CLASS_NAMES[i]:>12}: {row[0]:>12} {row[1]:>10}")
    print()
    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

    # index 0 = Alzheimer's, so:
    tp_alz = cm[0, 0]   # true Alzheimer's, predicted Alzheimer's
    fn_alz = cm[0, 1]   # true Alzheimer's, predicted Normal
    tn_nrm = cm[1, 1]   # true Normal, predicted Normal
    fp_nrm = cm[1, 0]   # true Normal, predicted Alzheimer's
    sensitivity = tp_alz / (tp_alz + fn_alz) if (tp_alz + fn_alz) else 0.0
    specificity = tn_nrm / (tn_nrm + fp_nrm) if (tn_nrm + fp_nrm) else 0.0
    print(f"Sensitivity (Alzheimer's recall): {sensitivity:.4f}")
    print(f"Specificity (Normal recall):      {specificity:.4f}")

    # roc_auc_score wants prob of the positive label. Treat Alzheimer's (class 0) as positive:
    # build y_true_alz where 1 = Alzheimer's
    y_true_alz = (y_true == 0).astype(int)
    try:
        print(f"ROC-AUC: {roc_auc_score(y_true_alz, y_prob_alz):.4f}")
    except ValueError:
        print("ROC-AUC: undefined")


if __name__ == "__main__":
    evaluate_2d_cnn()