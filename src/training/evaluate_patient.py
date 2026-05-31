import os
import numpy as np
import torch
from collections import defaultdict
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.model.model_2d_cnn import Alzheimer2DCNN
from src.training.utils import _patient_id, get_slice_dataloaders

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PROCESSED_SLICES_DIR = os.path.join(ROOT, "dataset", "OASIS1", "processed", "slices")
MODEL_PATH = os.path.join(ROOT, "saved_models", "best_2d_cnn.pth")

# index 0 = folder "0" = Alzheimer's ; index 1 = folder "1" = Normal
CLASS_NAMES = ["Alzheimer's", "Normal"]


def evaluate_patient_level(device="cuda" if torch.cuda.is_available() else "cpu"):
    # rebuild the SAME val split, but keep file paths so we can group by patient
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    full = datasets.ImageFolder(PROCESSED_SLICES_DIR, transform=transform)

    # reuse the exact split logic from utils so val patients match training
    from sklearn.model_selection import GroupShuffleSplit
    groups = [_patient_id(p) for p, _ in full.samples]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, val_idx = next(gss.split(range(len(full)), groups=groups))

    model = Alzheimer2DCNN(num_classes=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # accumulate per-patient slice probabilities
    patient_probs = defaultdict(list)
    patient_true = {}

    with torch.no_grad():
        for i in val_idx:
            path, label = full.samples[i]
            pid = _patient_id(path)
            img, _ = full[i]
            img = img.unsqueeze(0).to(device)
            prob_alz = torch.softmax(model(img), dim=1)[0, 0].item()  # P(class 0 = Alzheimer's)
            patient_probs[pid].append(prob_alz)
            patient_true[pid] = label

    # aggregate: mean probability across a patient's slices
    pids = list(patient_probs.keys())
    y_prob = np.array([np.mean(patient_probs[p]) for p in pids])   # mean P(Alzheimer's)
    y_pred = (y_prob >= 0.5).astype(int)                            # 1 if Alzheimer's
    # NOTE: class 0 = Alzheimer's, so "predict Alzheimer's" means class 0.
    # Convert to label space: pred Alzheimer's -> 0, else -> 1
    y_pred_label = np.where(y_prob >= 0.5, 0, 1)
    y_true = np.array([patient_true[p] for p in pids])

    print(f"\n=== PATIENT-LEVEL RESULTS ({len(pids)} patients) ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred_label):.4f}")
    # AUC: probability of class 0 (Alzheimer's). roc_auc expects prob of positive label=1,
    # so feed prob of Normal = 1 - y_prob against y_true.
    print(f"ROC-AUC : {roc_auc_score(y_true, 1 - y_prob):.4f}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred_label))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred_label, target_names=CLASS_NAMES, digits=4))


if __name__ == "__main__":
    evaluate_patient_level()