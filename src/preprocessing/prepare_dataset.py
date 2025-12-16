import os
import csv
from glob import glob
from typing import Dict, Optional

from src.preprocessing.convert_to_slices import volume_to_slices

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
RAW_DIR = os.path.join(ROOT, "dataset", "OASIS1", "raw")
PROCESSED_SLICES_DIR = os.path.join(ROOT, "dataset", "OASIS1", "processed", "slices")
LABELS_CSV = os.path.join(ROOT, "dataset", "OASIS1", "labels.csv")

def load_label_mapping(csv_path: str) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    if not os.path.isfile(csv_path):
        print(f"[WARN] labels.csv not found at {csv_path}. Treating all subjects as unlabeled.")
        return mapping

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subj = row.get("subject") or row.get("id") or row.get("folder")
            if subj:
                mapping[subj] = int(row["label"])
    return mapping

def find_first_mri_file(subject_folder: str) -> Optional[str]:
    search_locations = [
        os.path.join(subject_folder, "PROCESSED"),
        os.path.join(subject_folder, "RAW"),
        subject_folder
    ]
    patterns = ["*.nii", "*.nii.gz", "*.hdr", "*.img", "*.img.gz", "*.nhdr"]

    for loc in search_locations:
        if not os.path.isdir(loc):
            continue
        for pat in patterns:
            matches = glob(os.path.join(loc, pat))
            if matches:
                print(f"[INFO] MRI file found: {matches[0]}")
                return matches[0]
    return None

def preprocess_all_subjects() -> None:
    label_map = load_label_mapping(LABELS_CSV)

    if not os.path.isdir(RAW_DIR):
        raise RuntimeError(f"Raw directory not found: {RAW_DIR}")

    subjects = [d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))]
    print(f"[INFO] Found {len(subjects)} subject folders.")

    for subj in subjects:
        subj_dir = os.path.join(RAW_DIR, subj)
        mri_file = find_first_mri_file(subj_dir)

        if mri_file is None:
            print(f"[WARN] No MRI file found in {subj_dir}, skipping.")
            continue

        label = label_map.get(subj, -1)
        label_str = "unlabeled" if label == -1 else str(label)

        out_dir = os.path.join(PROCESSED_SLICES_DIR, label_str, subj)
        os.makedirs(out_dir, exist_ok=True)

        if len(os.listdir(out_dir)) > 0:
            print(f"[SKIP] Slices already exist for {subj}.")
            continue

        print(f"[INFO] Processing {subj} -> label {label_str}")

        try:
            volume_to_slices(
                mri_file,  # <-- pass file path, not array
                out_dir,
                num_slices=32,
                axis=2,
                resize=(128, 128)
            )
        except Exception as e:
            print(f"[ERROR] Failed processing {subj}: {e}")
            continue

    print("[DONE] Preprocessing completed.")

if __name__ == "__main__":
    preprocess_all_subjects()
