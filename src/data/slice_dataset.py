import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class MRISliceDataset(Dataset):
    """
    Expects a CSV with columns: subject,label
    And a slice root like: dataset/OASIS1/processed_slices/<subject>/*.png (or jpg)
    """
    def __init__(self, csv_path, slices_root, transform=None):
        self.df = pd.read_csv(csv_path)
        self.slices_root = slices_root
        self.transform = transform

        self.samples = []
        for _, row in self.df.iterrows():
            subject = row["subject"]
            label = int(row["label"])
            subj_dir = os.path.join(self.slices_root, subject)
            if not os.path.isdir(subj_dir):
                continue
            for f in os.listdir(subj_dir):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(subj_dir, f), label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No slice images found under {slices_root} using {csv_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")  # grayscale
        if self.transform:
            img = self.transform(img)
        return img, label, path
