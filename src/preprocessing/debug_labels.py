import os
import pandas as pd

# Build the correct absolute path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
LABEL_PATH = os.path.join(ROOT, "dataset", "OASIS1", "labels.csv")

print("[INFO] Loading:", LABEL_PATH)

df = pd.read_csv(LABEL_PATH)

print("\nLABEL DISTRIBUTION:")
print(df["label"].value_counts())

print("\nEXAMPLES FROM EACH CLASS:")
print(df.groupby("label").head(5))
