import os
import pandas as pd

# -------------------------------------------------------
# PATHS
# -------------------------------------------------------
EXCEL_PATH = r"D:\downloads\oasis_cross-sectional-5708aa0a98d82080.xlsx"

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
RAW_DIR = os.path.join(ROOT, "dataset", "OASIS1", "raw")
LABELS_PATH = os.path.join(ROOT, "dataset", "OASIS1", "labels.csv")


def create_labels():
    print(f"[INFO] Loading Excel: {EXCEL_PATH}")
    df = pd.read_excel(EXCEL_PATH)

    # Check required columns
    if "ID" not in df.columns or "CDR" not in df.columns:
        raise ValueError("Excel missing columns ID or CDR")

    # Clean subject IDs (remove MR session number)
    df["subject"] = df["ID"].str.split("_").str[0] + "_" + df["ID"].str.split("_").str[1]

    # Convert CDR into binary label
    # 0 = Healthy
    # 1 = Dementia (CDR >= 0.5)
    df["label"] = df["CDR"].apply(lambda x: 0 if float(x) == 0 else 1)

    # Prepare mapping
    excel_map = dict(zip(df["subject"], df["label"]))

    print("[INFO] Scanning raw folders...")
    rows = []

    for subj in os.listdir(RAW_DIR):
        if subj.startswith("OAS1"):
            base = subj.split("_MR")[0]  # OAS1_0001
            label = excel_map.get(base, 0)  # Default to 0 if missing
            rows.append([subj, label])

    out_df = pd.DataFrame(rows, columns=["subject", "label"])

    print(f"[INFO] Saving labels to: {LABELS_PATH}")
    out_df.to_csv(LABELS_PATH, index=False)
    print("[DONE] labels.csv created successfully!")


if __name__ == "__main__":
    create_labels()
