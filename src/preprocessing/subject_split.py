import re
import pandas as pd
from sklearn.model_selection import train_test_split

def base_subject(subject_id: str) -> str:
    # OAS1_0239_MR2 -> OAS1_0239
    return re.sub(r"_MR\d+$", "", subject_id)

def make_subject_wise_splits(labels_csv_path: str, out_dir: str, seed: int = 42, test_size=0.15, val_size=0.15):
    df = pd.read_csv(labels_csv_path)
    df["base_subject"] = df["subject"].apply(base_subject)

    # We want a single label per base_subject
    # If multiple MR sessions exist, assume same label; if not, take mode
    subj_df = df.groupby("base_subject")["label"].agg(lambda x: int(x.mode().iloc[0])).reset_index()

    train_subj, temp_subj = train_test_split(
        subj_df, test_size=(test_size + val_size), random_state=seed, stratify=subj_df["label"]
    )

    # split temp into val/test
    val_ratio_in_temp = val_size / (test_size + val_size)
    val_subj, test_subj = train_test_split(
        temp_subj, test_size=(1 - val_ratio_in_temp), random_state=seed, stratify=temp_subj["label"]
    )

    train_ids = set(train_subj["base_subject"].tolist())
    val_ids   = set(val_subj["base_subject"].tolist())
    test_ids  = set(test_subj["base_subject"].tolist())

    df["split"] = df["base_subject"].apply(lambda s: "train" if s in train_ids else ("val" if s in val_ids else "test"))

    # Save split CSVs
    train_df = df[df["split"] == "train"].drop(columns=["base_subject"])
    val_df   = df[df["split"] == "val"].drop(columns=["base_subject"])
    test_df  = df[df["split"] == "test"].drop(columns=["base_subject"])

    import os
    os.makedirs(out_dir, exist_ok=True)
    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(out_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False)

    print("[DONE] Subject-wise splits saved:", out_dir)
    print("Train:", train_df["label"].value_counts().to_dict())
    print("Val:", val_df["label"].value_counts().to_dict())
    print("Test:", test_df["label"].value_counts().to_dict())
