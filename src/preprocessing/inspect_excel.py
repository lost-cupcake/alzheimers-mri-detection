import pandas as pd

EXCEL_PATH = r"D:\downloads\oasis_cross-sectional-5708aa0a98d82080.xlsx"

df = pd.read_excel(EXCEL_PATH)

print("\n=== COLUMN NAMES IN YOUR EXCEL FILE ===")
for col in df.columns:
    print(col)
