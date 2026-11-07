
import pandas as pd

from pathlib import Path

files = [
    Path("data/processed/questions.csv"),
    Path("results/manual_annotation_audit.csv")
]

for file in files:

    print("\nFILE:", file)

    if not file.exists():
        print("File not found!")
        continue

    df = pd.read_csv(file)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst record:")
    print(df.head(1).to_string(index=False))
