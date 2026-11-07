
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path

INPUT_FILE = Path(
    "data/processed/annotations/sentence_candidates.csv"
)

OUTPUT_FOLDER = Path("data/processed")

# Load sentence annotations
df = pd.read_csv(INPUT_FILE)

# Basic validation
print("DATASET VALIDATION")
print("-------------------------")
print("Total records:", len(df))
print("Missing sentences:", df["sentence"].isna().sum())
print("Missing labels:", df["label"].isna().sum())
print("Unique labels:", sorted(df["label"].unique()))

# Check both classes are present
if set(df["label"].dropna().unique()) != {0, 1}:
    raise ValueError("Both labels 0 and 1 are required.")

# Keep paired questions in the same split
split1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=42
)

train_idx, temp_idx = next(
    split1.split(df, groups=df["question_id"])
)

train = df.iloc[train_idx].copy()
temp = df.iloc[temp_idx].copy()

# Divide remaining records equally
split2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=42
)

val_idx, test_idx = next(
    split2.split(temp, groups=temp["question_id"])
)

validation = temp.iloc[val_idx].copy()
test = temp.iloc[test_idx].copy()

# Save datasets
train.to_csv(OUTPUT_FOLDER / "train.csv", index=False)
validation.to_csv(
    OUTPUT_FOLDER / "validation.csv", index=False
)
test.to_csv(OUTPUT_FOLDER / "test.csv", index=False)

# Display results
print("\nDATASET SPLITTING COMPLETED")
print("-------------------------")

for name, dataset in [
    ("Training", train),
    ("Validation", validation),
    ("Testing", test)
]:
    print(f"\n{name} records:", len(dataset))
    print(dataset["label"].value_counts().to_dict())
    print("Unique questions:", dataset["question_id"].nunique())

# Verify no question ID overlaps
assert set(train["question_id"]).isdisjoint(
    validation["question_id"]
)
assert set(train["question_id"]).isdisjoint(
    test["question_id"]
)
assert set(validation["question_id"]).isdisjoint(
    test["question_id"]
)

print("\nNo question ID overlap between splits.")
