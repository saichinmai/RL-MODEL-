
import pandas as pd
from pathlib import Path

INPUT_FILE = Path("results/classifier_predictions.csv")
OUTPUT_FILE = Path("results/manual_annotation_audit.csv")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        "Run evaluate_classifier.py first."
    )

df = pd.read_csv(INPUT_FILE)

# Select samples from both classes
irrelevant = df[df["label"] == 0].sample(
    n=min(100, len(df[df["label"] == 0])),
    random_state=42
)

relevant = df[df["label"] == 1].sample(
    n=min(100, len(df[df["label"] == 1])),
    random_state=42
)

audit = pd.concat(
    [irrelevant, relevant],
    ignore_index=True
)

# Shuffle selected samples
audit = audit.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Prepare manual review columns
audit["human_verified_label"] = ""
audit["review_notes"] = ""

# Save audit file
audit.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nMANUAL ANNOTATION AUDIT CREATED")
print("--------------------------------")
print("Selected sentences:", len(audit))
print("Potentially irrelevant:", len(irrelevant))
print("Relevant:", len(relevant))
print("Saved:", OUTPUT_FILE)
