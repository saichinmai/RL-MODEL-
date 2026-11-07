
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# File paths
INPUT_FILE = Path(
    "results/manual_annotation_audit_with_questions.csv"
)
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        "Manual audit CSV file not found."
    )

# Load data
df = pd.read_csv(INPUT_FILE)

required = [
    "human_verified_label",
    "predicted_label"
]

for col in required:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

# Convert labels to numeric
df["human_verified_label"] = pd.to_numeric(
    df["human_verified_label"],
    errors="coerce"
)

df["predicted_label"] = pd.to_numeric(
    df["predicted_label"],
    errors="coerce"
)

# Keep only completed and valid human reviews
valid = df.dropna(
    subset=["human_verified_label", "predicted_label"]
).copy()

valid = valid[
    valid["human_verified_label"].isin([0, 1])
    & valid["predicted_label"].isin([0, 1])
]

if len(valid) == 0:
    raise ValueError(
        "No completed human-verified labels found. "
        "Review and save the CSV first."
    )

y_true = valid["human_verified_label"].astype(int)
y_pred = valid["predicted_label"].astype(int)

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(
    y_true, y_pred, zero_division=0
)
recall = recall_score(
    y_true, y_pred, zero_division=0
)
f1 = f1_score(
    y_true, y_pred, zero_division=0
)

print("\nHUMAN ANNOTATION AUDIT RESULTS")
print("--------------------------------")
print("Reviewed sentences:", len(valid))
print("Unreviewed sentences:", len(df) - len(valid))

print("\nCLASSIFIER PERFORMANCE")
print("--------------------------------")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")

print("\nCLASSIFICATION REPORT")
print("--------------------------------")
print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=[
        "Potentially Irrelevant",
        "Relevant"
    ],
    zero_division=0
))

# Confusion matrix
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

print("\nCONFUSION MATRIX")
print(cm)

# Save confusion matrix image
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Irrelevant", "Relevant"]
)

display.plot(values_format="d")
plt.title("Human-Audited Classifier Evaluation")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "human_audit_confusion_matrix.png",
    dpi=300
)
plt.close()

# Save metrics
summary = pd.DataFrame([{
    "reviewed_sentences": len(valid),
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}])

summary.to_csv(
    OUTPUT_DIR / "human_audit_evaluation.csv",
    index=False
)

print("\nEVALUATION COMPLETED")
print("--------------------------------")
print("Saved: results/human_audit_evaluation.csv")
print("Saved: results/human_audit_confusion_matrix.png")
