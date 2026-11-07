
import pandas as pd
import joblib
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

# -----------------------------------------
# FILE PATHS
# -----------------------------------------

TEST_FILE = Path("data/processed/test.csv")
MODEL_FILE = Path("models/relevance_classifier.pkl")
OUTPUT_DIR = Path("results")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------
# LOAD TEST DATA
# -----------------------------------------

if not TEST_FILE.exists():
    raise FileNotFoundError("test.csv not found")

if not MODEL_FILE.exists():
    raise FileNotFoundError("Trained classifier not found")

test = pd.read_csv(TEST_FILE)

required_columns = ["sentence", "label"]

for column in required_columns:
    if column not in test.columns:
        raise ValueError(
            f"Missing column in test.csv: {column}"
        )

X_test = test["sentence"].fillna("").astype(str)
y_test = test["label"].astype(int)

# -----------------------------------------
# LOAD TRAINED MODEL
# -----------------------------------------

model = joblib.load(MODEL_FILE)

# -----------------------------------------
# PREDICT TEST LABELS
# -----------------------------------------

print("\nTESTING RELEVANCE CLASSIFIER")
print("----------------------------")

y_pred = model.predict(X_test)

# -----------------------------------------
# CALCULATE METRICS
# -----------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, zero_division=0
)

recall = recall_score(
    y_test, y_pred, zero_division=0
)

f1 = f1_score(
    y_test, y_pred, zero_division=0
)

print("\nCLASSIFICATION RESULTS")
print("----------------------------")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")

print("\nCLASSIFICATION REPORT")
print("----------------------------")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=[
            "Potentially Irrelevant",
            "Relevant"
        ],
        zero_division=0
    )
)

# -----------------------------------------
# CONFUSION MATRIX
# -----------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

print("\nCONFUSION MATRIX")
print("----------------------------")
print(cm)

# -----------------------------------------
# SAVE CONFUSION MATRIX GRAPH
# -----------------------------------------

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Irrelevant",
        "Relevant"
    ]
)

display.plot(
    cmap="Blues",
    values_format="d"
)

plt.title("Relevance Classifier Confusion Matrix")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "classifier_confusion_matrix.png",
    dpi=300
)

plt.close()

# -----------------------------------------
# SAVE METRICS
# -----------------------------------------

metrics = pd.DataFrame([{
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "test_records": len(test)
}])

metrics.to_csv(
    OUTPUT_DIR / "classifier_evaluation.csv",
    index=False
)

# -----------------------------------------
# SAVE PREDICTIONS
# -----------------------------------------

prediction_results = test.copy()

prediction_results["predicted_label"] = y_pred

prediction_results["correct_prediction"] = (
    prediction_results["label"]
    == prediction_results["predicted_label"]
)

prediction_results.to_csv(
    OUTPUT_DIR / "classifier_predictions.csv",
    index=False
)

print("\nCLASSIFIER EVALUATION COMPLETED")
print("----------------------------")
print("Saved: results/classifier_evaluation.csv")
print("Saved: results/classifier_predictions.csv")
print("Saved: results/classifier_confusion_matrix.png")
