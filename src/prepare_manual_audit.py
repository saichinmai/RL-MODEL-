
import pandas as pd
from pathlib import Path

# File paths
QUESTIONS_FILE = Path("data/processed/questions.csv")
AUDIT_FILE = Path("results/manual_annotation_audit.csv")
OUTPUT_FILE = Path("results/manual_annotation_audit_with_questions.csv")

# Check files
if not QUESTIONS_FILE.exists():
    raise FileNotFoundError("questions.csv not found")

if not AUDIT_FILE.exists():
    raise FileNotFoundError("manual_annotation_audit.csv not found")

# Load datasets
questions = pd.read_csv(QUESTIONS_FILE)
audit = pd.read_csv(AUDIT_FILE)

# Required columns
required_questions = [
    "question_id",
    "question_type",
    "question"
]

required_audit = [
    "question_id",
    "question_type",
    "sentence",
    "label"
]

for col in required_questions:
    if col not in questions.columns:
        raise ValueError(f"Missing from questions.csv: {col}")

for col in required_audit:
    if col not in audit.columns:
        raise ValueError(f"Missing from audit file: {col}")

# Keep one original question per ID and type
question_context = questions[
    ["question_id", "question_type", "question"]
].drop_duplicates(
    subset=["question_id", "question_type"]
)

# Add original question to each audit row
updated = audit.merge(
    question_context,
    on=["question_id", "question_type"],
    how="left",
    validate="many_to_one"
)

# Check unmatched records
missing = updated["question"].isna().sum()

print("\nMANUAL AUDIT PREPARATION")
print("------------------------")
print("Audit sentences:", len(updated))
print("Questions matched:", len(updated) - missing)
print("Questions not matched:", missing)

if missing > 0:
    print("\nSome questions could not be matched.")
    print("Check question_id and question_type values.")

# Arrange useful columns first
preferred_columns = [
    "question_id",
    "question_type",
    "question",
    "sentence",
    "label",
    "predicted_label",
    "correct_prediction",
    "human_verified_label",
    "review_notes",
    "label_source"
]

columns = [
    col for col in preferred_columns
    if col in updated.columns
]

updated = updated[columns]

# Save without overwriting the original audit file
updated.to_csv(OUTPUT_FILE, index=False)

print("\nUpdated audit file saved:")
print(OUTPUT_FILE)
