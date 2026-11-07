
import pandas as pd
from pathlib import Path

RESULTS = Path("results")

classifier_file = RESULTS / "classifier_evaluation.csv"
answer_file = RESULTS / "final_evaluation_summary.csv"
audit_file = RESULTS / "human_audit_evaluation.csv"

rows = []

# 1. Classifier evaluation
if classifier_file.exists():
    df = pd.read_csv(classifier_file)
    r = df.iloc[0]

    rows.append({
        "evaluation": "Relevance Classifier",
        "condition": "Test dataset",
        "total_records": r.get("test_records", ""),
        "accuracy": r.get("accuracy", ""),
        "precision": r.get("precision", ""),
        "recall": r.get("recall", ""),
        "f1_score": r.get("f1_score", "")
    })

# 2. Mathematical answer evaluation
if answer_file.exists():
    df = pd.read_csv(answer_file)

    for _, r in df.iterrows():
        rows.append({
            "evaluation": "Mathematical Answer Evaluation",
            "condition": r["condition"],
            "total_records": r["total_questions"],
            "accuracy": r["accuracy"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1_score": r["f1_score"]
        })

# 3. Annotation audit
if audit_file.exists():
    df = pd.read_csv(audit_file)
    r = df.iloc[0]

    rows.append({
        "evaluation": "Annotation Audit",
        "condition": "Reviewed subset",
        "total_records": r["reviewed_sentences"],
        "accuracy": r["accuracy"],
        "precision": r["precision"],
        "recall": r["recall"],
        "f1_score": r["f1_score"]
    })

# Save complete comparison
if not rows:
    raise FileNotFoundError("No evaluation files found.")

result = pd.DataFrame(rows)

output = RESULTS / "overall_model_comparison.csv"
result.to_csv(output, index=False)

print("\nCOMPLETE MODEL RESULTS")
print("======================")
print(result.to_string(index=False))

print("\nSaved:", output)
