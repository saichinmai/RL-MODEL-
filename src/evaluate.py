
import pandas as pd
import re
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==================================================
# 1. FILE PATHS
# ==================================================

INPUT_FILE = Path("results/final_results.csv")
OUTPUT_DIR = Path("results")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = OUTPUT_DIR / "final_evaluation_summary.csv"
DETAIL_FILE = OUTPUT_DIR / "final_detailed_evaluation.csv"
GRAPH_FILE = OUTPUT_DIR / "final_accuracy_comparison.png"
METRICS_GRAPH = OUTPUT_DIR / "final_metrics_comparison.png"


# ==================================================
# 2. CHECK INPUT FILE
# ==================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        "final_results.csv not found. "
        "Run src/run_final_experiment.py first."
    )

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "question_id",
    "condition",
    "reference_answer",
    "model_answer"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("\nFINAL EXPERIMENT EVALUATION")
print("============================")

print("Total answer records:", len(df))
print("Question pairs:", df["question_id"].nunique())


# ==================================================
# 3. EXTRACT NUMERICAL ANSWERS
# ==================================================

def extract_answer(text):

    if pd.isna(text):
        return None

    text = str(text).replace(",", "")

    # Prefer the explicitly stated final answer
    patterns = [
        r"Final\s*Answer\s*:\s*([-+]?\d+(?:\.\d+)?)",
        r"Final\s*Answer\s+is\s+([-+]?\d+(?:\.\d+)?)"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            return float(match.group(1))

    # Fallback: use the last number in the response
    numbers = re.findall(
        r"[-+]?\d+(?:\.\d+)?",
        text
    )

    if not numbers:
        return None

    try:
        return float(numbers[-1])
    except ValueError:
        return None


df["predicted_answer"] = df["model_answer"].apply(
    extract_answer
)

df["correct_answer"] = df["reference_answer"].apply(
    extract_answer
)


# ==================================================
# 4. CALCULATE ANSWER CORRECTNESS
# ==================================================

df["answer_evaluated"] = (
    df["predicted_answer"].notna()
    & df["correct_answer"].notna()
)

df["correct"] = (
    df["answer_evaluated"]
    & (
        (
            df["predicted_answer"]
            - df["correct_answer"]
        ).abs() < 1e-9
    )
)

# Show answers that could not be evaluated
unevaluable = df[~df["answer_evaluated"]]

print("\nUnevaluable answer records:", len(unevaluable))


# ==================================================
# 5. CALCULATE METRICS FOR EACH CONDITION
# ==================================================

conditions = [
    "clean",
    "distracted",
    "filtered"
]

summary_records = []

for condition in conditions:

    subset = df[df["condition"] == condition].copy()

    if subset.empty:
        print(f"Warning: No records for {condition}")
        continue

    # Only evaluate answers with extractable reference
    # and predicted numerical answers.
    valid = subset[subset["answer_evaluated"]].copy()

    if valid.empty:
        print(f"Warning: No evaluable answers for {condition}")
        continue

    y_true = valid["correct_answer"]
    y_pred = valid["predicted_answer"]

    # Accuracy: proportion of mathematically correct answers
    accuracy = accuracy_score(
        valid["correct"],
        [True] * len(valid)
    )

    # For answer correctness, define incorrect=0, correct=1.
    y_actual = valid["correct"].astype(int)
    y_positive = [1] * len(valid)

    precision = precision_score(
        y_actual,
        y_positive,
        zero_division=0
    )

    recall = recall_score(
        y_actual,
        y_positive,
        zero_division=0
    )

    f1 = f1_score(
        y_actual,
        y_positive,
        zero_division=0
    )

    summary_records.append({
        "condition": condition,
        "total_questions": subset["question_id"].nunique(),
        "evaluated_answers": len(valid),
        "unevaluable_answers": len(subset) - len(valid),
        "correct_answers": int(valid["correct"].sum()),
        "incorrect_answers": int((~valid["correct"]).sum()),
        "accuracy": accuracy,
        "accuracy_percentage": accuracy * 100,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })


# ==================================================
# 6. SAVE EVALUATION SUMMARY
# ==================================================

summary = pd.DataFrame(summary_records)

if summary.empty:
    raise ValueError(
        "No valid evaluation results were generated."
    )

summary.to_csv(SUMMARY_FILE, index=False)

print("\nFINAL EVALUATION SUMMARY")
print("============================")

print(
    summary[
        [
            "condition",
            "total_questions",
            "correct_answers",
            "accuracy_percentage",
            "precision",
            "recall",
            "f1_score"
        ]
    ].to_string(index=False)
)


# ==================================================
# 7. GENERATE CONFUSION MATRICES
# ==================================================

print("\nANSWER CORRECTNESS CONFUSION MATRICES")
print("======================================")

for condition in conditions:

    subset = df[
        (df["condition"] == condition)
        & df["answer_evaluated"]
    ]

    if subset.empty:
        continue

    matrix = confusion_matrix(
        subset["correct"].astype(int),
        [1] * len(subset),
        labels=[0, 1]
    )

    print(f"\n{condition.upper()}")
    print(matrix)


# ==================================================
# 8. SAVE DETAILED RESULTS
# ==================================================

df.to_csv(DETAIL_FILE, index=False)


# ==================================================
# 9. ACCURACY COMPARISON GRAPH
# ==================================================

plt.figure(figsize=(9, 5))

plt.bar(
    summary["condition"],
    summary["accuracy_percentage"]
)

plt.title(
    "Mathematical Answer Accuracy Comparison"
)

plt.xlabel("Experimental Condition")
plt.ylabel("Accuracy (%)")

plt.ylim(0, 100)

plt.tight_layout()

plt.savefig(
    GRAPH_FILE,
    dpi=300
)

plt.close()


# ==================================================
# 10. PRECISION, RECALL AND F1 COMPARISON GRAPH
# ==================================================

plt.figure(figsize=(9, 5))

metrics = [
    "precision",
    "recall",
    "f1_score"
]

for metric in metrics:

    plt.plot(
        summary["condition"],
        summary[metric],
        marker="o",
        label=metric.replace("_", " ").title()
    )

plt.title("Evaluation Metrics Comparison")

plt.xlabel("Experimental Condition")
plt.ylabel("Score")

plt.ylim(0, 1.05)

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    METRICS_GRAPH,
    dpi=300
)

plt.close()


# ==================================================
# 11. FINAL REPORT
# ==================================================

print("\nEVALUATION COMPLETED")
print("============================")

print("Summary:", SUMMARY_FILE)
print("Detailed results:", DETAIL_FILE)
print("Accuracy graph:", GRAPH_FILE)
print("Metrics graph:", METRICS_GRAPH)

print("\nAll evaluation files saved successfully!")
