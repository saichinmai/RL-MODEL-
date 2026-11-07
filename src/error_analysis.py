
import pandas as pd
from pathlib import Path

INPUT_FILE = Path("results/detailed_evaluation.csv")
OUTPUT_DIR = Path("results")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        "Run src/evaluate.py first to generate detailed_evaluation.csv"
    )

df = pd.read_csv(INPUT_FILE)

# Separate the three experimental conditions
clean = df[df["condition"] == "clean"].copy()
distracted = df[df["condition"] == "distracted"].copy()
filtered = df[df["condition"] == "filtered"].copy()

# Compare results using question IDs
comparison = clean[
    ["question_id", "correct"]
].rename(columns={"correct": "clean_correct"})

comparison = comparison.merge(
    distracted[["question_id", "correct"]].rename(
        columns={"correct": "distracted_correct"}
    ),
    on="question_id"
)

comparison = comparison.merge(
    filtered[["question_id", "correct"]].rename(
        columns={"correct": "filtered_correct"}
    ),
    on="question_id"
)

# Identify improvement and degradation
comparison["filtering_improved"] = (
    (~comparison["distracted_correct"])
    & comparison["filtered_correct"]
)

comparison["filtering_harmed"] = (
    comparison["distracted_correct"]
    & (~comparison["filtered_correct"])
)

comparison["both_correct"] = (
    comparison["distracted_correct"]
    & comparison["filtered_correct"]
)

comparison["both_wrong"] = (
    (~comparison["distracted_correct"])
    & (~comparison["filtered_correct"])
)

# Print analysis
print("\nFILTERING ERROR ANALYSIS")
print("----------------------------")

print("Question pairs:", len(comparison))

print(
    "Filtering improved:",
    comparison["filtering_improved"].sum()
)

print(
    "Filtering harmed:",
    comparison["filtering_harmed"].sum()
)

print(
    "Both conditions correct:",
    comparison["both_correct"].sum()
)

print(
    "Both conditions incorrect:",
    comparison["both_wrong"].sum()
)

# Save question-level comparison
comparison.to_csv(
    OUTPUT_DIR / "condition_comparison.csv",
    index=False
)

# Extract cases where filtering caused an incorrect answer
harmful_ids = comparison.loc[
    comparison["filtering_harmed"],
    "question_id"
]

harmful_cases = filtered[
    filtered["question_id"].isin(harmful_ids)
].copy()

# Extract cases where filtering improved the answer
improved_ids = comparison.loc[
    comparison["filtering_improved"],
    "question_id"
]

improved_cases = filtered[
    filtered["question_id"].isin(improved_ids)
].copy()

harmful_cases.to_csv(
    OUTPUT_DIR / "filtering_harmed_cases.csv",
    index=False
)

improved_cases.to_csv(
    OUTPUT_DIR / "filtering_improved_cases.csv",
    index=False
)

# Save all filtered sentences for manual inspection
removed_sentences = filtered[
    ["question_id", "input_question", "removed_sentences"]
].copy()

removed_sentences.to_csv(
    OUTPUT_DIR / "removed_sentence_audit.csv",
    index=False
)

print("\nFILES GENERATED")
print("----------------------------")
print("condition_comparison.csv")
print("filtering_harmed_cases.csv")
print("filtering_improved_cases.csv")
print("removed_sentence_audit.csv")

print("\nERROR ANALYSIS COMPLETED")
