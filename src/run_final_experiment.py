
import pandas as pd
from pathlib import Path

from filtering import RelevanceFilter
from reasoner import ReasoningModel

QUESTIONS_FILE = Path("data/processed/questions.csv")
TEST_FILE = Path("data/processed/test.csv")
OUTPUT_FILE = Path("results/final_results.csv")

questions = pd.read_csv(QUESTIONS_FILE)
test = pd.read_csv(TEST_FILE)

# Select 100 reproducible random test question IDs
test_ids = (
    test["question_id"]
    .drop_duplicates()
    .sample(
        n=min(100, test["question_id"].nunique()),
        random_state=42
    )
    .tolist()
)

relevance_filter = RelevanceFilter()
reasoner = ReasoningModel()

results = []

print("FINAL THREE-CONDITION EXPERIMENT")
print("--------------------------------")

for count, question_id in enumerate(test_ids, start=1):

    pair = questions[
        questions["question_id"] == question_id
    ]

    clean_rows = pair[pair["question_type"] == "clean"]
    distracted_rows = pair[
        pair["question_type"] == "distracted"
    ]

    if clean_rows.empty or distracted_rows.empty:
        print("Skipping:", question_id)
        continue

    clean = clean_rows.iloc[0]
    distracted = distracted_rows.iloc[0]

    clean_question = str(clean["question"])
    distracted_question = str(distracted["question"])
    reference_answer = clean["answer"]

    print(f"\nProcessing {count}/{len(test_ids)}")

    # Condition 1: Clean
    clean_answer = reasoner.answer(clean_question)

    # Condition 2: Distracted
    distracted_answer = reasoner.answer(distracted_question)

    # Condition 3: Filtered
    filtered_question, removed = (
        relevance_filter.filter_question(distracted_question)
    )

    filtered_answer = reasoner.answer(filtered_question)

    conditions = [
        ("clean", clean_question, clean_answer, []),
        ("distracted", distracted_question, distracted_answer, []),
        ("filtered", filtered_question, filtered_answer, removed)
    ]

    for condition, input_question, answer, removed_sentences in conditions:

        results.append({
            "question_id": question_id,
            "condition": condition,
            "input_question": input_question,
            "reference_answer": reference_answer,
            "model_answer": answer,
            "removed_sentences": " | ".join(removed_sentences)
        })

    print("Completed:", question_id)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

pd.DataFrame(results).to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nFINAL EXPERIMENT COMPLETED")
print("----------------------------")
print("Question pairs processed:", len(results) // 3)
print("Total answer records:", len(results))
print("Saved:", OUTPUT_FILE)
