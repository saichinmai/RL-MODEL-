
import pandas as pd
from pathlib import Path

from filtering import RelevanceFilter
from reasoner import ReasoningModel

QUESTIONS_FILE = Path("data/processed/questions.csv")
TEST_FILE = Path("data/processed/test.csv")
OUTPUT_DIR = Path("results")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

questions = pd.read_csv(QUESTIONS_FILE)
test = pd.read_csv(TEST_FILE)

# Select 20 test question IDs for the pilot
test_ids = test["question_id"].drop_duplicates().head(20)

relevance_filter = RelevanceFilter()
reasoner = ReasoningModel()

results = []

print("STARTING THREE-CONDITION PILOT")
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

    print(f"\nQuestion {count}/20")

    # A. Clean condition
    clean_answer = reasoner.answer(clean_question)

    # B. Distracted condition
    distracted_answer = reasoner.answer(distracted_question)

    # C. Filtered condition
    filtered_question, removed = (
        relevance_filter.filter_question(distracted_question)
    )

    filtered_answer = reasoner.answer(filtered_question)

    # Save all three conditions
    conditions = [
        ("clean", clean_question, clean_answer, []),
        (
            "distracted",
            distracted_question,
            distracted_answer,
            []
        ),
        (
            "filtered",
            filtered_question,
            filtered_answer,
            removed
        )
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

output_file = OUTPUT_DIR / "pilot_results.csv"

pd.DataFrame(results).to_csv(
    output_file,
    index=False
)

print("\nPILOT EXPERIMENT COMPLETED")
print("----------------------------")
print("Question pairs processed:", len(results) // 3)
print("Answer records:", len(results))
print("Saved:", output_file)
