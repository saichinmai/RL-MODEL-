
import json
import pandas as pd
from pathlib import Path

# Dataset locations
DATA_FOLDER = Path("GSM-IC")
OUTPUT_FOLDER = Path("data/processed")

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

files = [
    "GSM-IC_2step.json",
    "GSM-IC_mstep.json"
]

all_records = []

for filename in files:

    file_path = DATA_FOLDER / filename

    print(f"\nReading {filename}...")

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    for index, item in enumerate(data):

        question_id = f"{filename}_{index}"

        # Original question
        all_records.append({
            "question_id": question_id,
            "question_type": "clean",
            "question": item["original_question"],
            "answer": item["answer"],
            "n_steps": item.get("n_steps"),
            "role": item.get("role"),
            "number": item.get("number"),
            "sentence_template": item.get("sentence_template"),
            "role_label": item.get("role_label"),
            "number_label": item.get("number_label"),
            "sentence_label": item.get("sentence_label")
        })

        # Distracted question
        all_records.append({
            "question_id": question_id,
            "question_type": "distracted",
            "question": item["new_question"],
            "answer": item["answer"],
            "n_steps": item.get("n_steps"),
            "role": item.get("role"),
            "number": item.get("number"),
            "sentence_template": item.get("sentence_template"),
            "role_label": item.get("role_label"),
            "number_label": item.get("number_label"),
            "sentence_label": item.get("sentence_label")
        })

    print("Records loaded:", len(data))

# Convert records into DataFrame
df = pd.DataFrame(all_records)

# Save processed dataset
output_file = OUTPUT_FOLDER / "questions.csv"
df.to_csv(output_file, index=False)

print("\nDATASET CREATION COMPLETED")
print("--------------------------------")
print("Total rows:", len(df))
print("Unique question pairs:", df["question_id"].nunique())
print("\nQuestion types:")
print(df["question_type"].value_counts())

print("\nSaved to:", output_file)
print("\nFirst five records:")
print(df.head())
