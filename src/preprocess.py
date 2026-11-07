
import pandas as pd
import re
from pathlib import Path

INPUT_FILE = Path("data/processed/questions.csv")
OUTPUT_FOLDER = Path("data/processed/annotations")

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


def split_sentences(text):
    text = re.sub(r"\s+", " ", str(text)).strip()

    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]


def clean_text(text):
    text = re.sub(r"\s+", " ", str(text).strip())
    return text.lower().rstrip(" .!?")


def identify_distractor(sentence, template, role, number):

    if pd.isna(template) or pd.isna(role) or pd.isna(number):
        return False

    # Convert values to strings before filling the template
    role = str(role).strip()

    if isinstance(number, float) and number.is_integer():
        number = str(int(number))
    else:
        number = str(number).strip()

    try:
        expected_sentence = str(template).format(
            role=role,
            number=number
        )

        return clean_text(sentence) == clean_text(expected_sentence)

    except (KeyError, ValueError, IndexError):
        return False


df = pd.read_csv(INPUT_FILE)

records = []

for _, row in df.iterrows():

    sentences = split_sentences(row["question"])

    for sentence in sentences:

        if row["question_type"] == "clean":

            label = 1
            source = "original_question"

        else:

            is_distractor = identify_distractor(
                sentence,
                row.get("sentence_template"),
                row.get("role"),
                row.get("number")
            )

            if is_distractor:
                label = 0
                source = "matched_distractor_template"
            else:
                label = 1
                source = "provisionally_relevant"

        records.append({
            "question_id": row["question_id"],
            "question_type": row["question_type"],
            "sentence": sentence,
            "label": label,
            "label_source": source
        })


result = pd.DataFrame(records)

output_file = OUTPUT_FOLDER / "sentence_candidates.csv"
result.to_csv(output_file, index=False)

print("\nANNOTATION PROCESS COMPLETED")
print("--------------------------------")

print("Total sentences:", len(result))

print("\nLabel distribution:")
print(result["label"].value_counts())

print("\nLabel sources:")
print(result["label_source"].value_counts())

print("\nNumber of matched distractors:",
      (result["label_source"] == "matched_distractor_template").sum())

print("\nSample irrelevant sentences:")

print(
    result[result["label"] == 0]
    .head(10)
    .to_string(index=False)
)

print("\nSaved file:", output_file)
