
import joblib
import re
from pathlib import Path

MODEL_PATH = Path("models/relevance_classifier.pkl")


class RelevanceFilter:

    def __init__(self):
        self.model = joblib.load(MODEL_PATH)

    def split_sentences(self, question):

        question = re.sub(r"\s+", " ", question.strip())

        return [
            sentence.strip()
            for sentence in re.split(
                r"(?<=[.!?])\s+",
                question
            )
            if sentence.strip()
        ]

    def filter_question(self, question):

        sentences = self.split_sentences(question)

        predictions = self.model.predict_proba(sentences)

        classes = list(self.model.classes_)
        relevant_index = classes.index(1)

        filtered_sentences = []
        removed_sentences = []

        print("\nSENTENCE CLASSIFICATION")
        print("----------------------------")

        for sentence, probabilities in zip(
            sentences, predictions
        ):

            score = probabilities[relevant_index]

            if score >= 0.50:

                filtered_sentences.append(sentence)

                print(f"RELEVANT: {sentence}")

            else:

                removed_sentences.append(sentence)

                print(f"POTENTIALLY IRRELEVANT: {sentence}")

        # Safety: do not return an empty question
        if not filtered_sentences:
            filtered_sentences = sentences.copy()
            removed_sentences = []

        filtered_question = " ".join(filtered_sentences)

        return filtered_question, removed_sentences


if __name__ == "__main__":

    relevance_filter = RelevanceFilter()

    question = (
        "Steve is 5'6. "
        "He grows 6 inches. "
        "The height of Emma is 8 feet. "
        "How tall is Steve in inches?"
    )

    filtered, removed = relevance_filter.filter_question(
        question
    )

    print("\nORIGINAL QUESTION:")
    print(question)

    print("\nFILTERED QUESTION:")
    print(filtered)

    print("\nREMOVED SENTENCES:")
    print(removed)
