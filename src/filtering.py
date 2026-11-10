
import joblib
import re
from pathlib import Path


# Find the model relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "relevance_classifier.pkl"


class RelevanceFilter:

    def __init__(self, threshold=0.50):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained classifier not found: {MODEL_PATH}"
            )

        self.model = joblib.load(MODEL_PATH)
        self.threshold = threshold

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

    def analyze_question(self, question):

        sentences = self.split_sentences(question)

        if not sentences:
            return [], question, []

        predictions = self.model.predict_proba(sentences)

        classes = list(self.model.classes_)

        if 1 not in classes:
            raise ValueError(
                "Classifier does not contain relevant class 1."
            )

        relevant_index = classes.index(1)

        filtered_sentences = []
        removed_sentences = []
        details = []

        for sentence, probabilities in zip(
            sentences, predictions
        ):

            score = float(probabilities[relevant_index])

            is_relevant = score >= self.threshold

            details.append({
                "sentence": sentence,
                "relevance_score": score,
                "classification": (
                    "Relevant" if is_relevant
                    else "Potentially irrelevant"
                )
            })

            if is_relevant:
                filtered_sentences.append(sentence)
            else:
                removed_sentences.append(sentence)

        # Safety: never return an empty question
        if not filtered_sentences:
            filtered_sentences = sentences.copy()
            removed_sentences = []

            for item in details:
                item["classification"] = "Retained (safety fallback)"

        filtered_question = " ".join(filtered_sentences)

        return details, filtered_question, removed_sentences

    def filter_question(self, question):

        details, filtered_question, removed = (
            self.analyze_question(question)
        )

        return filtered_question, removed


if __name__ == "__main__":

    relevance_filter = RelevanceFilter()

    question = (
        "Steve is 5 feet 6 inches tall. "
        "He grows 6 inches. "
        "The height of Emma is 8 feet. "
        "How tall is Steve in inches?"
    )

    details, filtered, removed = (
        relevance_filter.analyze_question(question)
    )

    print("\nSENTENCE CLASSIFICATION")

    for item in details:
        print(item)

    print("\nFILTERED QUESTION:")
    print(filtered)

    print("\nREMOVED SENTENCES:")
    print(removed)
