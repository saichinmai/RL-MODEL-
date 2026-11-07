  
import pandas as pd
import joblib

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# File paths
TRAIN_FILE = Path("data/processed/train.csv")
VAL_FILE = Path("data/processed/validation.csv")
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load datasets
print("Loading training and validation datasets...")

train = pd.read_csv(TRAIN_FILE)
validation = pd.read_csv(VAL_FILE)

X_train = train["sentence"].fillna("").astype(str)
y_train = train["label"].astype(int)

X_val = validation["sentence"].fillna("").astype(str)
y_val = validation["label"].astype(int)

# Check that both classes are available
if set(y_train.unique()) != {0, 1}:
    raise ValueError("Training data must contain labels 0 and 1.")

# Build TF-IDF + Logistic Regression pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=50000,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
    )
])

# Train classifier
print("\nTraining TF-IDF + Logistic Regression...")

model.fit(X_train, y_train)

print("Training completed!")

# Validate model
print("\nEvaluating on validation dataset...")

predictions = model.predict(X_val)

accuracy = accuracy_score(y_val, predictions)
precision = precision_score(
    y_val, predictions, zero_division=0
)
recall = recall_score(
    y_val, predictions, zero_division=0
)
f1 = f1_score(
    y_val, predictions, zero_division=0
)

print("\nVALIDATION RESULTS")
print("-------------------------")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_val,
        predictions,
        target_names=[
            "Potentially Irrelevant",
            "Relevant"
        ],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, predictions))

# Save trained model
model_path = MODEL_DIR / "relevance_classifier.pkl"

joblib.dump(model, model_path)

print("\nModel saved successfully!")
print("Location:", model_path)
