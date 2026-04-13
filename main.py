
import os
import re
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    classification_report
)


# STEP 1 — LOAD DATASET

DATA_PATH = os.path.join("data", "tweets_dataset.csv")
df = pd.read_csv(DATA_PATH)

print("=" * 62)
print("  SENTIMENT ANALYSIS — CLAUDE 4 (ANTHROPIC) TWEETS")
print("=" * 62)
print(f"\nDataset loaded: {len(df)} tweets")
print(f"\nSentiment Distribution:\n{df['sentiment'].value_counts().to_string()}")


# STEP 2 — PREPROCESSING

def preprocess_tweet(text):
    """
    Cleans raw tweet text for NLP processing.
    Steps:
      1. Lowercase all text
      2. Remove URLs
      3. Remove @mentions
      4. Remove #hashtags
      5. Remove special characters, numbers, emojis
      6. Normalize whitespace
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_tweet"] = df["tweet"].apply(preprocess_tweet)
print("\nPreprocessing complete.")


# STEP 3 — LABEL ENCODING

label_map    = {"positive": 2, "neutral": 1, "negative": 0}
label_decode = {v: k for k, v in label_map.items()}
df["label"]  = df["sentiment"].map(label_map)


# STEP 4 — TRAIN / TEST SPLIT  (80 / 20)

X = df["clean_tweet"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y          # Preserve class distribution
)

print(f"\nTrain/Test Split:")
print(f"  Training set : {len(X_train)} tweets (80%)")
print(f"  Testing set  : {len(X_test)} tweets  (20%)")


# STEP 5 — TF-IDF VECTORIZATION

vectorizer = TfidfVectorizer(
    max_features=500,
    ngram_range=(1, 2),     # Unigrams + Bigrams
    stop_words="english",
    sublinear_tf=True        # Log normalization
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

print(f"\nTF-IDF vectorization complete.")
print(f"  Vocabulary size : {len(vectorizer.vocabulary_)} features")
print(f"  N-gram range    : (1, 2)")


# STEP 6 — TRAIN CLASSIFIERS & EVALUATE

classifiers = {
    "Naive Bayes"        : MultinomialNB(alpha=0.5),
    "SVM (LinearSVC)"    : LinearSVC(C=0.5, max_iter=2000, random_state=42),
    "Logistic Regression": LogisticRegression(C=0.5, max_iter=1000,
                                               solver="lbfgs", random_state=42),
}

target_names = ["Negative", "Neutral", "Positive"]
results      = {}

print("\n" + "=" * 62)
print("  MODEL TRAINING & EVALUATION RESULTS")
print("=" * 62)

for name, clf in classifiers.items():
    clf.fit(X_train_tfidf, y_train)
    y_pred = clf.predict(X_test_tfidf)

    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    results[name] = {"precision": precision, "recall": recall,
                     "f1": f1, "y_pred": y_pred}

    print(f"\n{'─' * 62}")
    print(f"  Classifier : {name}")
    print(f"{'─' * 62}")
    print(f"  Weighted Precision : {precision:.4f}")
    print(f"  Weighted Recall    : {recall:.4f}")
    print(f"  Weighted F1-Score  : {f1:.4f}")
    print()
    print(classification_report(y_test, y_pred,
                                 target_names=target_names,
                                 zero_division=0))


# STEP 7 — COMPARISON SUMMARY

print("=" * 62)
print("  CLASSIFIER COMPARISON SUMMARY")
print("=" * 62)
print(f"\n{'Classifier':<25} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
print("-" * 57)
for name, m in results.items():
    print(f"{name:<25} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")

best = max(results, key=lambda x: results[x]["f1"])
print(f"\n  Best Classifier : {best}")
print(f"  Best F1-Score   : {results[best]['f1']:.4f}")


# STEP 8 — SAMPLE PREDICTIONS ON TEST SET

print(f"\n{'─' * 62}")
print(f"  SAMPLE PREDICTIONS — {best}")
print(f"{'─' * 62}")

test_df = df.loc[y_test.index].copy().reset_index(drop=True)
test_df["predicted_label"] = [label_decode[p] for p in results[best]["y_pred"]]
test_df["correct"]         = test_df["sentiment"] == test_df["predicted_label"]

print(f"\n{'#':<4} {'Tweet':<55} {'Actual':<10} {'Pred':<10} {'OK'}")
print("-" * 85)
for i, row in test_df.head(15).iterrows():
    short = row["tweet"][:52] + "..." if len(row["tweet"]) > 52 else row["tweet"]
    tick  = "OK" if row["correct"] else "X"
    print(f"{i+1:<4} {short:<55} {row['sentiment']:<10} {row['predicted_label']:<10} {tick}")

accuracy = test_df["correct"].mean() * 100
print(f"\n  Test Accuracy : {accuracy:.1f}%  ({int(test_df['correct'].sum())}/20 correct)")
print("\n" + "=" * 62)
print("  DONE")
print("=" * 62)