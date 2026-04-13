# Sentiment Analysis on Claude 4 (Anthropic) Tweets

**Course:** Data Analytics and Visualisation (CSC601)  
**Module:** 4 — Text Analytics  
**College:** Rizvi College of Engineering, Mumbai  
**Department:** Artificial Intelligence & Data Science  
**Class:** TE AI&DS (2025-26) | Semester VI  
**Submission Date:** 13/04/2026  

---

## (1) Problem Statement

Choose a topic of interest, collect 100 tweets, hand-tag them as positive / neutral / negative, split into 80% training and 20% testing, run one or more classifiers for sentiment analysis, and calculate precision and recall for each classifier.

---

## (2) Topic — Claude 4 (Anthropic)

**Claude 4** is the latest family of large language models released by Anthropic. It comes in three tiers — **Haiku** (fast, low-cost), **Sonnet** (balanced), and **Opus** (maximum capability) — and is built on Anthropic's *Constitutional AI* framework, which trains the model to be simultaneously helpful, harmless, and honest.

Upon release, Claude 4 triggered intense public debate on Twitter/X covering model capability, AI safety, pricing, over-censorship, and comparisons with GPT-4o and Gemini Ultra.

**Why Claude 4?**  
It is a genuinely fresh buzzword in AI discourse with rich and diverse public sentiment — enthusiastic praise from developers alongside sharp criticism from open-source advocates — making it ideal for a 3-class sentiment classification task.

---

## (3) Repository Structure

```
74_MuhammedSaadZaveri_Assignment2/
│
├── data/
│   └── tweets_dataset.csv       ← 100 manually labelled tweets
│
├── notebooks/
│   └── sentiment_analysis.ipynb ← Full analysis with visualisations
│
├── main.py                      ← Standalone Python pipeline
├── README.md                    ← This report
└── requirements.txt             ← Python dependencies
```

---

## (4) Introduction

The goal of this assignment is to perform **sentiment analysis** on tweets related to **Claude 4**, Anthropic's latest AI model. We collect 100 tweets, manually annotate them with sentiment labels, and train three classifiers — Naïve Bayes, SVM, and Logistic Regression — to predict sentiment on unseen tweets. We then evaluate each classifier using precision, recall, and F1-score.

---

## (5) Data Collection & Preprocessing

### 5.1 Data Collection

| Detail | Value |
|--------|-------|
| Topic | Claude 4 (Anthropic) |
| Total tweets | 100 |
| Collection method | Manual collection of representative tweets |
| Keywords used | `Claude 4`, `Anthropic`, `Claude Opus`, `Claude Sonnet`, `Claude Haiku` |

### 5.2 Sentiment Distribution

| Sentiment | Count | Percentage |
|-----------|-------|------------|
| Neutral   | 40    | 40%        |
| Positive  | 30    | 30%        |
| Negative  | 30    | 30%        |

### 5.3 Manual Tagging Criteria

| Label | Criteria |
|-------|----------|
| **Positive** | Praise, excitement, gratitude, or satisfaction toward Claude 4 |
| **Neutral**  | Factual statements, news, announcements — no emotional tone |
| **Negative** | Criticism, frustration, fear, or disappointment about Claude 4 |

### 5.4 Preprocessing Pipeline

```python
def preprocess_tweet(text):
    text = text.lower()                          # 1. Lowercase
    text = re.sub(r"http\S+|www\S+", "", text)  # 2. Remove URLs
    text = re.sub(r"@\w+", "", text)             # 3. Remove @mentions
    text = re.sub(r"#\w+", "", text)             # 4. Remove #hashtags
    text = re.sub(r"[^a-z\s]", "", text)        # 5. Remove special chars & emojis
    text = re.sub(r"\s+", " ", text).strip()     # 6. Normalize whitespace
    return text
```

---

## (6) Data Splitting

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
```

| Set | Size |
|-----|------|
| Training set | 80 tweets (80%) |
| Testing set  | 20 tweets (20%) |

`stratify=y` ensures class ratios are preserved in both sets. `random_state=42` guarantees reproducibility.

---

## (7) Feature Extraction — TF-IDF

```python
vectorizer = TfidfVectorizer(
    max_features=500,
    ngram_range=(1, 2),    # Unigrams + Bigrams
    stop_words="english",
    sublinear_tf=True       # Log normalization
)
```

TF-IDF was chosen over Bag-of-Words because it penalizes overly common words and rewards informative sentiment-carrying terms. Bigrams capture phrases like *"not helpful"* or *"genuinely impressive"* that carry sentiment lost at the unigram level.

---

## (8) Model Training & Classification

### Classifiers Used

| Classifier | Class | Key Hyperparameters |
|---|---|---|
| **Naïve Bayes** | `MultinomialNB` | `alpha=0.5` |
| **SVM** | `LinearSVC` | `C=0.5`, `max_iter=2000` |
| **Logistic Regression** | `LogisticRegression` | `C=0.5`, `solver=lbfgs` |

```python
classifiers = {
    "Naive Bayes"        : MultinomialNB(alpha=0.5),
    "SVM (LinearSVC)"    : LinearSVC(C=0.5, max_iter=2000, random_state=42),
    "Logistic Regression": LogisticRegression(C=0.5, max_iter=1000,
                                               solver="lbfgs", random_state=42),
}
for name, clf in classifiers.items():
    clf.fit(X_train_tfidf, y_train)
    y_pred = clf.predict(X_test_tfidf)
```

---

## (9) Model Evaluation — Precision & Recall

### Naïve Bayes

| Class    | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Negative | 0.25      | 0.17   | 0.20     | 6       |
| Neutral  | 0.67      | 1.00   | 0.80     | 8       |
| Positive | 0.50      | 0.33   | 0.40     | 6       |
| **Weighted Avg** | **0.49** | **0.55** | **0.50** | 20 |

### SVM (LinearSVC)

| Class    | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Negative | 0.33      | 0.17   | 0.22     | 6       |
| Neutral  | 0.57      | 1.00   | 0.73     | 8       |
| Positive | 0.67      | 0.33   | 0.44     | 6       |
| **Weighted Avg** | **0.53** | **0.55** | **0.49** | 20 |

### Logistic Regression

| Class    | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Negative | 0.00      | 0.00   | 0.00     | 6       |
| Neutral  | 0.42      | 1.00   | 0.59     | 8       |
| Positive | 0.00      | 0.00   | 0.00     | 6       |
| **Weighted Avg** | **0.17** | **0.40** | **0.24** | 20 |

---

## (10) Classifier Comparison & Conclusion

### Summary Table

| Classifier | Weighted Precision | Weighted Recall | Weighted F1-Score |
|---|---|---|---|
| **Naïve Bayes** | 0.4917 | **0.5500** | **0.5000** ✅ |
| SVM (LinearSVC) | **0.5286** | **0.5500** | 0.4909 |
| Logistic Regression | 0.1684 | 0.4000 | 0.2370 |

### 🏆 Best Classifier: Naïve Bayes (F1 = 0.5000)

**Naïve Bayes achieves the highest weighted F1-score** of 0.50 with 55% test accuracy, making it the best classifier for this dataset.

**Why Naïve Bayes wins:**
- With only 80 training samples, its probabilistic priors prevent overfitting to noise
- It performs exceptionally well on sparse, high-dimensional TF-IDF feature vectors
- Achieves perfect recall (1.00) on the Neutral class, which has the most support
- Conditional independence assumption is reasonably valid for bag-of-words text features

**Why SVM is close:**  
SVM achieves higher precision (0.53) but a slightly lower F1 (0.49). With more training data, SVM would likely surpass Naïve Bayes as it is more discriminative in high-dimensional spaces.

**Why Logistic Regression underperforms:**  
With ~24 negative training samples, LR's optimization fails to reliably separate Negative from Neutral, resulting in 0.00 precision/recall for both Negative and Positive classes.

### Possible Improvements

1. **More data** — 1,000+ tweets would dramatically improve all classifiers
2. **BERT embeddings** — Pre-trained transformers capture semantic nuance that TF-IDF misses
3. **Hyperparameter tuning** — GridSearchCV to optimize `C`, `alpha`, `ngram_range`
4. **Data augmentation** — Back-translation to synthetically balance minority classes
5. **Ensemble voting** — Combining all three classifiers for more robust predictions

---

## (11) How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python main.py

# Or open the notebook
jupyter notebook notebooks/sentiment_analysis.ipynb
```

---

## (12) Requirements

```
pandas
numpy
scikit-learn
matplotlib
notebook
```

---

## (13) References

1. Manning, C., & Schütze, H. (1999). *Foundations of Statistical Natural Language Processing*. MIT Press.
2. Pang, B., & Lee, L. (2008). Opinion mining and sentiment analysis. *Foundations and Trends in IR*, 2(1–2), 1–135.
3. Scikit-learn Documentation — https://scikit-learn.org/stable/
4. Anthropic Claude 4 — https://www.anthropic.com
5. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media.
