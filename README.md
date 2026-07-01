# Email Spam Detection with Machine Learning

A complete machine learning pipeline to classify SMS/email messages as **spam** or **ham (not spam)** using Natural Language Processing and three classification models.

Built as a student portfolio project at **Abdul Wali Khan University Mardan** — 6th Semester, Artificial Intelligence.

---

## Project Overview

Spam detection is one of the most practical applications of machine learning in everyday life. Every time Gmail or Outlook moves a suspicious email to your spam folder, a model similar to this one is running in the background.

This project covers the complete pipeline from raw text data to a saved, deployable model:

```
Raw Text → Cleaning → NLP Preprocessing → Vectorization → Model Training → Prediction
```

**Dataset:** SMS Spam Collection — 5,572 labeled messages  
**Best Model:** Multinomial Naive Bayes — **98.12% accuracy**

---

## Project Structure

```
email-spam-detection/
│
├── email_spam_detection.py   ← Full training pipeline (run this first)
├── predict_spam.py           ← Load model and predict new messages
├── spam.csv                  ← Dataset (5,572 SMS messages)
│
├── MNB.pkl                   ← Saved: Multinomial Naive Bayes (best model)
├── RFC.pkl                   ← Saved: Random Forest Classifier
├── DTC.pkl                   ← Saved: Decision Tree Classifier
├── vectorizer.pkl            ← Saved: CountVectorizer (required for prediction)
│
├── label_distribution.png    ← Chart: Ham vs Spam distribution
├── confusion_matrices.png    ← Chart: Confusion matrices for all 3 models
│
├── requirements.txt          ← All Python dependencies
└── README.md                 ← This file
```

---

## Requirements

Python version: **3.8 or higher**

Install all dependencies with one command:

```bash
pip install -r requirements.txt
```

---

## How to Run

### Step 1 — Train the models

Run the main training script. This reads `spam.csv`, preprocesses the text, trains all three models, evaluates them, and saves the `.pkl` files.

```bash
python email_spam_detection.py
```

You only need to run this once. After this, the `.pkl` files are saved and ready.

---

### Step 2 — Predict new messages (Interactive Mode)

Run the prediction script. It loads the saved model and lets you type any message to check if it is spam.

```bash
python predict_spam.py
```

**Example session:**

```
============================================================
   EMAIL / SMS SPAM DETECTOR
   Model: Multinomial Naive Bayes
   Type 'quit' or 'exit' to stop
============================================================

Enter a message to check: Congratulations! You have won a FREE iPhone!

────────────────────────────────────────────────────────────
  [SPAM]    !! SPAM DETECTED !!
────────────────────────────────────────────────────────────
  Original  : Congratulations! You have won a FREE iPhone!
  Cleaned   : congratul won free iphon
  Ham  probability : 0.06%
  Spam probability : 99.94%
────────────────────────────────────────────────────────────

Enter a message to check: Are you coming to class tomorrow?

────────────────────────────────────────────────────────────
  [HAM]     Safe Message (HAM)
────────────────────────────────────────────────────────────
  Original  : Are you coming to class tomorrow?
  Cleaned   : come class tomorrow
  Ham  probability : 100.0%
  Spam probability : 0.0%
────────────────────────────────────────────────────────────

Enter a message to check: quit

  Exiting. Goodbye.
```

---

### Step 3 — Use in your own Python code (Batch Mode)

You can import the predictor directly into any Python script:

```python
from predict_spam import batch_predict

messages = [
    "Win £500 cash prize now! Text WIN to 88088.",
    "I will send you the notes after the lecture.",
    "URGENT: Your account has been compromised. Call immediately.",
    "See you at the library at 5pm.",
]

results = batch_predict(messages)

for r in results:
    print(f"[{r['label']}] {r['original']}")
    print(f"  Spam probability: {r['spam_chance']}%\n")
```

**Output:**

```
[SPAM] Win £500 cash prize now! Text WIN to 88088.
  Spam probability: 100.0%

[HAM] I will send you the notes after the lecture.
  Spam probability: 6.96%

[SPAM] URGENT: Your account has been compromised. Call immediately.
  Spam probability: 94.66%

[HAM] See you at the library at 5pm.
  Spam probability: 0.0%
```

---

## How It Works (Full Pipeline Explained)

### 1. Data Loading
The dataset (`spam.csv`) has 5 columns. Only two matter:
- `v1` — the label: `ham` or `spam`
- `v2` — the actual message text

The other 3 columns are almost entirely empty (NaN) and are dropped.

---

### 2. NLP Preprocessing

Raw text cannot be fed into a machine learning model. We clean it through 5 steps:

| Step | What it does | Example |
|------|-------------|---------|
| Remove special chars | Keeps only a-z letters | `"Win £1000!!"` → `"Win  1000 "` |
| Lowercase | Normalizes case | `"FREE ENTRY"` → `"free entry"` |
| Tokenize | Splits into word list | `"free entry"` → `["free", "entry"]` |
| Remove stopwords | Drops meaningless words | `["you", "have", "won"]` → `["won"]` |
| Stemming | Reduces to root form | `"entries"` → `"entri"`, `"winning"` → `"win"` |

**Why stemming?** The words "entry", "entries", and "entered" all mean the same thing. Stemming treats them as one word (`"entri"`), which reduces vocabulary size and improves model performance.

---

### 3. Bag of Words (Feature Engineering)

`CountVectorizer` converts the cleaned text into a numeric matrix:

- Builds a vocabulary of the **4,000 most frequent words** across all messages
- Each message becomes a row of numbers
- Each column represents one word from the vocabulary
- The value in each cell = how many times that word appears in that message

```
Vocabulary:  ["free", "win",  "call", "prize", "meet"]
Message:     "free win prize"
Vector:      [1,      1,     0,      1,       0    ]
```

This matrix (shape: 5572 × 4000) is what the models actually learn from.

---

### 4. Label Encoding

The target variable is converted from text to numbers:
- `ham`  → `0`
- `spam` → `1`

---

### 5. Train-Test Split (80/20)

The data is split:
- **80% training** — the model learns from this
- **20% testing** — the model is evaluated on this (it has never seen it)

This gives us an honest estimate of how the model performs on new, unseen messages.

---

### 6. Models Trained

| Model | How it works |
|-------|-------------|
| **Random Forest** | Builds 100 decision trees on random subsets. Each tree votes; majority wins. |
| **Decision Tree** | Builds one tree with yes/no questions based on word presence. |
| **Naive Bayes** | Uses probability: given these words, how likely is this spam? Assumes each word is independent. |

---

### 7. Results

| Model | Accuracy | Spam Recall | Spam Precision |
|-------|----------|-------------|----------------|
| Random Forest Classifier | 97.67% | 83% | 100% |
| Decision Tree Classifier | 97.49% | 85% | 96% |
| **Multinomial Naive Bayes** | **98.12%** | **94%** | **92%** |

**Why Naive Bayes is the winner:**

Accuracy alone doesn't tell the full story. In spam detection, **Recall** is the most important metric for the spam class. Recall tells us: *of all the actual spam messages, how many did the model correctly catch?*

- Random Forest has 100% precision but only 83% recall — it's very conservative and lets a lot of spam through.
- Naive Bayes achieves 94% recall — it catches almost all spam — which is exactly what we want.

---

### 8. Model Saving

All models are saved using Python's `pickle` library:

```python
import pickle
model = pickle.load(open('MNB.pkl', 'rb'))
```

This means you don't need to retrain every time. In a production system, you'd train once, save the model, and just load it whenever a new message needs checking.

> **Important:** Always save and load the `vectorizer.pkl` alongside the model. The vectorizer must be the same one used during training, or predictions will be wrong.

---

## Real-World Context

This is how spam filters work at companies like Google, Microsoft, and Yahoo:

1. A new email arrives at the server
2. The text is cleaned and preprocessed (same steps as above)
3. It's converted to a numeric vector using a pre-trained vectorizer
4. The saved model predicts: 0 (Ham) or 1 (Spam) — in milliseconds
5. If spam, the email is moved to the spam folder automatically

Production systems use more advanced models (like deep learning with BERT or LSTM), much larger datasets (billions of emails), and additional signals like sender reputation, IP address, and link analysis. But the core NLP pipeline is essentially the same as what you see here.

---

## Dataset Information

- **Source:** UCI Machine Learning Repository — SMS Spam Collection
- **Total messages:** 5,572
- **Ham (not spam):** 4,825 (86.6%)
- **Spam:** 747 (13.4%)
- **Language:** English
- **Encoding:** ISO-8859-1

The dataset is imbalanced (much more ham than spam), which is realistic — in real life, most messages you receive are not spam. This is why we look at recall and precision for the spam class specifically, rather than just overall accuracy.

---

## Author
**Name:** Salman Khan
**Student:** Abdul Wali Khan University Mardan  
**Department:** Computer Science / Artificial Intelligence  
**Semester:** 6th Semester  
**Project Type:** Machine Learning — Text Classification
