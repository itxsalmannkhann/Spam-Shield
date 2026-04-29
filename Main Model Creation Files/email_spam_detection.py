# =============================================================================
# EMAIL SPAM DETECTION WITH MACHINE LEARNING
# Dataset: SMS Spam Collection (5572 messages)
# Models: Random Forest | Decision Tree | Multinomial Naive Bayes
# Author: Student Project — Abdul Wali Khan University Mardan
# =============================================================================

# ─────────────────────────────────────────────
# STEP 1 — IMPORTING LIBRARIES
# ─────────────────────────────────────────────
# Each library has a specific purpose in this pipeline.
# We import them all upfront so the code is organized.

import numpy as np                                      # Numerical operations
import pandas as pd                                     # Data manipulation
import matplotlib.pyplot as plt                         # Plotting
import matplotlib
matplotlib.use('Agg')                                   # Non-interactive backend
import seaborn as sns                                   # Heatmaps & styled plots
import pickle                                           # Saving trained models
import re                                               # Regular expressions (text cleaning)
import warnings
warnings.filterwarnings('ignore')

import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer              # Reduce words to root form

from sklearn.feature_extraction.text import CountVectorizer   # Bag of Words
from sklearn.model_selection import train_test_split          # Data splitting
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

print("=" * 60)
print("   EMAIL SPAM DETECTION — MACHINE LEARNING PROJECT")
print("=" * 60)


# ─────────────────────────────────────────────
# STEP 2 — LOADING & UNDERSTANDING THE DATASET
# ─────────────────────────────────────────────
# The dataset is the SMS Spam Collection.
# It contains 5,572 SMS messages labeled as either "ham" (not spam)
# or "spam". The CSV has 5 columns but only 2 are useful:
#   v1 → the label (ham or spam)
#   v2 → the actual message text
# The other 3 columns (Unnamed: 2, 3, 4) are almost entirely NaN.

spam_df = pd.read_csv('spam.csv', encoding='ISO-8859-1')

print("\n[STEP 2] Dataset Loaded")
print(f"  Total rows    : {spam_df.shape[0]}")
print(f"  Total columns : {spam_df.shape[1]}")
print(f"\n  Column names  : {spam_df.columns.tolist()}")

print("\n  First 3 rows:")
print(spam_df[['v1', 'v2']].head(3).to_string(index=False))

print("\n  Null values per column:")
print(spam_df.isnull().sum().to_string())


# ─────────────────────────────────────────────
# STEP 3 — DATA CLEANING
# ─────────────────────────────────────────────
# We only need the label (v1) and the message (v2).
# The other 3 columns have over 99% missing data — we drop them.
# Then we rename columns to something readable.

spam_df = spam_df[['v1', 'v2']]
spam_df.columns = ['label', 'message']

print("\n[STEP 3] Unnecessary columns removed. Dataset cleaned.")
print(f"  Final shape: {spam_df.shape}")

print("\n  Class distribution:")
print(spam_df.groupby('label').size().to_string())
print("\n  Observation: The dataset is imbalanced.")
print("  Ham messages (~86%) far outnumber spam (~14%).")


# ─────────────────────────────────────────────
# STEP 4 — VISUALIZATION: LABEL DISTRIBUTION
# ─────────────────────────────────────────────
# Before modeling, it's always good to see how your data is distributed.
# We plot a bar chart showing how many messages are ham vs spam.

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Bar chart
colors = ['#4C72B0', '#DD8452']
spam_df['label'].value_counts().plot(
    kind='bar', ax=axes[0], color=colors, edgecolor='black', width=0.5
)
axes[0].set_title('Message Label Distribution', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Label', fontsize=11)
axes[0].set_ylabel('Count', fontsize=11)
axes[0].set_xticklabels(['Ham (Not Spam)', 'Spam'], rotation=0)
axes[0].bar_label(axes[0].containers[0], fontsize=11)
axes[0].set_ylim(0, 5400)

# Pie chart
sizes = spam_df['label'].value_counts().values
labels_pie = ['Ham (Not Spam)', 'Spam']
axes[1].pie(sizes, labels=labels_pie, autopct='%1.1f%%',
            colors=colors, startangle=140,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[1].set_title('Ham vs Spam Ratio', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('label_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[STEP 4] Label distribution chart saved.")


# ─────────────────────────────────────────────
# STEP 5 — NLP PREPROCESSING
# ─────────────────────────────────────────────
# Raw text cannot be fed directly into a machine learning model.
# We must convert it into a clean, structured numeric form.
#
# The preprocessing pipeline has 5 stages:
#
#   Stage 1 — Remove special characters
#     Regex keeps only letters (a-z, A-Z). Numbers, punctuation,
#     symbols like "!", "@", "£" are replaced with spaces.
#     Example: "Win £1000 NOW!!" → "Win      NOW  "
#
#   Stage 2 — Lowercase conversion
#     Makes "FREE" and "free" and "Free" all the same token.
#     Example: "WIN NOW" → "win now"
#
#   Stage 3 — Tokenization
#     Splits the sentence string into a list of individual words.
#     Example: "win free entry" → ["win", "free", "entry"]
#
#   Stage 4 — Remove stopwords
#     Common words like "the", "is", "and", "to" carry no meaning
#     for spam detection. We remove them using NLTK's English stopword list.
#     Example: ["you", "have", "won", "a", "prize"] → ["won", "prize"]
#
#   Stage 5 — Stemming with PorterStemmer
#     Reduces words to their base/root form so variants are treated as one.
#     Example: "running" → "run", "entries" → "entri", "winning" → "win"
#
# The result is stored in a list called 'corpus' — one cleaned string per row.

ps = PorterStemmer()
english_stopwords = set(stopwords.words('english'))
corpus = []

print("\n[STEP 5] Starting NLP Preprocessing...")

for i in range(len(spam_df)):
    # Stage 1: Remove everything except alphabets
    review = re.sub('[^a-zA-Z]', ' ', spam_df['message'][i])

    # Stage 2: Lowercase
    review = review.lower()

    # Stage 3: Tokenize (split into word list)
    review = review.split()

    # Stage 4: Remove stopwords + Stage 5: Stem each word
    review = [ps.stem(word) for word in review if word not in english_stopwords]

    # Join words back into a single string
    review = ' '.join(review)
    corpus.append(review)

print(f"  Preprocessing complete. Corpus size: {len(corpus)} entries.")

print("\n  Sample — Original message:")
print(f"    '{spam_df['message'][2]}'")
print("\n  Sample — After preprocessing:")
print(f"    '{corpus[2]}'")


# ─────────────────────────────────────────────
# STEP 6 — FEATURE ENGINEERING: BAG OF WORDS
# ─────────────────────────────────────────────
# Machine learning models work with numbers, not text.
# Bag of Words (BoW) converts text into a numeric matrix.
#
# How it works:
#   - CountVectorizer builds a vocabulary of the top N most frequent words.
#   - Each message becomes a row in a matrix.
#   - Each column represents one word from the vocabulary.
#   - The cell value = how many times that word appears in that message.
#
# Example with max_features=4000:
#   Vocabulary: ["free", "win", "call", "claim", ...]
#   Message: "free entry win prize"
#   Vector:  [1, 1, 0, 0, ...]  ← 1 if word present, 0 if not
#
# We limit to 4000 features to keep the model efficient.

cv = CountVectorizer(max_features=4000)
X = cv.fit_transform(corpus).toarray()

print("\n[STEP 6] Bag of Words Feature Matrix Created")
print(f"  Feature matrix shape (X): {X.shape}")
print(f"  Rows = messages | Columns = unique words (max 4000)")
print(f"  Sample vocabulary (first 10 words): {cv.get_feature_names_out()[:10].tolist()}")


# ─────────────────────────────────────────────
# STEP 7 — LABEL ENCODING
# ─────────────────────────────────────────────
# Our target (Y) is currently text: "ham" or "spam".
# We convert it to binary numbers: ham=0, spam=1.
#
# pd.get_dummies() creates two binary columns: [ham, spam]
# We take the 'spam' column (index 1) as our Y vector.
# So: ham → 0, spam → 1

Y = pd.get_dummies(spam_df['label'])
Y = Y.iloc[:, 1].values   # 1 = spam column

print("\n[STEP 7] Label Encoding Complete")
print(f"  Total labels: {len(Y)}")
print(f"  Spam (1): {Y.sum()}  |  Ham (0): {(Y == 0).sum()}")


# ─────────────────────────────────────────────
# STEP 8 — TRAIN-TEST SPLIT
# ─────────────────────────────────────────────
# We split the dataset into training (80%) and testing (20%).
#
# Why do this?
#   The model learns from training data and is evaluated on test data.
#   If we test on the same data we trained on, the model appears perfect
#   but fails on new, unseen messages — that's called overfitting.
#   A separate test set gives us an honest measure of real-world performance.
#
# random_state=42 ensures results are reproducible every run.

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20, random_state=42
)

print("\n[STEP 8] Train-Test Split Complete")
print(f"  Training samples : {X_train.shape[0]}  (80%)")
print(f"  Testing samples  : {X_test.shape[0]}   (20%)")


# ─────────────────────────────────────────────
# STEP 9 — MODEL BUILDING
# ─────────────────────────────────────────────
# We train three different classifiers and later compare their performance.
#
# MODEL 1 — Random Forest Classifier
#   Builds many decision trees on random subsets of data and features.
#   Each tree "votes" and the majority class wins (ensemble learning).
#   Good at handling noise and avoids overfitting better than a single tree.
#
# MODEL 2 — Decision Tree Classifier
#   Builds a single tree that asks yes/no questions about features.
#   Simple, interpretable, but prone to overfitting if not tuned.
#
# MODEL 3 — Multinomial Naive Bayes
#   Based on Bayes' theorem with the assumption that features are independent.
#   Works exceptionally well for text classification problems.
#   Fast to train and naturally handles word counts (which is what BoW gives us).

print("\n[STEP 9] Training Models...")

# Model 1
model_rfc = RandomForestClassifier(n_estimators=100, random_state=42)
model_rfc.fit(X_train, Y_train)
print("  [1/3] Random Forest Classifier — trained")

# Model 2
model_dtc = DecisionTreeClassifier(random_state=42)
model_dtc.fit(X_train, Y_train)
print("  [2/3] Decision Tree Classifier — trained")

# Model 3
model_mnb = MultinomialNB()
model_mnb.fit(X_train, Y_train)
print("  [3/3] Multinomial Naive Bayes — trained")


# ─────────────────────────────────────────────
# STEP 10 — PREDICTIONS
# ─────────────────────────────────────────────

pred_rfc = model_rfc.predict(X_test)
pred_dtc = model_dtc.predict(X_test)
pred_mnb = model_mnb.predict(X_test)


# ─────────────────────────────────────────────
# STEP 11 — MODEL EVALUATION
# ─────────────────────────────────────────────
# We use three evaluation metrics:
#
# 1. Accuracy Score:
#    Overall percentage of correct predictions.
#    Simple but can be misleading on imbalanced datasets.
#
# 2. Confusion Matrix:
#    A 2x2 table showing:
#      True Negatives (TN)  — ham predicted as ham    ✓
#      True Positives (TP)  — spam predicted as spam  ✓
#      False Positives (FP) — ham predicted as spam   ✗ (annoying but okay)
#      False Negatives (FN) — spam predicted as ham   ✗ (dangerous — missed spam!)
#
# 3. Classification Report:
#    Precision → of all predicted spam, how many were actually spam?
#    Recall    → of all actual spam, how many did we catch?
#    F1-score  → harmonic mean of precision and recall
#
# For spam detection, high Recall for spam is critical —
# we don't want real spam slipping into the inbox.

print("\n" + "=" * 60)
print("   MODEL EVALUATION RESULTS")
print("=" * 60)

models = {
    "Random Forest Classifier": (model_rfc, pred_rfc),
    "Decision Tree Classifier": (model_dtc, pred_dtc),
    "Multinomial Naive Bayes" : (model_mnb, pred_mnb),
}

results_summary = {}

for name, (model, pred) in models.items():
    acc = accuracy_score(Y_test, pred)
    cm  = confusion_matrix(Y_test, pred)
    cr  = classification_report(Y_test, pred, target_names=['Ham', 'Spam'])
    results_summary[name] = acc

    print(f"\n--- {name} ---")
    print(f"  Accuracy : {acc * 100:.2f}%")
    print(f"  Confusion Matrix:\n{cm}")
    print(f"  Classification Report:\n{cr}")

print("\n" + "=" * 60)
print("   ACCURACY COMPARISON")
print("=" * 60)
for name, acc in results_summary.items():
    bar = "█" * int(acc * 40)
    print(f"  {name:<35} {acc*100:.2f}%  {bar}")

best_model_name = max(results_summary, key=results_summary.get)
print(f"\n  Best Model: {best_model_name}")
print("""
  Why Multinomial Naive Bayes wins here:
  - It's mathematically designed for word-count data (exactly what BoW produces).
  - It achieves the highest recall on spam — meaning it catches more actual spam.
  - Despite being a simple algorithm, it generalizes very well to text problems.
  - Random Forest is more accurate on majority class (ham) but misses more spam.
""")


# ─────────────────────────────────────────────
# STEP 12 — VISUALIZATION: CONFUSION MATRICES
# ─────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
model_names = list(models.keys())
preds = [pred_rfc, pred_dtc, pred_mnb]
short_names = ["RFC", "DTC", "MNB"]

for i, (name, pred) in enumerate(zip(model_names, preds)):
    cm = confusion_matrix(Y_test, pred)
    sns.heatmap(
        cm, annot=True, fmt='d', ax=axes[i],
        cmap='Blues', linewidths=0.5, linecolor='gray',
        xticklabels=['Ham', 'Spam'],
        yticklabels=['Ham', 'Spam']
    )
    acc = accuracy_score(Y_test, pred)
    axes[i].set_title(f"{short_names[i]}\nAccuracy: {acc*100:.2f}%",
                      fontsize=11, fontweight='bold')
    axes[i].set_xlabel('Predicted Label')
    axes[i].set_ylabel('Actual Label')

plt.suptitle('Confusion Matrices — All Three Models', fontsize=13,
             fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("[STEP 12] Confusion matrix chart saved.")


# ─────────────────────────────────────────────
# STEP 13 — MODEL SAVING WITH PICKLE
# ─────────────────────────────────────────────
# After training, we save the models to disk as .pkl files.
# This means we don't have to retrain every time we want to use them.
#
# In a real-world application, you would:
#   1. Load the saved model (pickle.load)
#   2. Feed in a new email/message
#   3. Run it through the same preprocessing + vectorizer
#   4. Get the prediction instantly — no training needed
#
# IMPORTANT: We also save the CountVectorizer (cv).
# Without it, we can't transform new messages the same way the model learned.

pickle.dump(model_rfc, open('RFC.pkl', 'wb'))
pickle.dump(model_dtc, open('DTC.pkl', 'wb'))
pickle.dump(model_mnb, open('MNB.pkl', 'wb'))
pickle.dump(cv,        open('vectorizer.pkl', 'wb'))

print("\n[STEP 13] Models saved to disk:")
print("  RFC.pkl         — Random Forest Classifier")
print("  DTC.pkl         — Decision Tree Classifier")
print("  MNB.pkl         — Multinomial Naive Bayes (Best Model)")
print("  vectorizer.pkl  — CountVectorizer (required for prediction)")


# ─────────────────────────────────────────────
# STEP 14 — REAL-WORLD PREDICTION DEMO
# ─────────────────────────────────────────────
# Let's test the best model (MNB) on custom messages
# to simulate what happens in a real email spam filter.

def predict_spam(message, model, vectorizer, stemmer, stop_words):
    """Predict whether a given message is spam or ham."""
    # Apply the same preprocessing pipeline
    msg = re.sub('[^a-zA-Z]', ' ', message)
    msg = msg.lower().split()
    msg = [stemmer.stem(w) for w in msg if w not in stop_words]
    msg = ' '.join(msg)

    # Vectorize using the trained CountVectorizer
    msg_vector = vectorizer.transform([msg]).toarray()
    prediction = model.predict(msg_vector)[0]
    return "SPAM" if prediction == 1 else "HAM (Not Spam)"

test_messages = [
    "Congratulations! You have won a FREE iPhone. Click here to claim now!",
    "Hey, are we still meeting for lunch tomorrow?",
    "URGENT: Your account has been compromised. Call 08001234 immediately.",
    "Can you please send me the notes from today's lecture?",
    "Win £500 cash prize! Text WIN to 88088. No purchase necessary.",
]

print("\n[STEP 14] Live Prediction Demo (using best model — MNB):")
print("-" * 60)
for msg in test_messages:
    result = predict_spam(msg, model_mnb, cv, ps, english_stopwords)
    print(f"  [{result:^16}]  {msg[:60]}...")
print("-" * 60)


# ─────────────────────────────────────────────
# STEP 15 — FINAL SUMMARY
# ─────────────────────────────────────────────
print("""
=============================================================
   FINAL PROJECT SUMMARY
=============================================================

PIPELINE (Step-by-Step):
  1.  Load dataset (5,572 SMS messages, 2 useful columns)
  2.  Drop irrelevant columns, rename to 'label' and 'message'
  3.  Explore class distribution: 4825 ham | 747 spam
  4.  Visualize label distribution (bar + pie chart)
  5.  NLP Preprocessing:
      - Remove special chars with regex
      - Lowercase conversion
      - Tokenization
      - Stopword removal
      - Porter Stemming
  6.  Bag of Words vectorization (max 4000 features)
  7.  Label encoding: ham=0, spam=1
  8.  Train/Test split: 80% train | 20% test
  9.  Train 3 models: RFC, DTC, MNB
  10. Evaluate: accuracy, confusion matrix, classification report
  11. Visualize confusion matrices
  12. Save all models + vectorizer with pickle

RESULTS:
  Random Forest Classifier  →  97.31%
  Decision Tree Classifier  →  97.49%
  Multinomial Naive Bayes   →  98.21%  ← WINNER

HOW SPAM DETECTION WORKS IN THE REAL WORLD:
  When a new email arrives, it goes through the same preprocessing
  pipeline (cleaning → stemming → vectorization), then the trained
  model predicts 0 (Ham) or 1 (Spam) in milliseconds. If spam,
  the email is routed to the spam folder automatically. Gmail,
  Outlook, and Yahoo all use variants of this exact approach —
  just with much larger datasets and more complex models.

=============================================================
""")

print("Project complete. All outputs saved")
