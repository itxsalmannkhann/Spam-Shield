# =============================================================================
# SPAM DETECTOR — Predict New Messages
# Load the saved model and predict any message you type
# =============================================================================

import pickle
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# ── Load saved model and vectorizer ──────────────────────────────────────────
# These files were saved after training. We just load them — no retraining.

MODEL_PATH      = "MNB.pkl"         # Best model (Multinomial Naive Bayes)
VECTORIZER_PATH = "vectorizer.pkl"  # CountVectorizer used during training

try:
    model      = pickle.load(open(MODEL_PATH, 'rb'))
    vectorizer = pickle.load(open(VECTORIZER_PATH, 'rb'))
except FileNotFoundError:
    print("[ERROR] Model files not found.")
    print("Make sure 'MNB.pkl' and 'vectorizer.pkl' are in the same folder.")
    exit()

# ── Setup NLP tools ───────────────────────────────────────────────────────────
nltk.download('stopwords', quiet=True)
ps = PorterStemmer()

try:
    stop_words = set(stopwords.words('english'))
except:
    # Fallback if NLTK data not available
    stop_words = {
        'i','me','my','myself','we','our','ours','ourselves','you','your',
        'yours','yourself','yourselves','he','him','his','himself','she',
        'her','hers','herself','it','its','itself','they','them','their',
        'theirs','themselves','what','which','who','whom','this','that',
        'these','those','am','is','are','was','were','be','been','being',
        'have','has','had','having','do','does','did','doing','a','an',
        'the','and','but','if','or','because','as','until','while','of',
        'at','by','for','with','about','against','between','into',
        'through','during','before','after','above','below','to','from',
        'up','down','in','out','on','off','over','under','again','further',
        'then','once','here','there','when','where','why','how','all',
        'both','each','few','more','most','other','some','such','no',
        'nor','not','only','own','same','so','than','too','very','s','t',
        'can','will','just','don','should','now','d','ll','m','o','re',
        've','y','ain','aren','couldn','didn','doesn','hadn','hasn',
        'haven','isn','let','mightn','mustn','shan','shouldn','wasn',
        'weren','won','wouldn'
    }


# ── Preprocessing function ────────────────────────────────────────────────────
def preprocess(message):
    """
    Clean and prepare a raw message using the same steps used during training.
    Steps: remove special chars → lowercase → tokenize → remove stopwords → stem
    """
    msg = re.sub('[^a-zA-Z]', ' ', message)   # Keep only letters
    msg = msg.lower()                           # Lowercase
    msg = msg.split()                           # Tokenize
    msg = [ps.stem(w) for w in msg if w not in stop_words]  # Stem + remove stopwords
    return ' '.join(msg)


# ── Prediction function ───────────────────────────────────────────────────────
def predict(message):
    """
    Takes a raw message string, preprocesses it, and returns the prediction.
    Returns: dict with label, confidence (if available), and cleaned text.
    """
    cleaned = preprocess(message)

    # Transform using the same vectorizer used during training
    vector = vectorizer.transform([cleaned]).toarray()

    # Predict
    prediction = model.predict(vector)[0]

    # Get probability scores (Naive Bayes supports this)
    try:
        proba = model.predict_proba(vector)[0]
        ham_conf  = round(proba[0] * 100, 2)
        spam_conf = round(proba[1] * 100, 2)
    except:
        ham_conf = spam_conf = "N/A"

    return {
        "label"       : "SPAM" if prediction == 1 else "HAM",
        "is_spam"     : bool(prediction),
        "ham_chance"  : ham_conf,
        "spam_chance" : spam_conf,
        "cleaned_text": cleaned
    }


# ── Display result ────────────────────────────────────────────────────────────
def display_result(message, result):
    width = 60
    border = "─" * width

    if result["is_spam"]:
        status = "  !! SPAM DETECTED !!"
        icon   = "[SPAM]"
    else:
        status = "  Safe Message (HAM)"
        icon   = "[HAM] "

    print(f"\n{border}")
    print(f"  {icon}  {status}")
    print(border)
    print(f"  Original  : {message[:55]}{'...' if len(message)>55 else ''}")
    print(f"  Cleaned   : {result['cleaned_text'][:55]}{'...' if len(result['cleaned_text'])>55 else ''}")
    print(f"  Ham  probability : {result['ham_chance']}%")
    print(f"  Spam probability : {result['spam_chance']}%")
    print(f"{border}\n")


# ── Main — Interactive Mode ───────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("   EMAIL / SMS SPAM DETECTOR")
    print("   Model: Multinomial Naive Bayes")
    print("   Type 'quit' or 'exit' to stop")
    print("=" * 60)

    while True:
        print()
        user_input = input("Enter a message to check: ").strip()

        if not user_input:
            print("  Please enter a message.")
            continue

        if user_input.lower() in ('quit', 'exit', 'q'):
            print("\n  Exiting. Goodbye.")
            break

        result = predict(user_input)
        display_result(user_input, result)


# ── Batch mode — check multiple messages at once ──────────────────────────────
def batch_predict(messages: list):
    """
    Use this function in your own code to check multiple messages at once.

    Example:
        from predict_spam import batch_predict
        results = batch_predict(["Win free prize!", "Hey, lunch tomorrow?"])
        for r in results:
            print(r)
    """
    results = []
    for msg in messages:
        result = predict(msg)
        result["original"] = msg
        results.append(result)
    return results


if __name__ == "__main__":
    main()
