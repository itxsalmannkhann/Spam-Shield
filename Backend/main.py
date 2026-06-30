# =============================================================================
# main.py — FastAPI Backend for Email Spam Detector
# Deploy this on Render.com (free tier)
# =============================================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import re
import os
from nltk.stem.porter import PorterStemmer

# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Spam Detector API",
    description="Detects whether a given message is spam or ham using Multinomial Naive Bayes.",
    version="1.0.0"
)

# ── CORS — Allow requests from your Vercel frontend ───────────────────────────
# During development: allow all origins (*)
# After deployment: replace "*" with your actual Vercel URL
# Example: "https://your-app-name.vercel.app"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Model and Vectorizer ─────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    # Load model and vectorizer from pickle files
    model      = pickle.load(open(os.path.join(BASE_DIR, "MNB.pkl"), "rb"))
    vectorizer = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))
    print("Model and vectorizer loaded successfully.")
except FileNotFoundError as e:
    # Handle file not found error
    print(f"ERROR: Could not load model files. {e}")
    model = None
    vectorizer = None

# ── Stopwords (inline — no NLTK download needed on server) ───────────────────
# Define a set of stopwords to ignore during preprocessing
STOPWORDS = {
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

# Initialize PorterStemmer for stemming words
ps = PorterStemmer()


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    """Apply the same cleaning pipeline used during model training."""
    # Remove non-alphabetic characters and convert to lowercase
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    # Remove stopwords and stem words
    text = [ps.stem(w) for w in text if w not in STOPWORDS]
    # Join the cleaned words back into a string
    return ' '.join(text)


# ── Request / Response Models ─────────────────────────────────────────────────
class MessageRequest(BaseModel):
    message: str

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Congratulations! You have won a FREE iPhone. Click here to claim now!"
            }
        }

class PredictionResponse(BaseModel):
    original_message : str
    cleaned_text     : str
    label            : str          # "SPAM" or "HAM"
    is_spam          : bool
    spam_probability : float        # 0.0 to 100.0
    ham_probability  : float        # 0.0 to 100.0
    confidence       : str          # "High" / "Medium" / "Low"


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Spam Detector API is running.",
        "usage"  : "POST /predict with JSON body: { 'message': 'your text here' }",
        "docs"   : "/docs"
    }

@app.get("/health")
def health():
    return {
        "status"      : "ok",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: MessageRequest):
    """
    Predict whether a message is spam or ham.
    Returns label, probabilities, and confidence level.
    """
    # Check if model and vectorizer are loaded
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")

    # Check if message is empty
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # Preprocess the message
    cleaned = preprocess(request.message)

    # Vectorize the cleaned message
    vector = vectorizer.transform([cleaned]).toarray()

    # Make prediction using the model
    prediction   = model.predict(vector)[0]
    proba        = model.predict_proba(vector)[0]
    ham_prob     = round(float(proba[0]) * 100, 2)
    spam_prob    = round(float(proba[1]) * 100, 2)

    # Determine confidence level based on winning probability
    max_prob = max(ham_prob, spam_prob)
    if max_prob >= 90:
        confidence = "High"
    elif max_prob >= 70:
        confidence = "Medium"
    else:
        confidence = "Low"

    return PredictionResponse(
        original_message = request.message,
        cleaned_text     = cleaned,
        label            = "SPAM" if prediction == 1 else "HAM",
        is_spam          = bool(prediction),
        spam_probability = spam_prob,
        ham_probability  = ham_prob,
        confidence       = confidence
    )