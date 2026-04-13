import os
import re
import math
import random
import pickle

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from flask import Flask, render_template, request, jsonify

# ---------------------------------------------------------------------------
# IMPORTANT: clean_text must be defined BEFORE loading the model so that
# pickle can locate it as __main__.clean_text when deserialising the pipeline.
# ---------------------------------------------------------------------------

nltk.download('stopwords', quiet=True)

_stop_words = set(stopwords.words('english'))
_stemmer = PorterStemmer()


def clean_text(text):
    """Mirror of the preprocessing function used during model training."""
    # Lowercase
    text = text.lower()
    # Remove punctuation & numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenise, remove stopwords, apply stemming
    words = [
        _stemmer.stem(word)
        for word in text.split()
        if word not in _stop_words
    ]
    return " ".join(words)


# ---------------------------------------------------------------------------
# Load pre-trained model
# ---------------------------------------------------------------------------

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pkl')

try:
    with open(_MODEL_PATH, 'rb') as _f:
        model = pickle.load(_f)
except FileNotFoundError:
    raise RuntimeError(
        "model.pkl not found. Please run all cells in model_training.ipynb "
        "to train and save the model first."
    )

# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------

app = Flask(__name__)

# Product categories present in the training dataset (Amazon categories)
CATEGORIES = [
    "Home_&_Kitchen",
    "Electronics",
    "Sports_&_Outdoors",
    "Arts,_Crafts_&_Sewing",
    "Clothing,_Shoes_&_Jewelry",
    "Health_&_Personal_Care",
    "Tools_&_Home_Improvement",
    "Toys_&_Games",
    "Books",
    "Grocery_&_Gourmet_Food",
    "Movies_&_TV",
    "Musical_Instruments",
    "Office_Products",
    "Pet_Supplies",
    "Software",
]

# Template pool used by the /generate endpoint
FAKE_REVIEW_TEMPLATES = [
    "Amazing product! Loved it, great quality and value.",
    "Highly recommend this, works perfectly and looks great.",
    "Very satisfied, exceeded my expectations.",
    "Great purchase, would definitely buy again.",
    "Excellent quality for the price, very happy with this!",
    "Best product I have ever bought. Absolutely five stars!",
    "Arrived quickly and exactly as described. Love it!",
    "Fantastic item, worth every penny. Will definitely order again.",
    "Works as advertised. Super easy to use and great build quality!",
    "Top notch product! Customer service was also outstanding.",
    "Incredible value. I bought two already and gifting one to a friend!",
    "Couldn't be happier with this purchase. 10/10 would recommend.",
    "Blew me away with its quality. Looks premium and feels sturdy.",
    "Fast shipping, beautiful packaging, product is perfect!",
    "I was skeptical at first but this exceeded every expectation.",
]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    """Homepage – renders the main form."""
    return render_template('index.html', categories=CATEGORIES)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Accept review text, category, and rating via JSON or form data.
    Returns prediction label and a confidence score.
    """
    # Support both JSON (fetch) and regular form submission
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form

    review  = (data.get('review')  or '').strip()
    category = (data.get('category') or '').strip()
    rating_raw = (data.get('rating') or '').strip()

    # --- Input validation ---
    errors = []
    if not review:
        errors.append('Review text is required.')
    if len(review) > 5000:
        errors.append('Review text must be 5 000 characters or fewer.')
    if not category:
        errors.append('Please select a product category.')

    try:
        rating = float(rating_raw)
        if not (1.0 <= rating <= 5.0):
            raise ValueError
    except (ValueError, TypeError):
        errors.append('Rating must be a number between 1 and 5.')
        rating = None

    if errors:
        return jsonify({'error': ' '.join(errors)}), 400

    # --- Build input DataFrame (must match training column names) ---
    input_df = pd.DataFrame([{
        'text_':    review,
        'category': category,
        'rating':   rating,
    }])

    # --- Predict ---
    prediction = model.predict(input_df)[0]
    is_fake = bool(prediction == 1)
    label = 'CG – Fake Review' if is_fake else 'OR – Genuine Review'
    label_class = 'danger' if is_fake else 'success'

    # --- Confidence score via decision function (LinearSVC) ---
    confidence = None
    try:
        decision_value = float(model.decision_function(input_df)[0])
        # Sigmoid transform to convert raw distance to a 0-100% confidence
        confidence = round(1 / (1 + math.exp(-abs(decision_value))) * 100, 1)
    except Exception:
        pass  # Not all estimators expose decision_function

    return jsonify({
        'prediction':  label,
        'label_class': label_class,
        'is_fake':     is_fake,
        'confidence':  confidence,
    })


@app.route('/generate')
def generate():
    """Return a randomly chosen fake-review template string."""
    return jsonify({'review': random.choice(FAKE_REVIEW_TEMPLATES)})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    app.run(debug=True)
