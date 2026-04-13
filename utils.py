import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download at runtime (IMPORTANT)
nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = [
        stemmer.stem(word)
        for word in text.split()
        if word not in stop_words
    ]
    return " ".join(words)