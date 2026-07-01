"""
=========================================================
Emotion Detection & Learning Support Engine
Text Preprocessing Module
=========================================================
"""

import re
import string
from pathlib import Path

import nltk
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import (
    RAW_DATASET_DIR,
    PROCESSED_DATASET_DIR,
    RANDOM_STATE,
)


# Ensure required NLTK resources are available (download if missing)
def _ensure_nltk_resources():
    try:
        stopwords.words("english")
    except Exception:
        nltk.download("stopwords")

    try:
        WordNetLemmatizer().lemmatize("running")
    except Exception:
        nltk.download("wordnet")
        nltk.download("omw-1.4")


_ensure_nltk_resources()


# Initialize commonly used NLP helpers
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """
    Clean raw text.

    Steps:
    1. Lowercase
    2. Remove URLs
    3. Remove HTML tags
    4. Remove punctuation
    5. Remove numbers
    6. Remove stopwords
    7. Lemmatize words
    """

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize
    words = text.split()

    # Remove stopwords and lemmatize
    cleaned_words = []

    for word in words:
        if word not in stop_words:
            cleaned_words.append(lemmatizer.lemmatize(word))

    return " ".join(cleaned_words)