"""
text_cleaner.py

Professional text preprocessing module.

This module performs:
- Lowercase conversion
- URL removal
- Email removal
- HTML removal
- Punctuation removal
- Number removal
- Stopword removal
- Lemmatization
"""

import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# Initialize
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Clean raw text for emotion detection.

    Parameters
    ----------
    text : str
        Input sentence

    Returns
    -------
    str
        Cleaned sentence
    """

    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Remove Emails
    text = re.sub(r"\S+@\S+", "", text)

    # Remove HTML Tags
    text = re.sub(r"<.*?>", "", text)

    # Remove Numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stop words
    words = [
        word
        for word in text.split()
        if word not in STOP_WORDS
    ]

    # Lemmatization
    words = [
        LEMMATIZER.lemmatize(word)
        for word in words
    ]

    return " ".join(words)
