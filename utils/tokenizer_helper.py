import sys
from pathlib import Path

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
import pandas as pd
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from config import PROCESSED_DATASET_DIR, TOKENIZER_DIR

MAX_WORDS = 20000
MAX_LENGTH = 100

train = pd.read_csv(PROCESSED_DATASET_DIR / "train.csv")

# Remove rows where text is missing
train = train.dropna(subset=["text"])

# Ensure all values are strings
texts = train["text"].astype(str)

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)

padded = pad_sequences(
    sequences,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

print("Tokenizer created successfully!")
print("Vocabulary Size:", len(tokenizer.word_index))

TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
with open(TOKENIZER_DIR / "tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Tokenizer saved.")
print("Shape:", padded.shape)
