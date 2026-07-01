import os
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

# ==============================
# Project Root
# ==============================

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import (
    PROCESSED_DATASET_DIR,
    TOKENIZER_DIR,
    LABEL_ENCODER_DIR,
    BILSTM_MODEL_DIR
)

print("=" * 60)
print("Emotion Detection BiLSTM Training")
print("=" * 60)
# ==============================
# Load Datasets
# ==============================

train_df = pd.read_csv(
    os.path.join(PROCESSED_DATASET_DIR, "train.csv")
)

val_df = pd.read_csv(
    os.path.join(PROCESSED_DATASET_DIR, "validation.csv")
)

print("Training samples:", len(train_df))
print("Validation samples:", len(val_df))

# Remove missing values
train_df = train_df.dropna(subset=["text", "emotion"])
val_df = val_df.dropna(subset=["text", "emotion"])

# Convert to string
train_df["text"] = train_df["text"].astype(str)
val_df["text"] = val_df["text"].astype(str)

print("Dataset loaded successfully!")
print(train_df.head())

# ==============================
# Load Tokenizer and Label Encoder
# ==============================

tokenizer_path = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
label_encoder_path = os.path.join(LABEL_ENCODER_DIR, "label_encoder.pkl")

with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)

with open(label_encoder_path, "rb") as f:
    label_encoder = pickle.load(f)

print("Tokenizer and Label Encoder loaded!")

# ==============================
# Tokenize and Pad Sequences
# ==============================

from config import MAX_SEQUENCE_LENGTH, VOCAB_SIZE, EMBEDDING_DIM

train_sequences = tokenizer.texts_to_sequences(train_df["text"].values)
val_sequences = tokenizer.texts_to_sequences(val_df["text"].values)

train_padded = pad_sequences(
    train_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="post"
)
val_padded = pad_sequences(
    val_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="post"
)

print(f"Train padded shape: {train_padded.shape}")
print(f"Validation padded shape: {val_padded.shape}")

# ==============================
# Encode Labels
# ==============================

train_labels = label_encoder.transform(train_df["emotion"].values)
val_labels = label_encoder.transform(val_df["emotion"].values)

num_classes = len(label_encoder.classes_)
train_labels_cat = to_categorical(train_labels, num_classes)
val_labels_cat = to_categorical(val_labels, num_classes)

print(f"Number of classes: {num_classes}")
print(f"Train labels shape: {train_labels_cat.shape}")

# ==============================
# Build BiLSTM Model
# ==============================

model = Sequential([
    Embedding(
        input_dim=VOCAB_SIZE + 1,
        output_dim=EMBEDDING_DIM,
        input_shape=(MAX_SEQUENCE_LENGTH,)
    ),

    Bidirectional(
        LSTM(64, return_sequences=True)
    ),

    Dropout(0.2),

    Bidirectional(
        LSTM(32)
    ),

    Dropout(0.2),

    Dense(16, activation="relu"),

    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel Architecture:")
model.summary()

# ==============================
# Define Callbacks
# ==============================

checkpoint = ModelCheckpoint(
os.path.join(BILSTM_MODEL_DIR, "bilstm_best_model.keras"),
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

# ==============================
# Train Model
# ==============================

from config import EPOCHS, BATCH_SIZE

print("\n===== DEBUG INFO =====")

print(type(train_padded))
print(train_padded.dtype)
print(train_padded.shape)

print(type(train_labels_cat))
print(train_labels_cat.dtype)
print(train_labels_cat.shape)

print("Max token:", train_padded.max())
print("Min token:", train_padded.min())

print("NaN in X:", np.isnan(train_padded).any())
print("NaN in y:", np.isnan(train_labels_cat).any())

print("======================\n")
print("\nStarting training...")
history = model.fit(
    train_padded[:512],
    train_labels_cat[:512],
    validation_data=(
        val_padded[:128],
        val_labels_cat[:128]
    ),
    epochs=1,
    batch_size=32,
    verbose=1
)

# ==============================
# Save Model
# ==============================
model.save(os.path.join(BILSTM_MODEL_DIR, "bilstm_final_model.keras"))
print(f"\nModel saved to {BILSTM_MODEL_DIR}")

print("=" * 60)
print("Training Complete!")
print("=" * 60)