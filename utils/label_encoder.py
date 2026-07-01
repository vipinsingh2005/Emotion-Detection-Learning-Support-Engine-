import os
import sys
from pathlib import Path

import joblib
import pandas as pd

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from sklearn.preprocessing import LabelEncoder

from config import PROCESSED_DATASET_DIR, LABEL_ENCODER_DIR

# -----------------------------
# Paths
# -----------------------------
LABEL_ENCODER_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(PROCESSED_DATASET_DIR / "train.csv")

# Remove missing values
df = df.dropna(subset=["emotion"])

# -----------------------------
# Encode labels
# -----------------------------
label_encoder = LabelEncoder()

labels = label_encoder.fit_transform(df["emotion"])

# -----------------------------
# Save encoder
# -----------------------------
joblib.dump(
    label_encoder,
    LABEL_ENCODER_DIR / "label_encoder.pkl"
)

print("\nLabel Encoder Created Successfully!\n")

print("Classes:\n")

for i, emotion in enumerate(label_encoder.classes_):
    print(f"{i} -> {emotion}")

print("\nTotal Classes:", len(label_encoder.classes_))