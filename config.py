"""
=========================================================
Emotion Detection & Learning Support Engine
Configuration Module
=========================================================

This file contains all global configuration variables
used across the project.

Author : Team Google Cloud GenAI
"""

from pathlib import Path
import os
try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    import warnings

    warnings.warn(
        "python-dotenv not installed; skipping loading .env file. "
        "Install with: pip install python-dotenv",
        ImportWarning,
    )

    def load_dotenv(*args, **kwargs):
        return False

# =========================================================
# Load Environment Variables
# =========================================================

load_dotenv()

# =========================================================
# Base Project Directory
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

# =========================================================
# Dataset Paths
# =========================================================

DATASET_DIR = BASE_DIR / "dataset"

RAW_DATASET_DIR = DATASET_DIR / "raw"

PROCESSED_DATASET_DIR = DATASET_DIR / "processed"

# =========================================================
# Model Paths
# =========================================================

MODEL_DIR = BASE_DIR / "models"

BILSTM_MODEL_DIR = MODEL_DIR / "bilstm"

BERT_MODEL_DIR = MODEL_DIR / "bert"

TOKENIZER_DIR = MODEL_DIR / "tokenizer"

LABEL_ENCODER_DIR = MODEL_DIR / "label_encoder"

# =========================================================
# Assets
# =========================================================

ASSETS_DIR = BASE_DIR / "assets"

CSS_DIR = ASSETS_DIR / "css"

IMAGE_DIR = ASSETS_DIR / "images"

ICON_DIR = ASSETS_DIR / "icons"

# =========================================================
# Logging
# =========================================================

LOG_DIR = BASE_DIR / "logs"

LOG_FILE = LOG_DIR / "application.log"

# =========================================================
# Prediction History
# =========================================================

HISTORY_FILE = BASE_DIR / "history.csv"

# =========================================================
# Gemini API
# =========================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# =========================================================
# Model Configuration
# =========================================================

MAX_SEQUENCE_LENGTH = 128

VOCAB_SIZE = 23308

EMBEDDING_DIM = 128

BATCH_SIZE = 32

EPOCHS = 10

LEARNING_RATE = 0.001

TEST_SIZE = 0.20

RANDOM_STATE = 42

# =========================================================
# Supported Emotions
# =========================================================

EMOTIONS = [
    "Angry",
    "Confused",
    "Curious",
    "Frustrated",
    "Happy",
    "Neutral",
    "Sad"
]

# =========================================================
# Streamlit
# =========================================================

APP_TITLE = "🎓 Emotion Detection & Learning Support Engine"

PAGE_ICON = "🧠"

LAYOUT = "wide"



