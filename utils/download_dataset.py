"""
=========================================================
Emotion Detection & Learning Support Engine
Dataset Download Module
=========================================================

This module downloads the GoEmotions dataset and stores
it inside the dataset/raw folder.
"""
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from datasets import load_dataset
import pandas as pd

from config import RAW_DATASET_DIR


def download_goemotions():
    """
    Download the Google GoEmotions dataset
    and save it as CSV files.
    """

    print("Downloading GoEmotions dataset...")

    dataset = load_dataset("google-research-datasets/go_emotions")

    RAW_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    train_df = dataset["train"].to_pandas()
    validation_df = dataset["validation"].to_pandas()
    test_df = dataset["test"].to_pandas()

    train_df.to_csv(
        RAW_DATASET_DIR / "train.csv",
        index=False
    )

    validation_df.to_csv(
        RAW_DATASET_DIR / "validation.csv",
        index=False
    )

    test_df.to_csv(
        RAW_DATASET_DIR / "test.csv",
        index=False
    )

    print("Dataset downloaded successfully!")


if __name__ == "__main__":
    download_goemotions()