import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
"""
dataset_splitter.py

Creates processed datasets for model training.
"""

import os
import pandas as pd

from config import PROCESSED_DATASET_DIR
from utils.data_loader import load_goemotions
from utils.text_cleaner import clean_text
from utils.label_mapper import map_label


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean text and map labels.
    """

    processed_rows = []

    for _, row in df.iterrows():

        labels = row["labels"]

        # Skip samples with multiple labels
        if len(labels) != 1:
            continue

        label = map_label(labels[0])

        processed_rows.append(
            {
                "text": clean_text(row["text"]),
                "emotion": label
            }
        )

    return pd.DataFrame(processed_rows)


def save_processed_data():

    train_df, val_df, test_df = load_goemotions()

    print("Processing training dataset...")
    train_df = process_dataframe(train_df)

    print("Processing validation dataset...")
    val_df = process_dataframe(val_df)

    print("Processing testing dataset...")
    test_df = process_dataframe(test_df)

    os.makedirs(PROCESSED_DATASET_DIR, exist_ok=True)

    train_df.to_csv(
        os.path.join(PROCESSED_DATASET_DIR, "train.csv"),
        index=False
    )

    val_df.to_csv(
        os.path.join(PROCESSED_DATASET_DIR, "validation.csv"),
        index=False
    )

    test_df.to_csv(
        os.path.join(PROCESSED_DATASET_DIR, "test.csv"),
        index=False
    )

    print("\nDatasets saved successfully!")

    print("\nTraining:", len(train_df))
    print("Validation:", len(val_df))
    print("Testing:", len(test_df))


if __name__ == "__main__":
    save_processed_data()