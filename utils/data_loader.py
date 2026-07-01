"""
data_loader.py

Loads the GoEmotions dataset and converts it into
Pandas DataFrames.
"""

from datasets import load_dataset
import pandas as pd


def load_goemotions():
    """
    Load GoEmotions dataset from HuggingFace.

    Returns
    -------
    train_df : pd.DataFrame
    validation_df : pd.DataFrame
    test_df : pd.DataFrame
    """

    print("Loading GoEmotions dataset...")

    dataset = load_dataset(
        "google-research-datasets/go_emotions",
        "simplified"
    )

    train_df = pd.DataFrame(dataset["train"])
    validation_df = pd.DataFrame(dataset["validation"])
    test_df = pd.DataFrame(dataset["test"])

    print("Dataset loaded successfully!")

    return train_df, validation_df, test_df


if __name__ == "__main__":

    train_df, val_df, test_df = load_goemotions()

    print("\nTraining Samples:", len(train_df))
    print("Validation Samples:", len(val_df))
    print("Testing Samples:", len(test_df))

    print("\nFirst Five Rows:\n")
    print(train_df.head())