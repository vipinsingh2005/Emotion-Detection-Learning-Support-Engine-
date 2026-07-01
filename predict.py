import os
import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from config import (
    TOKENIZER_DIR,
    LABEL_ENCODER_DIR,
    BILSTM_MODEL_DIR,
    MAX_SEQUENCE_LENGTH
)


class EmotionPredictor:

    def __init__(self):
        self.tokenizer = None
        self.label_encoder = None
        self.model = None

        self.load_components()

    def load_components(self):

        tokenizer_path = os.path.join(
            TOKENIZER_DIR,
            "tokenizer.pkl"
        )

        label_encoder_path = os.path.join(
            LABEL_ENCODER_DIR,
            "label_encoder.pkl"
        )

        model_path = os.path.join(
            BILSTM_MODEL_DIR,
            "bilstm_final_model.keras"
        )

        with open(tokenizer_path, "rb") as f:
            self.tokenizer = pickle.load(f)

        with open(label_encoder_path, "rb") as f:
            self.label_encoder = pickle.load(f)

        if os.path.exists(model_path):
            self.model = load_model(model_path)
        else:
            print("Model not found.")
            self.model = None

    def preprocess(self, text):

        sequence = self.tokenizer.texts_to_sequences([text])

        padded = pad_sequences(
            sequence,
            maxlen=MAX_SEQUENCE_LENGTH,
            padding="post"
        )

        return padded

    def predict(self, text):

        if self.model is None:
            return {
                "emotion": "Model Not Trained",
                "confidence": 0.0
            }

        x = self.preprocess(text)

        prediction = self.model.predict(x, verbose=0)

        index = np.argmax(prediction)

        emotion = self.label_encoder.inverse_transform([index])[0]

        confidence = float(np.max(prediction))

        return {
            "emotion": emotion,
            "confidence": confidence
        }