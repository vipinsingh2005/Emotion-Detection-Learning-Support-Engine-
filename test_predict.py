from predict import EmotionPredictor

predictor = EmotionPredictor()

result = predictor.predict(
    "I am feeling stressed because of exams."
)

print(result)