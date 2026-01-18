from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

def detect_emotion(text):
    results = classifier(text)[0]
    top = max(results, key=lambda x: x["score"])
    return top["label"], top["score"]

if __name__ == "__main__":
    text = input("Write how you feel: ")
    emotion, confidence = detect_emotion(text)
    print(f"Detected emotion: {emotion} (confidence: {confidence:.2f})")
# emotion_detector.py from transformers import pipeline