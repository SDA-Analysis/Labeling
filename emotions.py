import pandas as pd
from transformers import pipeline

# Load Excel
df = pd.read_excel("Twitter_Full.xlsx")

# Use a public model
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")


def classify_tweet(text):
    result = classifier(str(text), top_k=1)[0]  # take the most likely emotion
    return result["label"], result["score"]
df[["label", "confidence"]] = df["text"].apply(lambda x: pd.Series(classify_tweet(x)))
# Save results
df.to_excel("tweets_labeled.xlsx", index=False)
print("✅ Done! Saved tweets_labeled.xlsx")
