import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


# 1. Load dataset
df = pd.read_excel("labeled2.xlsx")

# 2. Map original emotions → stress/anxiety/depression
def map_emotion(emotion):
    if emotion == "fear":
        return "anxiety"
    elif emotion == "sadness":
        return "depression"
    elif emotion == "anger":
        return "stress"
    else:
        return None  # ignore other emotions
df["res"] = df["label"].apply(map_emotion)
df = df.dropna(subset=["res"])   # keep only mapped rows

print("Samples per class:")
print(df["res"].value_counts())
