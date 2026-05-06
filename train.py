import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import pickle

# Load dataset
df = pd.read_csv("data/dataset.csv")

# Features & labels
X = df["text"]
y = df["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1,2),
    max_df=0.9
)

pipeline = Pipeline([
    ("vectorizer", vectorizer),
    ("classifier", LogisticRegression(max_iter=1000))
])

scores = cross_val_score(pipeline, X, y, cv=5)
print("Cross-validation accuracy:", scores.mean())

pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Save pipeline
pickle.dump(pipeline, open("pipeline.pkl", "wb"))

# Write report
report = classification_report(y_test, y_pred)
with open("metrics.txt", "w") as f:
    f.write(report)