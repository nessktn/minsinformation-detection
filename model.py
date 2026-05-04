import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def predict(text):
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    probability = model.predict_proba(vec)[0].max()

    return prediction, probability