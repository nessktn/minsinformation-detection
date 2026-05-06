import pickle

pipeline = pickle.load(open("pipeline.pkl", "rb"))

def predict(text):
    prediction = pipeline.predict([text])[0]
    probability = pipeline.predict_proba([text])[0].max()
    return prediction, probability