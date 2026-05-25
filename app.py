from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model/disease_model.pkl", "rb"))
encoder = pickle.load(open("model/encoder.pkl", "rb"))

symptoms = [
    "fever",
    "cough",
    "headache",
    "nausea",
    "vomiting",
    "fatigue",
    "sore_throat",
    "chills",
    "body_pain",
    "loss_of_appetite",
    "abdominal_pain",
    "diarrhea",
    "sweating",
    "rapid_breathing",
    "dizziness"
]

@app.route("/")
def home():
    return render_template(
        "index.html",
        symptoms=symptoms
    )

@app.route("/predict", methods=["POST"])
def predict():

    values = []

    for symptom in symptoms:
        values.append(
            int(request.form[symptom])
        )

    values = np.array(values).reshape(1, -1)

    prediction = model.predict(values)[0]

    disease = encoder.inverse_transform(
        [prediction]
    )[0]

    confidence = max(
        model.predict_proba(values)[0]
    ) * 100

    return render_template(
        "result.html",
        disease=disease,
        confidence=round(confidence,2)
    )

if __name__ == "__main__":
    app.run(debug=True)