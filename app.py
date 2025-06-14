from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

@app.route("/")
def index():
    df = pd.read_csv("dataset.csv")
    symptoms = df.columns[:-1].tolist()
    return render_template("index.html", symptoms=symptoms)


@app.route("/predict", methods=["POST"])
def predict():
    selected_symptoms = request.form.getlist("symptoms")
    df = pd.read_csv("dataset.csv")
    symptom_columns = df.columns[:-1].tolist()
    input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptom_columns]

    model = pickle.load(open("model.pkl", "rb"))
    prediction = model.predict([input_vector])[0]

    return render_template("result.html", prediction=prediction)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)