import pickle
import pandas as pd
from flask import Flask, render_template, request
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)

# Load model and feature names
model, feature_names, le = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def index():
    return render_template('index.html', symptoms=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    selected_symptoms = request.form.getlist('symptoms')

    # Create input vector with correct 24 features
    input_vector = [1 if symptom in selected_symptoms else 0 for symptom in feature_names]

    pred_encoded = model.predict([input_vector])[0]
    prediction = le.inverse_transform([pred_encoded])[0]

    return render_template('result.html', prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
