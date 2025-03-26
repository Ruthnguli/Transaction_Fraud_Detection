from flask import Flask, request, jsonify
import joblib
import numpy as np


# Load the trained model
model = joblib.load("Transaction_fraud_detection.pkl")

# Initialize Flask app

_name_ = "app"

# Create a Flask app instance
app = Flask(_name_)


@app.route("/")
def home():
    return "Fraud Detection API is running!"



@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']
    prediction = model.predict(np.array([data]))
    return jsonify({'fraud_prediction': int(prediction[0])})

if _name_ == '_main_':
    app.run(host='0.0.0.0', port=5000)