# import joblib
# import pandas as pd
# from flask import Flask, request, jsonify

# # Load the model
# model = joblib.load("Transaction_fraud_detect.pkl")

# app = Flask(__name__)

# @app.route('/')
# def home():
#     return "Fraud Detection API is running!"

# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         # Get JSON data
#         data = request.get_json()
        
#         # Convert data into DataFrame
#         df = pd.DataFrame(data)
        
#         # Make prediction
#         prediction = model.predict(df)
#         probability = model.predict_proba(df)[:, 1]

#         return jsonify({
#             "prediction": int(prediction[0]),
#             "fraud_probability": round(float(probability[0]), 4)
#         })
    
#     except Exception as e:
#         return jsonify({"error": str(e)})

# if __name__ == "__main__":

#     app.run(host="0.0.0.0", port=5000)



import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Load the model
model = joblib.load("Transaction_fraud_detect.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return "Fraud Detection API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data
        data = request.get_json()

        # Convert to DataFrame correctly
        if isinstance(data, dict):  
            df = pd.DataFrame([data])  # Convert single JSON object to DataFrame
        else:
            df = pd.DataFrame(data)  # If list of JSON objects, use directly

        # Make prediction
        prediction = model.predict(df)
        probability = model.predict_proba(df)[:, 1]

        return jsonify({
            "prediction": int(prediction[0]),
            "fraud_probability": round(float(probability[0]), 4)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
