import pandas as pd
import joblib
from flask import Flask, request, jsonify

# --- 1. Initialize the Flask App ---
app = Flask(__name__)

# --- 2. Load the Trained Model ---
# This is done once when the server starts.
model_filename = 'diabetes_model.joblib'
try:
    model = joblib.load(model_filename)
    print(f"Model '{model_filename}' loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file '{model_filename}' not found. Please run the training script first.")
    model = None

# --- Add a Welcome/Root Endpoint ---
@app.route('/', methods=['GET'])
def welcome():
    """A simple welcome message to confirm the API is running."""
    return "<h1>Diabetes Prediction API</h1><p>The API is running. Use the /predict endpoint with a POST request to get a prediction.</p>"

# --- 3. Define the Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives patient data in JSON format, makes a prediction, and returns it.
    """
    if model is None:
        return jsonify({"error": "Model is not loaded."}), 500

    # Get the JSON data sent from n8n or another client
    json_data = request.get_json()
    print(f"Received data: {json_data}")

    # The features our model expects, in the correct order
    feature_names = ['gender', 'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    
    # Create a pandas DataFrame from the received data
    data_for_prediction = pd.DataFrame([json_data], columns=feature_names)

    # Make the prediction (the result is a numpy array, e.g., [1])
    prediction = model.predict(data_for_prediction)
    
    # Return the prediction as a JSON response (we extract the single value)
    return jsonify({'predicted_diabetes': int(prediction[0])})

# --- 4. Run the App ---
if __name__ == '__main__':
    # The app will be accessible at http://127.0.0.1:5000
    app.run(debug=True, port=5000)
