# app.py

import os
import pandas as pd
from flask import Flask, request, jsonify
import mlflow
import mlflow.pyfunc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import glob
import time

# Initialize Flask app
app = Flask(__name__)

# --- Model Loading ---
print("Loading model...")
# Use the correct path to the most recent model version found
model_path = "mlruns/706661852386230193/models/m-ee453bc25b5e4f54805db27e1e3d65ee/artifacts"
model = None

try:
    # Use MLflow's native loading function with the direct path
    model = mlflow.pyfunc.load_model(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Could not find a model to load.")

# --- Data Preprocessing ---
def preprocess_input(data):
    df = pd.DataFrame([data])

    original_df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    original_df = original_df.drop(columns=['Attrition', 'EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'])

    categorical_cols = original_df.select_dtypes(include=['object']).columns

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(original_df[categorical_cols])

    encoded_cols = pd.DataFrame(encoder.transform(df[categorical_cols]),
                                columns=encoder.get_feature_names_out(categorical_cols))

    df_processed = df.drop(columns=categorical_cols).reset_index(drop=True)
    df_processed = pd.concat([df_processed, encoded_cols], axis=1)

    training_cols = list(original_df.drop(categorical_cols, axis=1).columns) + list(
        encoder.get_feature_names_out(categorical_cols))

    df_processed = df_processed.reindex(columns=training_cols, fill_value=0)

    # --- FIX: Convert boolean-like columns to the correct data type ---
    boolean_cols = [col for col in df_processed.columns if col.startswith(('BusinessTravel_', 'Department_', 'EducationField_', 'Gender_', 'JobRole_', 'MaritalStatus_', 'OverTime_'))]
    for col in boolean_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(bool)

    return df_processed


# --- API Endpoints ---
@app.route('/')
def home():
    return "ML Prediction API is running!"


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model is not loaded. Check server logs.'}), 500

    try:
        request_data = request.get_json(force=True)
        processed_data = preprocess_input(request_data)

        prediction_proba = model.predict(processed_data)[0]
        prediction_label = "Yes" if prediction_proba > 0.5 else "No"

        response = {
            'prediction': prediction_label,
            'confidence': float(prediction_proba)
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)