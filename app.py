# app.py

import os
import pandas as pd
from flask import Flask, request, jsonify
import mlflow
import mlflow.pyfunc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Initialize Flask app
app = Flask(__name__)

# --- Model Loading ---
# We'll need to load the model from the local `mlruns` directory.
# In a real-world scenario, you would use a remote MLflow tracking server.
# For this free, simple setup, we'll assume the model is in the 'mlruns' directory
# that will be present in the container.
print("Loading model...")
try:
    # This will load the latest run's model. In a real-world scenario, you'd
    # specify a specific run ID or a registered model name.
    # For simplicity, we'll assume the model is located at a known path.
    # We will need to adjust this path to match our container's file structure.
    model_path = "mlruns/0/<<YOUR_LATEST_RUN_ID>>/artifacts/random_forest_model"
    model = mlflow.pyfunc.load_model(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


# --- Data Preprocessing ---
# We need to apply the same preprocessing to the new data as we did during training.
# In a robust system, you would save the scaler and encoder, but for simplicity,
# we'll redefine a simple one-hot encoder and assume we can re-fit it.
def preprocess_input(data):
    """
    Preprocesses raw input data to match the model's training data format.
    Note: In a production system, a pre-fitted encoder should be used.
    """
    df = pd.DataFrame([data])

    # We need to fit the encoder on the original training data to ensure
    # all possible categories are known. This is a simplification.
    original_df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    original_df = original_df.drop(columns=['Attrition', 'EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'])

    categorical_cols = original_df.select_dtypes(include=['object']).columns

    # One-hot encode the categorical columns
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(original_df[categorical_cols])

    encoded_cols = pd.DataFrame(encoder.transform(df[categorical_cols]),
                                columns=encoder.get_feature_names_out(categorical_cols))

    # Drop original categorical columns and concatenate encoded ones
    df_processed = df.drop(columns=categorical_cols).reset_index(drop=True)
    df_processed = pd.concat([df_processed, encoded_cols], axis=1)

    # Align columns to match the training data. This is a critical step.
    training_cols = list(original_df.drop(categorical_cols, axis=1).columns) + list(
        encoder.get_feature_names_out(categorical_cols))

    # Reindex with the training columns and fill missing with 0
    df_processed = df_processed.reindex(columns=training_cols, fill_value=0)

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
        # Get data from POST request
        request_data = request.get_json(force=True)

        # Preprocess the data
        processed_data = preprocess_input(request_data)

        # Make prediction
        prediction_proba = model.predict(processed_data)[0]
        prediction_label = "Yes" if prediction_proba > 0.5 else "No"

        response = {
            'prediction': prediction_label,
            'confidence': float(prediction_proba)  # Convert to float for JSON serialization
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    # This is for local development. In production, a WSGI server like Gunicorn is used.
    app.run(host='0.0.0.0', port=5000, debug=True)

with mlflow.start_run() as run:
    run_id = run.info.run_id
    print(f"Run ID: {run_id}")