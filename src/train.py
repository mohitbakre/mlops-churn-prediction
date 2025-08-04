# src/train.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature


def train_model():
    """
    Loads data, trains a RandomForest model, and logs metrics with MLflow.
    """
    print("Starting MLflow experiment...")

    # --- FIX: Explicitly set MLflow tracking URI to a relative path ---
    # This ensures MLflow stores artifacts in a local 'mlruns' directory
    # relative to where the script is run, preventing issues with absolute
    # Windows paths on Linux environments (like CI/CD).
    mlflow.set_tracking_uri("file:./mlruns")

    mlflow.set_experiment("HR_Attrition_Prediction_Experiment")

    with mlflow.start_run():
        # --- Data Loading and Preprocessing ---
        print("Loading and preprocessing data...")
        try:
            # Ensure the correct dataset path
            df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
        except FileNotFoundError:
            print("The dataset was not found. Please ensure it is in the 'data/' directory.")
            return

        # Convert 'Attrition' to numerical (0 or 1)
        df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)

        # Perform one-hot encoding for categorical variables
        # drop_first=True avoids multicollinearity by dropping the first category
        df = pd.get_dummies(df, drop_first=True)

        # Drop columns with no variance or are not relevant for modeling.
        # 'errors='ignore'' ensures the script doesn't fail if a column is already
        # dropped by get_dummies (e.g., 'Over18' if it has only one unique value)
        # or if it's simply not present in a different version of the dataset.
        df = df.drop(columns=['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], errors='ignore')

        # Split data into features (X) and target (y)
        X = df.drop('Attrition', axis=1)
        y = df['Attrition']

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # --- Model Training ---
        print("Training RandomForestClassifier...")
        # Define hyperparameters for MLflow logging
        n_estimators = 100 # Number of trees in the forest
        max_depth = 10     # Maximum depth of the tree

        # Log parameters to MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # Initialize and train the RandomForestClassifier model
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        # --- Evaluation ---
        print("Evaluating model...")
        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=1) # pos_label=1 for 'Yes' attrition
        recall = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)

        # Log metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        print(f"Model Metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")

        # --- Model Saving ---
        print("Saving model artifact...")
        # Infer the model signature from the input and output data
        signature = infer_signature(X_train, model.predict(X_train))

        # Log the trained scikit-learn model as an MLflow artifact
        # Using 'artifact_path' for the path within the MLflow run's artifact storage
        mlflow.sklearn.log_model(
            sk_model=model,
            name="random_forest_model",
            signature=signature,
            input_example=X_train.head(5) # Provide a small sample of input data
        )

        # Save some example predictions as an artifact for later review
        predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        predictions_df.to_csv("predictions.csv", index=False)
        mlflow.log_artifact("predictions.csv")


if __name__ == "__main__":
    train_model()
