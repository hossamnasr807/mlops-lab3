import os
import json
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv
import mlflow

# Load Dagshub MLflow credentials from .env
load_dotenv()

# Set experiment name on Dagshub
mlflow.set_experiment("Car Evaluation Baseline")

# Load preprocessed training data
train_df = pd.read_csv('data/train.csv')
X_train = train_df.drop('class', axis=1)
y_train = train_df['class']

# Train Logistic Regression model
model = LogisticRegression(max_iter=2000, random_state=42)
model.fit(X_train, y_train)

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

# Save trained model locally
model_path = 'models/model.pkl'
joblib.dump(model, model_path)
print(f"Saved model to {model_path}")

# Compute training accuracy
train_acc = accuracy_score(y_train, model.predict(X_train))

# Save metrics to JSON for DVC tracking
metrics_path = 'train_metrics.json'
with open(metrics_path, 'w') as f:
    json.dump({'train_accuracy': train_acc}, f)
print(f"Saved training metrics to {metrics_path}")

# Log to MLflow (DagsHub)
with mlflow.start_run(run_name="logreg_baseline"):
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_artifact(model_path)
    mlflow.log_artifact(metrics_path)

print(f"Logged experiment to MLflow (Car Evaluation Baseline)")
