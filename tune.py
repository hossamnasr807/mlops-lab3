# tune.py
import os
import json
import joblib
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow

# Load Dagshub credentials (from .env file)
load_dotenv()

# Set experiment name
mlflow.set_experiment("Car Evaluation Tuning")

# Create output dir
os.makedirs("models", exist_ok=True)

# Load data
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

X_train = train_df.drop("class", axis=1)
y_train = train_df["class"]

X_test = test_df.drop("class", axis=1)
y_test = test_df["class"]

# Hyperparameter grid
n_estimators_list = [50, 100, 150]
max_depth_list = [5, 10, None]

best_acc = -1
best_params = None
best_model_path = None

with mlflow.start_run(run_name="rf_parent_tuning"):
    mlflow.log_param("search_type", "grid")
    
    for n_est in n_estimators_list:
        for depth in max_depth_list:
            with mlflow.start_run(nested=True, run_name=f"rf_n{n_est}_d{depth}"):
                params = {"n_estimators": n_est, "max_depth": depth, "random_state": 42}
                mlflow.log_params(params)

                model = RandomForestClassifier(**params)
                model.fit(X_train, y_train)

                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)

                mlflow.log_metric("test_accuracy", acc)

                model_path = f"models/rf_{n_est}_{depth}.pkl"
                joblib.dump(model, model_path)
                mlflow.log_artifact(model_path)

                if acc > best_acc:
                    best_acc = acc
                    best_params = params
                    best_model_path = model_path

    # Save best model details
    if best_model_path:
        best_model_final = "models/best_rf.pkl"
        joblib.dump(joblib.load(best_model_path), best_model_final)
        mlflow.log_artifact(best_model_final)
        mlflow.log_metric("best_test_accuracy", best_acc)
        mlflow.log_param("best_params", str(best_params))

# Save best metrics locally for DVC tracking
with open("tune_metrics.json", "w") as f:
    json.dump({
        "best_test_accuracy": best_acc,
        "best_params": best_params
    }, f)

print(f"Best Model: {best_params} -> Accuracy: {best_acc:.4f}")
print("Saved best model to models/best_rf.pkl")
