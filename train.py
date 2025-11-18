# train_with_mlflow.py
"""
More realistic MLflow usage example:
- Simulate reading prepared features
- Run multiple experiments with different hyperparameters
- Log parameters, metrics, and feature set
"""

from typing import List, Dict, Any
import json
import os

import mlflow
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib


def load_data(feature_cols: List[str]) -> Dict[str, Any]:
    """
    Here we simulate it with make_classification, and pretend that
    each column has a human-readable feature name.
    """
    n_samples = 2000
    n_features = len(feature_cols)

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        random_state=42,
    )

    return {
        "X": X,
        "y": y,
        "n_samples": n_samples,
    }


def train_one_run(
    feature_cols: List[str],
    test_size: float,
    random_seed: int,
    C: float,
    penalty: str,
) -> None:
    """
    Train a single logistic regression model and log everything to MLflow.
    """

    # --- Data loading / splitting ---
    data = load_data(feature_cols)
    X, y = data["X"], data["y"]
    n_samples = data["n_samples"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )

    # --- Start MLflow run ---
    with mlflow.start_run(run_name=f"logreg_C{C}_penalty_{penalty}_seed{random_seed}"):

        # 🔹 Log configuration / parameters
        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_param("n_samples", n_samples)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_seed", random_seed)
        mlflow.log_param("C", C)
        mlflow.log_param("penalty", penalty)

        # Record feature set as a single param (string)
        # Example: "age,income,transactions_last_30d"
        feature_set_str = ",".join(feature_cols)
        mlflow.log_param("feature_set", feature_set_str)

        # ALSO: record feature_set as a structured artifact (json)
        feature_config = {"feature_cols": feature_cols}
        os.makedirs("artifacts", exist_ok=True)
        run_id = mlflow.active_run().info.run_id  
        feature_config_path = os.path.join("artifacts", f"feature_config_{run_id}.json")
        with open(feature_config_path, "w", encoding="utf-8") as f:
            json.dump(feature_config, f, indent=2)
        mlflow.log_artifact(feature_config_path, artifact_path="config")

        # --- Train model ---
        model = LogisticRegression(C=C, penalty=penalty, solver="liblinear")
        model.fit(X_train, y_train)

        # --- Evaluate ---
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Here we only log accuracy,
        # but in a real project you might log multiple metrics.
        mlflow.log_metric("test_accuracy", acc)

        print(
            f"[Run] C={C}, penalty={penalty}, seed={random_seed} "
            f"-> test_accuracy={acc:.4f}"
        )

        # --- Save model and log as artifact ---
        filename = f"model_C{C}_penalty_{penalty}_seed{random_seed}.pkl"
        model_path = f"artifacts/{filename}"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path, artifact_path="models")


def main():
    # Group multiple runs under one experiment name
    mlflow.set_experiment("logreg-demo")

    # Suppose these are the feature names used in the model
    feature_cols = [
        "age",
        "income",
        "num_transactions_last_30d",
        "avg_transaction_amount",
        "num_chargebacks_last_90d",
    ]

    test_size = 0.2

    # Hyperparameter grid (more realistic than a single run)
    C_values = [0.1, 1.0, 10.0]
    penalties = ["l1", "l2"]
    seeds = [42, 43]  # multiple seeds = multiple runs with same params

    for seed in seeds:
        for C in C_values:
            for penalty in penalties:
                train_one_run(
                    feature_cols=feature_cols,
                    test_size=test_size,
                    random_seed=seed,
                    C=C,
                    penalty=penalty,
                )


if __name__ == "__main__":
    main()
