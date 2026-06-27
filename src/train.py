"""Train multiple churn-prediction models and log params/metrics/artifacts to MLflow."""
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

from features import build_preprocessor, get_feature_columns, load_clean, split_data

MLFLOW_TRACKING_URI = "file:" + os.path.join(os.path.dirname(__file__), "..", "mlruns")
EXPERIMENT_NAME = "telco-churn-prediction"
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")


def get_models() -> dict:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced", random_state=42
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42,
        ),
    }


def log_confusion_matrix(y_test, preds, model_name: str) -> str:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, preds, ax=ax)
    ax.set_title(f"Confusion matrix - {model_name}")
    path = os.path.join(ARTIFACTS_DIR, f"confusion_matrix_{model_name}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def train_and_log(model_name: str, model, X_train, X_test, y_train, y_test, preprocessor) -> str:
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    with mlflow.start_run(run_name=model_name) as run:
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        proba = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, proba),
        }

        mlflow.log_param("model_type", model_name)
        mlflow.log_params({f"model__{k}": v for k, v in model.get_params().items()})
        mlflow.log_metrics(metrics)

        cm_path = log_confusion_matrix(y_test, preds, model_name)
        mlflow.log_artifact(cm_path)

        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        print(f"[{model_name}] {metrics}")
        return run.info.run_id


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_clean()
    numeric, categorical = get_feature_columns(df)
    preprocessor = build_preprocessor(numeric, categorical)
    X_train, X_test, y_train, y_test = split_data(df)

    run_ids = {}
    for name, model in get_models().items():
        run_ids[name] = train_and_log(name, model, X_train, X_test, y_train, y_test, preprocessor)

    print("\nRun IDs:")
    for name, run_id in run_ids.items():
        print(f"  {name}: {run_id}")


if __name__ == "__main__":
    main()
