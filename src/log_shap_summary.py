"""Computes a global SHAP summary (beeswarm) plot for the Production model over a sample of
the test set, and logs it as an artifact on that model's MLflow run. This gives anyone
browsing the MLflow UI a feature-importance view for the model currently in Production,
complementing the per-prediction explanations served by the API's /explain endpoint."""
import os

import matplotlib.pyplot as plt
import mlflow
import shap
from mlflow.tracking import MlflowClient

from explain import build_explainer
from features import get_feature_columns, load_clean, split_data
from promote_model import REGISTERED_MODEL_NAME

MLFLOW_TRACKING_URI = "file:" + os.path.join(os.path.dirname(__file__), "..", "mlruns")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
SAMPLE_SIZE = 200


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])
    if not versions:
        raise RuntimeError(f"No Production version found for '{REGISTERED_MODEL_NAME}'. Run promote_model.py first.")

    run_id = versions[0].run_id
    print(f"Logging SHAP summary to run {run_id} ({REGISTERED_MODEL_NAME} v{versions[0].version})")

    pipeline = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/Production")
    preprocessor = pipeline.named_steps["preprocessor"]

    df = load_clean()
    numeric, categorical = get_feature_columns(df)
    _, X_test, _, _ = split_data(df)
    sample = X_test[numeric + categorical].sample(n=min(SAMPLE_SIZE, len(X_test)), random_state=42)

    explainer, feature_names = build_explainer(pipeline)

    transformed = preprocessor.transform(sample)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    shap_values = explainer(transformed)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    plot_path = os.path.join(ARTIFACTS_DIR, "shap_summary.png")

    plt.figure()
    shap.summary_plot(shap_values, transformed, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

    client.log_artifact(run_id, plot_path)
    print(f"Logged {plot_path} to run {run_id}")


if __name__ == "__main__":
    main()
