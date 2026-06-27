"""Pick the best run by a metric and register/promote it to Production in the MLflow Model Registry."""
import argparse
import os

import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = "file:" + os.path.join(os.path.dirname(__file__), "..", "mlruns")
EXPERIMENT_NAME = "telco-churn-prediction"
REGISTERED_MODEL_NAME = "telco-churn-classifier"


def get_best_run(client: MlflowClient, experiment_id: str, metric: str):
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No runs found in experiment.")
    return runs[0]


def promote(metric: str = "roc_auc") -> str:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found. Run train.py first.")

    best_run = get_best_run(client, experiment.experiment_id, metric)
    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"

    print(f"Best run: {run_id} ({best_run.data.params.get('model_type')}), "
          f"{metric}={best_run.data.metrics.get(metric):.4f}")

    result = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)

    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=result.version,
        stage="Production",
        archive_existing_versions=True,
    )

    print(f"Registered '{REGISTERED_MODEL_NAME}' v{result.version} -> Production")
    return result.version


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default="roc_auc", help="Metric to select the best run by.")
    args = parser.parse_args()
    promote(args.metric)
