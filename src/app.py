"""FastAPI service that serves the Production-stage churn model from the MLflow registry."""
import os

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient

from promote_model import REGISTERED_MODEL_NAME
from schemas import CustomerFeatures, HealthResponse, PredictionResponse

MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", "file:" + os.path.join(os.path.dirname(__file__), "..", "mlruns")
)

app = FastAPI(
    title="Telco Churn Prediction API",
    description="Serves the Production-stage model registered in MLflow for the telco-churn-classifier.",
    version="1.0.0",
)

_model = None
_model_version = None


@app.on_event("startup")
def load_model() -> None:
    global _model, _model_version
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])
    if not versions:
        raise RuntimeError(
            f"No Production version found for '{REGISTERED_MODEL_NAME}'. Run promote_model.py first."
        )

    _model_version = str(versions[0].version)
    _model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/Production")
    print(f"Loaded {REGISTERED_MODEL_NAME} v{_model_version} (Production)")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return HealthResponse(status="ok", model_name=REGISTERED_MODEL_NAME, model_version=_model_version)


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures) -> PredictionResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    row = pd.DataFrame([features.model_dump()])
    pred = int(_model.predict(row)[0])
    proba = float(_model.predict_proba(row)[0, 1])

    return PredictionResponse(
        churn_prediction=pred,
        churn_probability=proba,
        model_version=_model_version,
    )
