"""FastAPI service that serves the Production-stage churn model from the MLflow registry."""
import os

import mlflow
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient

from explain import build_explainer, explain_instance
from narrate import narrate
from promote_model import REGISTERED_MODEL_NAME
from schemas import (
    CustomerFeatures,
    ExplanationResponse,
    FeatureContribution,
    HealthResponse,
    NarrativeExplanationResponse,
    PredictionResponse,
)

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
_explainer = None


@app.on_event("startup")
def load_model() -> None:
    global _model, _model_version, _explainer
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

    _explainer, _ = build_explainer(_model)
    print("SHAP explainer ready")


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


@app.post("/explain", response_model=ExplanationResponse)
def explain(features: CustomerFeatures) -> ExplanationResponse:
    if _model is None or _explainer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    row = pd.DataFrame([features.model_dump()])
    pred = int(_model.predict(row)[0])
    proba = float(_model.predict_proba(row)[0, 1])

    contributions = explain_instance(_model, _explainer, row, top_n=5)

    return ExplanationResponse(
        churn_prediction=pred,
        churn_probability=proba,
        model_version=_model_version,
        top_contributing_features=[FeatureContribution(**c) for c in contributions],
    )


@app.post("/explain-narrative", response_model=NarrativeExplanationResponse)
def explain_narrative(features: CustomerFeatures) -> NarrativeExplanationResponse:
    if _model is None or _explainer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    row = pd.DataFrame([features.model_dump()])
    pred = int(_model.predict(row)[0])
    proba = float(_model.predict_proba(row)[0, 1])

    contributions = explain_instance(_model, _explainer, row, top_n=5)

    try:
        narrative = narrate(pred, proba, contributions)
    except requests.exceptions.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Ollama unavailable: {exc}") from exc

    return NarrativeExplanationResponse(
        churn_prediction=pred,
        churn_probability=proba,
        model_version=_model_version,
        top_contributing_features=[FeatureContribution(**c) for c in contributions],
        narrative=narrative,
    )
