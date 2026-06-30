"""SHAP-based explanation for a single prediction from the registered Production pipeline.

Works regardless of which model type is currently in Production: shap.Explainer auto-selects
TreeExplainer for tree models (XGBoost/RandomForest) or falls back to a model-agnostic
explainer for anything else (e.g. LogisticRegression), so this doesn't need to change when
a new model gets promoted.
"""
import os

import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from features import DATA_PATH, get_feature_columns, load_clean

BACKGROUND_SAMPLE_SIZE = 100


def _get_background(preprocessor) -> "pd.DataFrame":
    df = load_clean(DATA_PATH)
    numeric, categorical = get_feature_columns(df)
    X = df[numeric + categorical].sample(n=BACKGROUND_SAMPLE_SIZE, random_state=42)
    return X


def build_explainer(pipeline: Pipeline) -> tuple[shap.Explainer, "pd.Index"]:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    background_raw = _get_background(preprocessor)
    background_transformed = preprocessor.transform(background_raw)
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(background_transformed, "toarray"):
        background_transformed = background_transformed.toarray()

    explainer = shap.Explainer(model, background_transformed, feature_names=feature_names)
    return explainer, feature_names


def explain_instance(pipeline: Pipeline, explainer: shap.Explainer, row: "pd.DataFrame", top_n: int = 5) -> list[dict]:
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(row)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    shap_values = explainer(transformed)
    values = shap_values.values[0]

    feature_names = preprocessor.get_feature_names_out()
    contributions = sorted(
        zip(feature_names, values), key=lambda pair: abs(pair[1]), reverse=True
    )[:top_n]

    return [
        {"feature": name, "shap_value": float(value)}
        for name, value in contributions
    ]


def explain_batch(pipeline: Pipeline, explainer: shap.Explainer, rows: "pd.DataFrame", top_n: int = 5) -> list[list[dict]]:
    """Same as explain_instance but vectorized over many rows in one SHAP call."""
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(rows)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    shap_values = explainer(transformed)
    feature_names = preprocessor.get_feature_names_out()

    results = []
    for values in shap_values.values:
        contributions = sorted(
            zip(feature_names, values), key=lambda pair: abs(pair[1]), reverse=True
        )[:top_n]
        results.append([{"feature": name, "shap_value": float(value)} for name, value in contributions])
    return results


def aggregate_churn_drivers(per_customer_contributions: list[list[dict]], top_n: int = 5) -> list[dict]:
    """Collapses per-customer SHAP contributions into the drivers most often pushing customers toward churn.

    Only counts positive contributions (push toward churn) since the goal is identifying
    shared risk factors a retention team can act on across a group, not retention factors.
    """
    stats: dict[str, list[float]] = {}
    for contributions in per_customer_contributions:
        for c in contributions:
            if c["shap_value"] > 0:
                stats.setdefault(c["feature"], []).append(c["shap_value"])

    ranked = sorted(stats.items(), key=lambda item: len(item[1]), reverse=True)[:top_n]
    return [
        {
            "feature": feature,
            "customers_affected": len(values),
            "avg_shap_value": sum(values) / len(values),
        }
        for feature, values in ranked
    ]


if __name__ == "__main__":
    import mlflow

    mlflow.set_tracking_uri("file:" + os.path.join(os.path.dirname(__file__), "..", "mlruns"))
    pipeline = mlflow.sklearn.load_model("models:/telco-churn-classifier/Production")

    explainer, _ = build_explainer(pipeline)

    df = load_clean(DATA_PATH)
    numeric, categorical = get_feature_columns(df)
    sample_row = df[numeric + categorical].iloc[[0]]

    for contribution in explain_instance(pipeline, explainer, sample_row):
        print(contribution)
