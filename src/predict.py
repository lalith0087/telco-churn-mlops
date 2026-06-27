"""Load the registered Production model from MLflow and run predictions on new data."""
import argparse
import os

import mlflow
import pandas as pd

from promote_model import REGISTERED_MODEL_NAME

MLFLOW_TRACKING_URI = "file:" + os.path.join(os.path.dirname(__file__), "..", "mlruns")


def load_production_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{REGISTERED_MODEL_NAME}/Production"
    return mlflow.sklearn.load_model(model_uri)


def predict(input_csv: str) -> pd.DataFrame:
    model = load_production_model()
    df = pd.read_csv(input_csv)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    for col in ("customerID", "Churn"):
        if col in df.columns:
            df = df.drop(columns=[col])

    preds = model.predict(df)
    proba = model.predict_proba(df)[:, 1]

    result = df.copy()
    result["churn_prediction"] = preds
    result["churn_probability"] = proba
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", help="Path to a CSV with the same raw columns as the training data.")
    parser.add_argument("--output", default=None, help="Where to write predictions CSV. Prints to stdout if omitted.")
    args = parser.parse_args()

    result = predict(args.input_csv)
    if args.output:
        result.to_csv(args.output, index=False)
        print(f"Wrote predictions to {args.output}")
    else:
        print(result[["churn_prediction", "churn_probability"]].head(20))
