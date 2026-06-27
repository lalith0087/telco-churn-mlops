"""Cleaning, preprocessing pipeline, and train/test split for the Telco churn dataset."""
import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "telco_churn.csv")

TARGET = "Churn"
ID_COL = "customerID"
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]


def load_clean(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    df = df.drop(columns=[ID_COL])
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = [c for c in df.columns if c not in NUMERIC_FEATURES + [TARGET]]
    return NUMERIC_FEATURES, categorical


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


if __name__ == "__main__":
    df = load_clean()
    numeric, categorical = get_feature_columns(df)
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Numeric features: {numeric}")
    print(f"Categorical features: {categorical}")
