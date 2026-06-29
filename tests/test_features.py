import pandas as pd
import pytest
from features import DATA_PATH, build_preprocessor, get_feature_columns, load_clean, split_data


@pytest.fixture(scope="module")
def df():
    return load_clean(DATA_PATH)


def test_load_clean_drops_id_and_encodes_target(df):
    assert "customerID" not in df.columns
    assert set(df["Churn"].unique()) <= {0, 1}


def test_load_clean_coerces_total_charges_numeric(df):
    assert pd.api.types.is_numeric_dtype(df["TotalCharges"])
    assert df["TotalCharges"].isna().sum() == 0


def test_get_feature_columns_excludes_target(df):
    numeric, categorical = get_feature_columns(df)
    assert "Churn" not in numeric
    assert "Churn" not in categorical
    assert set(numeric) == {"tenure", "MonthlyCharges", "TotalCharges"}


def test_build_preprocessor_transforms_to_expected_columns(df):
    numeric, categorical = get_feature_columns(df)
    preprocessor = build_preprocessor(numeric, categorical)
    transformed = preprocessor.fit_transform(df[numeric + categorical])
    assert transformed.shape[0] == len(df)


def test_split_data_is_stratified_and_disjoint(df):
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)
    assert len(X_train) + len(X_test) == len(df)
    assert set(X_train.index).isdisjoint(set(X_test.index))

    train_churn_rate = y_train.mean()
    test_churn_rate = y_test.mean()
    assert abs(train_churn_rate - test_churn_rate) < 0.02
