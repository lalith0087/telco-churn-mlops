import pytest
from pydantic import ValidationError
from schemas import CustomerFeatures

VALID_PAYLOAD = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}


def test_valid_payload_parses():
    features = CustomerFeatures(**VALID_PAYLOAD)
    assert features.tenure == 1


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("gender", "Other"),
        ("Contract", "Quarterly"),
        ("InternetService", "Cable"),
        ("SeniorCitizen", 2),
        ("tenure", -1),
        ("MonthlyCharges", -10.0),
    ],
)
def test_invalid_values_are_rejected(field, bad_value):
    payload = {**VALID_PAYLOAD, field: bad_value}
    with pytest.raises(ValidationError):
        CustomerFeatures(**payload)


def test_missing_field_is_rejected():
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "tenure"}
    with pytest.raises(ValidationError):
        CustomerFeatures(**payload)
