from narrate import _build_batch_prompt, _build_prompt, _humanize_feature


def test_humanize_feature_strips_prefix_and_underscores():
    assert _humanize_feature("num__tenure") == "tenure"
    assert _humanize_feature("cat__Contract_Month-to-month") == "Contract Month-to-month"


def test_build_prompt_includes_probability_and_direction():
    contributions = [
        {"feature": "num__tenure", "shap_value": 1.08},
        {"feature": "cat__OnlineSecurity_No", "shap_value": -0.3},
    ]
    prompt = _build_prompt(prediction=1, probability=0.71, contributions=contributions)

    assert "71%" in prompt
    assert "likely to churn" in prompt
    assert "pushes toward churn" in prompt
    assert "pushes toward retention" in prompt
    assert "Do not mention SHAP" in prompt


def test_build_prompt_reflects_retention_verdict_when_not_churning():
    prompt = _build_prompt(prediction=0, probability=0.12, contributions=[])
    assert "likely to stay" in prompt


def test_build_batch_prompt_includes_counts_and_drivers():
    drivers = [
        {"feature": "cat__Contract_Month-to-month", "customers_affected": 18, "avg_shap_value": 0.55},
        {"feature": "num__tenure", "customers_affected": 12, "avg_shap_value": 0.91},
    ]
    prompt = _build_batch_prompt(
        total_customers=50, high_risk_count=23, average_probability=0.46, top_shared_drivers=drivers
    )

    assert "50 customers" in prompt
    assert "23" in prompt
    assert "46%" in prompt
    assert "18 of 50" in prompt
    assert "Contract Month-to-month" in prompt
    assert "Do not mention SHAP" in prompt


def test_build_batch_prompt_handles_no_shared_drivers():
    prompt = _build_batch_prompt(
        total_customers=10, high_risk_count=0, average_probability=0.05, top_shared_drivers=[]
    )
    assert "no shared churn drivers detected" in prompt
