from narrate import _build_prompt, _humanize_feature


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
