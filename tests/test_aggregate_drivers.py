from explain import aggregate_churn_drivers


def test_aggregate_counts_customers_affected_per_feature():
    per_customer = [
        [
            {"feature": "cat__Contract_Month-to-month", "shap_value": 0.6},
            {"feature": "num__tenure", "shap_value": 0.9},
        ],
        [
            {"feature": "cat__Contract_Month-to-month", "shap_value": 0.4},
            {"feature": "cat__OnlineSecurity_No", "shap_value": 0.2},
        ],
    ]

    result = aggregate_churn_drivers(per_customer)
    by_feature = {d["feature"]: d for d in result}

    assert by_feature["cat__Contract_Month-to-month"]["customers_affected"] == 2
    assert by_feature["cat__Contract_Month-to-month"]["avg_shap_value"] == 0.5
    assert by_feature["num__tenure"]["customers_affected"] == 1


def test_aggregate_ignores_retention_pushing_contributions():
    per_customer = [[{"feature": "num__tenure", "shap_value": -0.9}]]
    result = aggregate_churn_drivers(per_customer)
    assert result == []


def test_aggregate_respects_top_n():
    per_customer = [
        [{"feature": f"feature_{i}", "shap_value": 0.1 * i} for i in range(1, 8)]
    ]
    result = aggregate_churn_drivers(per_customer, top_n=3)
    assert len(result) == 3
