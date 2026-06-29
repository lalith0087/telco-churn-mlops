"""Turns SHAP feature contributions into a plain-English retention narrative via a local Ollama LLM.

Keeps the GenAI piece decoupled from the ML pipeline: this module only ever sees the
already-computed prediction + SHAP contributions, never the raw model, so the model
backing /predict and /explain can change without touching this file.
"""
import os

import requests

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "30"))

_FEATURE_NAME_HINTS = {
    "num__": "",
    "cat__": "",
}


def _humanize_feature(name: str) -> str:
    for prefix in _FEATURE_NAME_HINTS:
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name.replace("_", " ")


def _build_prompt(prediction: int, probability: float, contributions: list[dict]) -> str:
    verdict = "likely to churn" if prediction == 1 else "likely to stay"
    lines = [
        f"- {_humanize_feature(c['feature'])} ({'pushes toward churn' if c['shap_value'] > 0 else 'pushes toward retention'}, weight {abs(c['shap_value']):.2f})"
        for c in contributions
    ]
    factors = "\n".join(lines)

    return (
        "You are a customer retention analyst. A churn model scored a telecom customer as "
        f"{verdict} (churn probability {probability:.0%}). The top drivers from SHAP, ranked by impact, are:\n\n"
        f"{factors}\n\n"
        "In 3-4 sentences, explain to a non-technical account manager why this customer is at risk "
        "(or not) and recommend one concrete retention action if the churn probability is above 50%. "
        "Do not mention SHAP, models, or weights — speak in plain business language."
    )


def narrate(prediction: int, probability: float, contributions: list[dict]) -> str:
    prompt = _build_prompt(prediction, probability, contributions)

    response = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=OLLAMA_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


if __name__ == "__main__":
    sample_contributions = [
        {"feature": "num__tenure", "shap_value": 1.08},
        {"feature": "cat__Contract_Month-to-month", "shap_value": 0.63},
        {"feature": "cat__OnlineSecurity_No", "shap_value": 0.30},
        {"feature": "num__TotalCharges", "shap_value": 0.27},
        {"feature": "cat__PaymentMethod_Electronic check", "shap_value": 0.22},
    ]
    print(narrate(1, 0.71, sample_contributions))
