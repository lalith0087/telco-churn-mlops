# Telco Customer Churn — End-to-End ML Pipeline with MLflow

Predicts whether a telecom customer will churn, using the [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) dataset (7,043 customers, 26 features).

Pipeline stages: **EDA → feature engineering → multi-model training → hyperparameter tuning → MLflow experiment tracking → model registry promotion → inference → FastAPI deployment → SHAP explainability**.

## Project structure

```
ml-pipeline-mlflow/
├── data/telco_churn.csv      # raw dataset
├── src/
│   ├── eda.py                # exploratory analysis, saves plots to artifacts/
│   ├── features.py           # cleaning + sklearn preprocessing pipeline + train/test split
│   ├── train.py              # trains LogisticRegression, RandomForest, XGBoost; logs to MLflow
│   ├── tune.py                # Optuna hyperparameter search per model, logs every trial to MLflow
│   ├── promote_model.py      # picks best run by metric, registers + promotes to Production
│   ├── predict.py            # loads Production model from the registry, scores new data
│   ├── explain.py            # SHAP explainer wrapper, model-agnostic via shap.Explainer
│   ├── schemas.py            # pydantic request/response models for the API
│   └── app.py                # FastAPI service serving the Production model
├── artifacts/                # EDA plots, confusion matrices
├── mlruns/                   # local MLflow tracking store
├── Dockerfile                # containerizes the FastAPI service
└── requirements.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the pipeline

```bash
cd src

# 1. Exploratory analysis (writes plots to ../artifacts)
python eda.py

# 2. Train models — logs params/metrics/confusion matrix/model artifact to MLflow
python train.py

# 3. View experiments in the MLflow UI
mlflow ui --backend-store-uri ../mlruns
# open http://127.0.0.1:5000

# 4. Promote the best run (by roc_auc) to the Production stage in the Model Registry
python promote_model.py --metric roc_auc

# 5. Score new data with the registered Production model
python predict.py ../data/telco_churn.csv --output ../artifacts/predictions.csv
```

## What gets tracked per run

For each of the three models, MLflow logs:
- **Params**: model type + full hyperparameter set
- **Metrics**: accuracy, precision, recall, F1, ROC-AUC
- **Artifacts**: confusion matrix plot, the full fitted sklearn `Pipeline` (preprocessing + model) as an MLflow model

This means each run is fully reproducible and deployable straight from the registry — no retraining needed to serve predictions.

## Hyperparameter tuning with Optuna

`train.py` trains each model once with hand-picked hyperparameters. `tune.py` goes further: it runs an Optuna study per model (25 trials each, 5-fold cross-validated ROC-AUC as the objective), logging **every trial as a nested MLflow run** under one `optuna_tuning_parent` run. After each study finishes, the best-found hyperparameters are refit on the full training set and logged as a normal top-level run — so it's directly comparable to (and competes with) the runs from `train.py` when `promote_model.py` picks the best one.

```bash
cd src
python tune.py
```

This turns the experiment from 3 static runs into 75+ trial runs plus 3 tuned "winner" runs, all visible in the MLflow UI's table/chart/parallel-coordinates views — useful for seeing which hyperparameters actually move the needle (e.g. XGBoost's `learning_rate` and `max_depth` mattered far more than `n_estimators` here).

In this run, tuned XGBoost (`max_depth=3, learning_rate=0.026, subsample=0.63`) beat every model from `train.py`, lifting test ROC-AUC from 0.835 → **0.841**, and was promoted to Production as model version 3.

## Model registry

`promote_model.py` queries all runs in the `telco-churn-prediction` experiment, picks the highest-`roc_auc` run, registers it under `telco-churn-classifier`, and transitions it to the `Production` stage (archiving any prior production version). `predict.py` always loads `models:/telco-churn-classifier/Production`, so swapping in a better model later is just: train → promote → done, with no code changes downstream.

## Serving predictions via FastAPI

Once a model has been promoted to `Production` (step 4 above), serve it over HTTP:

```bash
cd src
uvicorn app:app --host 0.0.0.0 --port 8000
```

- `GET /health` — returns the registered model name + version currently loaded
- `POST /predict` — accepts a single customer's raw features as JSON, returns `churn_prediction` (0/1) and `churn_probability`
- `POST /explain` — same input, but also returns the top 5 SHAP feature contributions driving that specific prediction
- Interactive docs: `http://127.0.0.1:8000/docs`

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
  "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
  "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
  "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
  "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85
}'
# {"churn_prediction":1,"churn_probability":0.71,"model_version":"5"}
```

The model is loaded once at startup directly from the MLflow registry (`models:/telco-churn-classifier/Production`) — no copying model files into the API code, so re-promoting a new model and restarting the service is the entire deploy step.

## Explainability with SHAP

Knowing a customer is likely to churn isn't enough on its own — a retention team needs to know *why*. `POST /explain` runs the same input through `shap.Explainer` against the live Production pipeline and returns the top contributing features for that specific prediction:

```bash
curl -X POST http://127.0.0.1:8000/explain -H "Content-Type: application/json" -d '{
  "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
  "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
  "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
  "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
  "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85
}'
```

```json
{
  "churn_prediction": 1,
  "churn_probability": 0.71,
  "model_version": "5",
  "top_contributing_features": [
    {"feature": "num__tenure", "shap_value": 1.08},
    {"feature": "cat__Contract_Month-to-month", "shap_value": 0.63},
    {"feature": "cat__OnlineSecurity_No", "shap_value": 0.30},
    {"feature": "num__TotalCharges", "shap_value": 0.27},
    {"feature": "cat__PaymentMethod_Electronic check", "shap_value": 0.22}
  ]
}
```

Positive SHAP values push the prediction toward churn; negative values push toward retention. `explain.py` uses `shap.Explainer` (not `TreeExplainer` directly), so it auto-selects the right algorithm whether the Production model is a tree ensemble (XGBoost/RandomForest) or a linear model (LogisticRegression) — no code change needed when a different model gets promoted.

### Running via Docker

```bash
docker build -t telco-churn-api .
docker run -p 8000:8000 telco-churn-api
```

The image bundles `mlruns/` (the local MLflow registry) so the container is self-contained — no external MLflow server needed for this local-dev setup. In a real deployment you'd point `MLFLOW_TRACKING_URI` at a shared MLflow tracking server/database instead of a local file store.

## Results

### Baseline (`train.py`)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.726 | 0.490 | 0.797 | 0.607 | 0.835 |
| Random Forest | 0.749 | 0.519 | 0.773 | 0.621 | 0.834 |
| XGBoost | 0.785 | 0.609 | 0.529 | 0.567 | 0.830 |

### After tuning (`tune.py`)

| Model | Best CV ROC-AUC | Test ROC-AUC |
|---|---|---|
| Logistic Regression (tuned) | 0.846 | — |
| Random Forest (tuned) | 0.848 | — |
| XGBoost (tuned) | 0.850 | **0.841** |

Tuned XGBoost was promoted to Production (model version 3), improving test ROC-AUC over the best baseline. Recall is prioritized via `class_weight="balanced"` on the linear/tree models since missing a churner is costlier than a false alarm in this business context.
