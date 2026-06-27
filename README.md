# Telco Customer Churn — End-to-End ML Pipeline with MLflow

Predicts whether a telecom customer will churn, using the [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) dataset (7,043 customers, 26 features).

Pipeline stages: **EDA → feature engineering → multi-model training → MLflow experiment tracking → model registry promotion → inference → FastAPI deployment**.

## Project structure

```
ml-pipeline-mlflow/
├── data/telco_churn.csv      # raw dataset
├── src/
│   ├── eda.py                # exploratory analysis, saves plots to artifacts/
│   ├── features.py           # cleaning + sklearn preprocessing pipeline + train/test split
│   ├── train.py              # trains LogisticRegression, RandomForest, XGBoost; logs to MLflow
│   ├── promote_model.py      # picks best run by metric, registers + promotes to Production
│   ├── predict.py            # loads Production model from the registry, scores new data
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
# {"churn_prediction":1,"churn_probability":0.81,"model_version":"2"}
```

The model is loaded once at startup directly from the MLflow registry (`models:/telco-churn-classifier/Production`) — no copying model files into the API code, so re-promoting a new model and restarting the service is the entire deploy step.

### Running via Docker

```bash
docker build -t telco-churn-api .
docker run -p 8000:8000 telco-churn-api
```

The image bundles `mlruns/` (the local MLflow registry) so the container is self-contained — no external MLflow server needed for this local-dev setup. In a real deployment you'd point `MLFLOW_TRACKING_URI` at a shared MLflow tracking server/database instead of a local file store.

## Results (current run)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.726 | 0.490 | 0.797 | 0.607 | **0.835** |
| Random Forest | 0.749 | 0.519 | 0.773 | 0.621 | 0.834 |
| XGBoost | 0.785 | 0.609 | 0.529 | 0.567 | 0.830 |

Logistic Regression was promoted to Production (highest ROC-AUC). Recall is prioritized via `class_weight="balanced"` since missing a churner is costlier than a false alarm in this business context.
