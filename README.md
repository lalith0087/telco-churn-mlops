# Telco Customer Churn — End-to-End ML Pipeline with MLflow

Predicts whether a telecom customer will churn, using the [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) dataset (7,043 customers, 26 features).

Pipeline stages: **EDA → feature engineering → multi-model training → MLflow experiment tracking → model registry promotion → inference**.

## Project structure

```
ml-pipeline-mlflow/
├── data/telco_churn.csv      # raw dataset
├── src/
│   ├── eda.py                # exploratory analysis, saves plots to artifacts/
│   ├── features.py           # cleaning + sklearn preprocessing pipeline + train/test split
│   ├── train.py              # trains LogisticRegression, RandomForest, XGBoost; logs to MLflow
│   ├── promote_model.py      # picks best run by metric, registers + promotes to Production
│   └── predict.py            # loads Production model from the registry, scores new data
├── artifacts/                # EDA plots, confusion matrices
├── mlruns/                   # local MLflow tracking store (gitignored)
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

## Results (current run)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.726 | 0.490 | 0.797 | 0.607 | **0.835** |
| Random Forest | 0.749 | 0.519 | 0.773 | 0.621 | 0.834 |
| XGBoost | 0.785 | 0.609 | 0.529 | 0.567 | 0.830 |

Logistic Regression was promoted to Production (highest ROC-AUC). Recall is prioritized via `class_weight="balanced"` since missing a churner is costlier than a false alarm in this business context.
