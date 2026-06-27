"""Hyperparameter tuning with Optuna. Runs a study per model type, logs every trial to
MLflow as a nested run under a parent 'tuning' run, then refits the best params on the
full train set and logs that as a top-level run (so promote_model.py can pick it up
alongside the runs from train.py)."""
import os

import mlflow
import mlflow.sklearn
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from features import build_preprocessor, get_feature_columns, load_clean, split_data

MLFLOW_TRACKING_URI = "file:" + os.path.join(os.path.dirname(__file__), "..", "mlruns")
EXPERIMENT_NAME = "telco-churn-prediction"
N_TRIALS = 25
CV_FOLDS = 5


def make_logreg(trial: optuna.Trial) -> LogisticRegression:
    return LogisticRegression(
        C=trial.suggest_float("C", 1e-3, 10.0, log=True),
        max_iter=1000,
        class_weight="balanced",
        solver=trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
    )


def make_random_forest(trial: optuna.Trial) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
        max_depth=trial.suggest_int("max_depth", 3, 15),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        class_weight="balanced",
        random_state=42,
    )


def make_xgboost(trial: optuna.Trial) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
        max_depth=trial.suggest_int("max_depth", 2, 8),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        eval_metric="logloss",
        random_state=42,
    )


MODEL_BUILDERS = {
    "logistic_regression": make_logreg,
    "random_forest": make_random_forest,
    "xgboost": make_xgboost,
}


def tune_model(model_name: str, builder, X_train, y_train, preprocessor) -> dict:
    def objective(trial: optuna.Trial) -> float:
        model = builder(trial)
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

        with mlflow.start_run(run_name=f"{model_name}_trial_{trial.number}", nested=True):
            scores = cross_val_score(pipeline, X_train, y_train, cv=CV_FOLDS, scoring="roc_auc")
            mean_score = scores.mean()

            mlflow.log_param("model_type", model_name)
            mlflow.log_params(trial.params)
            mlflow.log_metric("cv_roc_auc_mean", mean_score)
            mlflow.log_metric("cv_roc_auc_std", scores.std())

        return mean_score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    print(f"[{model_name}] best cv_roc_auc={study.best_value:.4f}, params={study.best_params}")
    return study.best_params


def refit_and_log_best(model_name: str, builder, best_params: dict, X_train, X_test, y_train, y_test, preprocessor) -> None:
    class _FrozenTrial:
        def __init__(self, params):
            self.params = params

        def suggest_float(self, name, *_args, **_kwargs):
            return self.params[name]

        def suggest_int(self, name, *_args, **_kwargs):
            return self.params[name]

        def suggest_categorical(self, name, *_args, **_kwargs):
            return self.params[name]

    model = builder(_FrozenTrial(best_params))
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    with mlflow.start_run(run_name=f"{model_name}_tuned", nested=True):
        pipeline.fit(X_train, y_train)
        proba = pipeline.predict_proba(X_test)[:, 1]
        preds = pipeline.predict(X_test)

        mlflow.log_param("model_type", model_name)
        mlflow.log_params(best_params)
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, proba))
        mlflow.log_metric("accuracy", (preds == y_test).mean())

        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        print(f"Logged tuned {model_name} as a top-level run.")


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_clean()
    numeric, categorical = get_feature_columns(df)
    preprocessor = build_preprocessor(numeric, categorical)
    X_train, X_test, y_train, y_test = split_data(df)

    with mlflow.start_run(run_name="optuna_tuning_parent"):
        for model_name, builder in MODEL_BUILDERS.items():
            best_params = tune_model(model_name, builder, X_train, y_train, preprocessor)
            refit_and_log_best(model_name, builder, best_params, X_train, X_test, y_train, y_test, preprocessor)


if __name__ == "__main__":
    main()
