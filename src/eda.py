"""Quick EDA on the Telco churn dataset. Saves summary stats and plots to artifacts/."""
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "telco_churn.csv")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")


def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def run_eda(df: pd.DataFrame, out_dir: str = ARTIFACTS_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)

    print(df.shape)
    print(df.dtypes)
    print(df.isna().sum()[lambda s: s > 0])
    print(df["Churn"].value_counts(normalize=True))

    plt.figure(figsize=(4, 4))
    df["Churn"].value_counts().plot(kind="bar", title="Churn distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "churn_distribution.png"))
    plt.close()

    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    plt.figure(figsize=(10, 6))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(1, 3, i)
        sns.histplot(data=df, x=col, hue="Churn", kde=True, element="step")
        plt.title(col)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "numeric_distributions.png"))
    plt.close()

    plt.figure(figsize=(6, 5))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
    plt.title("Numeric feature correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "correlation_heatmap.png"))
    plt.close()


if __name__ == "__main__":
    run_eda(load_raw())
