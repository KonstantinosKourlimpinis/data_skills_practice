import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error


def plot_missing(df: pd.DataFrame) -> None:
    missing = df.isnull().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        print("No missing values.")
        return
    missing.plot(kind="bar", figsize=(10, 4), title="Missing value rate")
    plt.ylabel("Fraction missing")
    plt.tight_layout()
    plt.show()


def detect_outliers_iqr(series: pd.Series, factor: float = 1.5) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return (series < q1 - factor * iqr) | (series > q3 + factor * iqr)


def regression_metrics(y_true, y_pred, label: str = "") -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100
    metrics = {"MAE": mae, "RMSE": rmse, "MAPE%": mape}
    prefix = f"{label} — " if label else ""
    for k, v in metrics.items():
        print(f"{prefix}{k}: {v:.3f}")
    return metrics
