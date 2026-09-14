"""Train the project's two ML models and save them as joblib files.

Model 1 — RandomForestRegressor: predicts claim_amount (severity).
Model 2 — IsolationForest: flags the ~5% most anomalous claims.

These are the ONLY two ML models in the project (see CLAUDE.md §4).

Run:  .venv/Scripts/python -m src.ml.train_models
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from src.ml.features import SEVERITY_TARGET, anomaly_matrix, severity_matrix

SEED = 42
DATA_PATH = Path("data/claims.csv")
MODELS_DIR = Path("models")
SEVERITY_MODEL_PATH = MODELS_DIR / "severity_rf.joblib"
ANOMALY_MODEL_PATH = MODELS_DIR / "anomaly_iforest.joblib"
ANOMALY_CONTAMINATION = 0.05  # flag ~5% of claims


def load_claims(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Generate it with: .venv/Scripts/python -m src.data.make_sample_claims"
        )
    return pd.read_csv(path)


def train_severity_model(df: pd.DataFrame) -> tuple[RandomForestRegressor, float]:
    """Model 1. Returns the fitted regressor and hold-out RMSE."""
    x = severity_matrix(df)
    y = df[SEVERITY_TARGET].astype(float)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=SEED)

    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=5,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    rmse = float(np.sqrt(mean_squared_error(y_test, model.predict(x_test))))
    return model, rmse


def train_anomaly_model(df: pd.DataFrame) -> tuple[IsolationForest, int]:
    """Model 2. Returns the fitted detector and the number of claims it flags."""
    x = anomaly_matrix(df)
    model = IsolationForest(
        n_estimators=300,
        contamination=ANOMALY_CONTAMINATION,
        random_state=SEED,
    )
    model.fit(x)
    flags = model.predict(x)  # -1 = anomaly, 1 = normal
    return model, int((flags == -1).sum())


def main() -> None:
    df = load_claims()
    print(f"Loaded {len(df):,} claims from {DATA_PATH}")

    severity_model, rmse = train_severity_model(df)
    print(f"[Model 1] RandomForestRegressor  hold-out RMSE = ${rmse:,.0f}  "
          f"(mean claim ${df[SEVERITY_TARGET].mean():,.0f})")

    anomaly_model, n_anomalies = train_anomaly_model(df)
    print(f"[Model 2] IsolationForest        anomalies flagged = {n_anomalies} "
          f"of {len(df):,} ({n_anomalies / len(df):.1%})")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(severity_model, SEVERITY_MODEL_PATH)
    joblib.dump(anomaly_model, ANOMALY_MODEL_PATH)
    print(f"Saved {SEVERITY_MODEL_PATH} and {ANOMALY_MODEL_PATH}")


if __name__ == "__main__":
    main()
