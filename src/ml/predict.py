"""Inference for the project's two ML models + the claims-intelligence summary.

Loads models/severity_rf.joblib and models/anomaly_iforest.joblib (train with
`python -m src.ml.train_models`). No training happens here.
"""

from functools import lru_cache
from pathlib import Path
from typing import TypedDict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor

from src.ml.features import anomaly_matrix, severity_matrix

MODELS_DIR = Path("models")
SEVERITY_MODEL_PATH = MODELS_DIR / "severity_rf.joblib"
ANOMALY_MODEL_PATH = MODELS_DIR / "anomaly_iforest.joblib"


@lru_cache(maxsize=1)
def load_models() -> tuple[RandomForestRegressor, IsolationForest]:
    for p in (SEVERITY_MODEL_PATH, ANOMALY_MODEL_PATH):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Train with: .venv/Scripts/python -m src.ml.train_models")
    return joblib.load(SEVERITY_MODEL_PATH), joblib.load(ANOMALY_MODEL_PATH)


def predict_severity(df: pd.DataFrame) -> np.ndarray:
    """Model 1: predicted claim_amount per row."""
    model, _ = load_models()
    return model.predict(severity_matrix(df))


ANOMALY_RATE = 0.05


def flag_anomalies(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Model 2: (is_anomaly bool array, anomaly score — lower = more anomalous).

    The cut is the bottom ANOMALY_RATE of scores *within the loaded book*, not the
    model's fixed training threshold: books with very different scales would otherwise
    be flagged wholesale (a healthcare book scored against a fleet book's threshold).
    """
    _, model = load_models()
    x = anomaly_matrix(df)
    scores = model.score_samples(x)
    cut = np.quantile(scores, ANOMALY_RATE) if len(scores) > 20 else -np.inf
    return scores <= cut, scores


class ClaimsIntelligence(TypedDict):
    claim_count: int
    latest_year: int
    latest_year_mean_severity: float
    prior_years_mean_severity: float
    severity_change_pct: float  # latest year vs prior 3-year average, in %
    predicted_mean_severity: float  # RF model, whole book
    historical_mean_severity: float
    anomaly_count: int
    anomaly_rate_pct: float
    anomalous_claim_ids: list[str]  # most anomalous first
    anomalous_amount_total: float
    severity_by_year: dict[int, float]


def get_claims_intelligence(df: pd.DataFrame, trend_window_years: int = 3) -> ClaimsIntelligence:
    """Severity trend + anomaly summary for the dashboard and the LLM tool."""
    df = df.copy()

    # --- severity trend (latest accident year vs the prior N-year average)
    by_year = df.groupby("accident_year")["claim_amount"].mean().sort_index()
    latest_year = int(by_year.index.max())
    prior = by_year.loc[by_year.index < latest_year].tail(trend_window_years)
    latest_mean = float(by_year.loc[latest_year])
    prior_mean = float(prior.mean()) if len(prior) else latest_mean
    change_pct = (latest_mean / prior_mean - 1.0) * 100.0 if prior_mean else 0.0

    # --- Model 1: what the RF thinks this book "should" cost
    predicted = predict_severity(df)

    # --- Model 2: anomalies
    is_anom, score = flag_anomalies(df)
    df["anomaly_score"] = score
    anomalies = df.loc[is_anom].sort_values("anomaly_score")

    return ClaimsIntelligence(
        claim_count=int(len(df)),
        latest_year=latest_year,
        latest_year_mean_severity=round(latest_mean, 2),
        prior_years_mean_severity=round(prior_mean, 2),
        severity_change_pct=round(change_pct, 1),
        predicted_mean_severity=round(float(predicted.mean()), 2),
        historical_mean_severity=round(float(df["claim_amount"].mean()), 2),
        anomaly_count=int(is_anom.sum()),
        anomaly_rate_pct=round(float(is_anom.mean() * 100.0), 1),
        anomalous_claim_ids=anomalies["claim_id"].astype(str).tolist(),
        anomalous_amount_total=round(float(anomalies["claim_amount"].sum()), 2),
        severity_by_year={int(y): round(float(v), 2) for y, v in by_year.items()},
    )


def anomaly_table(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Top-N most anomalous claims, for display."""
    is_anom, score = flag_anomalies(df)
    out = df.loc[is_anom].copy()
    out["anomaly_score"] = score[is_anom]
    cols = ["claim_id", "accident_year", "line_of_business", "claim_amount",
            "reported_amount", "paid_amount", "reserve", "development_month", "anomaly_score"]
    return out.sort_values("anomaly_score")[cols].head(top_n).reset_index(drop=True)
