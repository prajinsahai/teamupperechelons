"""Feature definitions shared by train_models.py and predict.py.

Kept in one place so training and inference never drift apart.
"""

import pandas as pd

# Model 1 — claim severity (Random Forest Regressor)
SEVERITY_TARGET = "claim_amount"
SEVERITY_FEATURES = ["exposure", "risk_class", "previous_claims", "deductible", "coverage_limit"]

# Model 2 — anomaly detection (Isolation Forest)
ANOMALY_FEATURES = ["claim_amount", "reported_amount", "paid_amount", "reserve", "development_month"]

# risk_class is ordinal: A (lowest risk) -> D (highest risk)
RISK_CLASS_MAP = {"A": 0, "B": 1, "C": 2, "D": 3}


def severity_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return the numeric feature matrix for the severity model."""
    x = df[SEVERITY_FEATURES].copy()
    x["risk_class"] = x["risk_class"].map(RISK_CLASS_MAP)
    if x["risk_class"].isna().any():
        bad = sorted(df.loc[x["risk_class"].isna(), "risk_class"].unique())
        raise ValueError(f"Unknown risk_class values: {bad}; expected {list(RISK_CLASS_MAP)}")
    return x.astype(float)


def anomaly_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return the numeric feature matrix for the anomaly model."""
    return df[ANOMALY_FEATURES].astype(float)
