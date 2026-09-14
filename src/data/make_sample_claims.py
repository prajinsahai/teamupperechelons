"""Generate an illustrative claims dataset for a hypothetical mid-cap manufacturer.

All figures are synthetic. Seeded so the file is reproducible (same input -> same output).

Run:  .venv/Scripts/python -m src.data.make_sample_claims
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_CLAIMS = 1_200
YEARS = list(range(2018, 2025))  # 7 claim years, as in the deck
OUT_PATH = Path("data/claims.csv")

RISK_CLASSES = ["A", "B", "C", "D"]  # A = lowest risk
LINES = ["property", "general_liability", "auto", "cyber", "business_interruption"]


def make_claims(n: int = N_CLAIMS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    risk_class = rng.choice(RISK_CLASSES, size=n, p=[0.35, 0.35, 0.2, 0.1])
    risk_idx = np.array([RISK_CLASSES.index(r) for r in risk_class])
    exposure = rng.lognormal(mean=np.log(2_000_000), sigma=0.6, size=n).round(-3)
    previous_claims = rng.poisson(lam=1.2 + 0.6 * risk_idx, size=n)
    deductible = rng.choice([25_000, 50_000, 100_000, 250_000], size=n, p=[0.3, 0.4, 0.2, 0.1])
    coverage_limit = rng.choice([1e6, 2e6, 5e6, 10e6], size=n, p=[0.25, 0.35, 0.3, 0.1])

    # Severity: lognormal driven by risk class, exposure and prior claim history.
    mu = (
        np.log(45_000)
        + 0.35 * risk_idx
        + 0.25 * (np.log(exposure) - np.log(2_000_000))
        + 0.08 * previous_claims
    )
    claim_amount = rng.lognormal(mean=mu, sigma=0.9, size=n)
    claim_amount = np.minimum(claim_amount, coverage_limit).round(0)

    accident_year = rng.choice(YEARS, size=n)
    development_month = rng.integers(1, 61, size=n)  # months since accident
    line = rng.choice(LINES, size=n, p=[0.3, 0.25, 0.2, 0.15, 0.1])

    # Reported / paid / reserve follow a plausible development pattern.
    dev_frac = np.clip(development_month / 48.0, 0.05, 1.0)
    reported_amount = (claim_amount * rng.uniform(0.9, 1.15, size=n)).round(0)
    paid_amount = (claim_amount * dev_frac * rng.uniform(0.85, 1.0, size=n)).round(0)
    reserve = np.maximum(reported_amount - paid_amount, 0).round(0)

    df = pd.DataFrame(
        {
            "claim_id": [f"CLM-{i:05d}" for i in range(1, n + 1)],
            "accident_year": accident_year,
            "line_of_business": line,
            "risk_class": risk_class,
            "exposure": exposure,
            "previous_claims": previous_claims,
            "deductible": deductible,
            "coverage_limit": coverage_limit,
            "development_month": development_month,
            "claim_amount": claim_amount,
            "reported_amount": reported_amount,
            "paid_amount": paid_amount,
            "reserve": reserve,
        }
    )

    # Inject a handful of obvious anomalies for the Isolation Forest to find.
    anomalies = rng.choice(n, size=25, replace=False)
    df.loc[anomalies[:10], "paid_amount"] = df.loc[anomalies[:10], "reported_amount"] * 3  # overpaid
    df.loc[anomalies[10:18], "reserve"] = df.loc[anomalies[10:18], "claim_amount"] * 8  # bloated reserve
    df.loc[anomalies[18:], "claim_amount"] = df.loc[anomalies[18:], "coverage_limit"] * 4  # above limit

    return df


def main() -> None:
    df = make_claims()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df):,} illustrative claims to {OUT_PATH}")
    print(df.describe(include="all").T[["count", "mean", "min", "max"]].head(13))


if __name__ == "__main__":
    main()
