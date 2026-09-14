"""IBNR (incurred-but-not-reported) reserve via the loss-development-factor method.

Pure Python / pandas. Deterministic. No ML, no LLM.

Method: for each accident year, the reported-to-date incurred is grossed up to
ultimate using a cumulative reporting pattern (an explicit assumption, below).
    ultimate = reported * LDF,  LDF = 1 / pct_reported(age)
    IBNR     = ultimate - reported
"""

from typing import TypedDict

import pandas as pd

# Cumulative % of ultimate reported by age in months (illustrative long-tail pattern).
# Age = months from the start of the accident year to the valuation date.
REPORTING_PATTERN: dict[int, float] = {12: 0.40, 24: 0.70, 36: 0.88, 48: 0.97, 60: 1.00}


class IbnrResult(TypedDict):
    valuation_year: int
    portfolio_loss: float  # total reported incurred to date
    paid_to_date: float
    case_reserves: float
    ibnr: float
    ultimate_loss: float
    by_accident_year: list[dict[str, float]]
    method: str
    assumptions: dict[str, object]


def pct_reported(age_months: int) -> float:
    """Cumulative reported fraction at a given age, from REPORTING_PATTERN (step function)."""
    for age, pct in sorted(REPORTING_PATTERN.items()):
        if age_months <= age:
            return pct
    return 1.0


def calculate_ibnr(df: pd.DataFrame, valuation_year: int | None = None) -> IbnrResult:
    """Estimate IBNR for the whole book, with an accident-year breakdown.

    Expects columns: accident_year, reported_amount, paid_amount, reserve.
    """
    required = {"accident_year", "reported_amount", "paid_amount", "reserve"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"claims data missing columns: {sorted(missing)}")

    val_year = int(valuation_year or df["accident_year"].max())
    by_ay = (
        df.groupby("accident_year")[["reported_amount", "paid_amount", "reserve"]]
        .sum()
        .sort_index()
    )

    rows: list[dict[str, float]] = []
    for ay, r in by_ay.iterrows():
        age = (val_year - int(ay) + 1) * 12  # valuation at year end
        pct = pct_reported(age)
        ldf = 1.0 / pct
        reported = float(r["reported_amount"])
        ultimate = reported * ldf
        rows.append(
            {
                "accident_year": int(ay),
                "age_months": age,
                "reported": round(reported, 2),
                "pct_reported": pct,
                "ldf": round(ldf, 4),
                "ultimate": round(ultimate, 2),
                "ibnr": round(ultimate - reported, 2),
            }
        )

    portfolio_loss = float(by_ay["reported_amount"].sum())
    ibnr_total = float(sum(r["ibnr"] for r in rows))
    return IbnrResult(
        valuation_year=val_year,
        portfolio_loss=round(portfolio_loss, 2),
        paid_to_date=round(float(by_ay["paid_amount"].sum()), 2),
        case_reserves=round(float(by_ay["reserve"].sum()), 2),
        ibnr=round(ibnr_total, 2),
        ultimate_loss=round(portfolio_loss + ibnr_total, 2),
        by_accident_year=rows,
        method="loss development factor (reported incurred x LDF), valuation at year end",
        assumptions={"reporting_pattern_cum_pct_by_age_months": REPORTING_PATTERN},
    )
