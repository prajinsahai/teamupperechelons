"""Loss development, adverse development and large-loss early indicators (Claims Analyst)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def development_pattern(df: pd.DataFrame) -> dict[str, Any]:
    """Paid / reported ratio by development bucket (12-month), and by accident year."""
    d = df.copy()
    d["dev_bucket"] = (np.ceil(d["development_month"] / 12).clip(1, 6) * 12).astype(int)
    by_bucket = d.groupby("dev_bucket").agg(paid=("paid_amount", "sum"), reported=("reported_amount", "sum"), n=("claim_id", "count"))
    by_bucket["paid_ratio"] = (by_bucket["paid"] / by_bucket["reported"]).round(3)
    by_ay = d.groupby("accident_year").agg(paid=("paid_amount", "sum"), reported=("reported_amount", "sum"), reserve=("reserve", "sum"), n=("claim_id", "count"))
    by_ay["reserve_ratio"] = (by_ay["reserve"] / by_ay["reported"]).round(3)
    return {
        "paid_to_reported_by_dev_month": {int(k): float(v) for k, v in by_bucket["paid_ratio"].items()},
        "claims_by_dev_month": {int(k): int(v) for k, v in by_bucket["n"].items()},
        "by_accident_year": [
            {"accident_year": int(y), "claims": int(r["n"]), "reported": round(float(r["reported"]), 2),
             "paid": round(float(r["paid"]), 2), "reserve": round(float(r["reserve"]), 2), "reserve_ratio": float(r["reserve_ratio"])}
            for y, r in by_ay.iterrows()
        ],
        "method": "paid/reported by 12-month development bucket; reserve/reported by accident year",
    }


def adverse_development(df: pd.DataFrame, z_threshold: float = 1.0) -> dict[str, Any]:
    """Accident years whose reserve ratio is unusually high for their age -> adverse development signal."""
    d = df.copy()
    latest = int(d["accident_year"].max())
    by_ay = d.groupby("accident_year").agg(reported=("reported_amount", "sum"), reserve=("reserve", "sum"), n=("claim_id", "count"))
    by_ay["age_months"] = (latest - by_ay.index + 1) * 12
    by_ay["reserve_ratio"] = by_ay["reserve"] / by_ay["reported"]
    # expected reserve ratio decays with age: fit log-linear on the years we have
    x = by_ay["age_months"].to_numpy(dtype=float)
    y = by_ay["reserve_ratio"].to_numpy(dtype=float)
    if len(x) >= 3 and np.isfinite(y).all():
        coef = np.polyfit(x, y, 1)
        expected = np.polyval(coef, x)
        resid = y - expected
        sd = resid.std(ddof=1) if len(resid) > 2 else 1e-9
        z = resid / sd if sd > 0 else np.zeros_like(resid)
    else:
        expected, z = y, np.zeros_like(y)
    flagged = []
    rows = []
    for (ay, r), e, zz in zip(by_ay.iterrows(), expected, z):
        row = {"accident_year": int(ay), "age_months": int(r["age_months"]), "claims": int(r["n"]),
               "reserve_ratio": round(float(r["reserve_ratio"]), 3), "expected_ratio_for_age": round(float(e), 3), "z_score": round(float(zz), 2)}
        rows.append(row)
        if zz > z_threshold:
            flagged.append(row)
    return {
        "flagged_accident_years": flagged,
        "all_years": rows,
        "adverse_development_detected": bool(flagged),
        "method": "reserve/reported ratio vs age-of-year trend; z > 1.0 flagged",
    }


def large_loss_indicators(df: pd.DataFrame, early_months: int = 12, top_n: int = 10) -> dict[str, Any]:
    """Early-development claims with unusually high reserves or reported amounts -> large-loss risk."""
    d = df.copy()
    p90 = float(d["claim_amount"].quantile(0.90))
    early = d[d["development_month"] <= early_months]
    big_early = early[early["reported_amount"] >= p90].sort_values("reported_amount", ascending=False)
    open_heavy = d[(d["reserve"] > 0) & (d["reserve"] >= d["reported_amount"] * 0.8)]
    return {
        "severity_p90": round(p90, 2),
        "early_large_claims_count": int(len(big_early)),
        "early_large_claims": [
            {"claim_id": str(r["claim_id"]), "accident_year": int(r["accident_year"]), "line_of_business": str(r["line_of_business"]),
             "reported_amount": round(float(r["reported_amount"]), 2), "development_month": int(r["development_month"])}
            for _, r in big_early.head(top_n).iterrows()
        ],
        "mostly_unpaid_claims_count": int(len(open_heavy)),
        "mostly_unpaid_total_reserve": round(float(open_heavy["reserve"].sum()), 2),
        "method": f"claims <= {early_months} dev months with reported >= p90; claims with reserve >= 80% of reported",
    }


def claim_pattern_shift(df: pd.DataFrame) -> dict[str, Any]:
    """Frequency and severity by line of business, latest year vs prior average."""
    d = df.copy()
    latest = int(d["accident_year"].max())
    rows = []
    for lob, g in d.groupby("line_of_business"):
        cur = g[g["accident_year"] == latest]
        prior = g[g["accident_year"] < latest]
        prior_years = max(prior["accident_year"].nunique(), 1)
        rows.append({
            "line_of_business": str(lob),
            "latest_year_claims": int(len(cur)),
            "prior_avg_claims_per_year": round(len(prior) / prior_years, 1),
            "latest_mean_severity": round(float(cur["claim_amount"].mean()), 2) if len(cur) else None,
            "prior_mean_severity": round(float(prior["claim_amount"].mean()), 2) if len(prior) else None,
            "severity_change_pct": round((float(cur["claim_amount"].mean()) / float(prior["claim_amount"].mean()) - 1) * 100, 1)
            if len(cur) and len(prior) and prior["claim_amount"].mean() else None,
        })
    rows.sort(key=lambda r: -(r["severity_change_pct"] or 0))
    return {"latest_year": latest, "by_line": rows, "method": "latest accident year vs prior-year averages, per line"}
