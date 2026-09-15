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


def loss_triangle(df: pd.DataFrame, bucket_months: int = 12) -> dict[str, Any]:
    """Incurred loss development triangle (accident year x development bucket, cumulative),
    volume-weighted age-to-age factors, and a chain-ladder ultimate per accident year.

    Claims data is a snapshot (one row per claim as of today), so the history is
    reconstructed: a claim enters the triangle once reported (report_lag_days) and its
    incurred grows along the reporting pattern from src/actuarial/ibnr.py up to its
    current value. That is a modelling assumption and is stated in `method`.
    """
    from src.actuarial.ibnr import pct_reported

    d = df.copy()
    latest = int(d["accident_year"].max())
    years = sorted(int(y) for y in d["accident_year"].unique())
    max_age = (latest - years[0] + 1) * 12
    buckets = list(range(bucket_months, min(max_age, 120) + 1, bucket_months))
    lag_m = (d["report_lag_days"] / 30.0) if "report_lag_days" in d else pd.Series(0.0, index=d.index)
    tri = pd.DataFrame(index=years, columns=buckets, dtype=float)
    for y in years:
        g = d[d["accident_year"] == y]
        age = (latest - y + 1) * 12
        cur = pct_reported(age)
        for bkt in buckets:
            if bkt > age:
                tri.loc[y, bkt] = np.nan
            else:
                entered = lag_m.loc[g.index] <= bkt
                tri.loc[y, bkt] = float((g.loc[entered, "reported_amount"] * (pct_reported(bkt) / cur)).sum())
    factors: dict[str, float] = {}
    for a, b2 in zip(buckets[:-1], buckets[1:]):
        both = tri[[a, b2]].dropna()
        both = both[both[a] > 0]
        factors[f"{a}-{b2}"] = round(float(both[b2].sum() / both[a].sum()), 4) if len(both) else 1.0
    cdf: dict[int, float] = {buckets[-1]: 1.0}
    acc = 1.0
    for a, b2 in reversed(list(zip(buckets[:-1], buckets[1:]))):
        acc *= factors[f"{a}-{b2}"]
        cdf[a] = round(acc, 4)
    ultimates = []
    for y in years:
        row = tri.loc[y].dropna()
        if row.empty:
            continue
        last_b = int(row.index[-1]); latest_val = float(row.iloc[-1]); ult = latest_val * cdf.get(last_b, 1.0)
        ultimates.append({"accident_year": y, "latest_bucket": last_b, "reported_to_date": round(latest_val, 2),
                          "cdf_to_ultimate": cdf.get(last_b, 1.0), "chain_ladder_ultimate": round(ult, 2), "ibnr": round(ult - latest_val, 2)})
    return {
        "bucket_months": bucket_months,
        "buckets": buckets,
        "triangle": {y: {int(k): (None if pd.isna(v) else round(float(v), 2)) for k, v in tri.loc[y].items()} for y in years},
        "age_to_age_factors": factors,
        "cdf_to_ultimate": {int(k): v for k, v in cdf.items()},
        "by_accident_year": ultimates,
        "total_chain_ladder_ibnr": round(float(sum(u["ibnr"] for u in ultimates)), 2),
        "method": "triangle reconstructed from claim snapshots using report lag and the reporting pattern assumption; volume-weighted age-to-age; chain-ladder ultimate",
    }


def claims_leakage(df: pd.DataFrame, cpi: float = 0.03) -> dict[str, Any]:
    """Leakage (paid above model-predicted severity on closed claims), litigation rate and
    severity multiple, social inflation (severity trend above an assumed CPI), reporting lag."""
    from src.ml.predict import predict_severity

    d = df.copy()
    d["predicted_severity"] = predict_severity(d)
    closed = d[d["status"] == "closed"] if "status" in d else d
    over = (d["paid_amount"] - d["reported_amount"]).clip(lower=0)  # paid more than was ever reported
    leak_total = float(over.sum())
    paid_total = float(d["paid_amount"].sum())
    top = d.assign(overpaid=over).nlargest(8, "overpaid")
    rf_flag = d["paid_amount"] > 2.0 * d["predicted_severity"]
    lit = d["litigated"].astype(bool) if "litigated" in d else pd.Series(False, index=d.index)
    lit_sev = float(d.loc[lit, "claim_amount"].mean()) if lit.any() else None
    nonlit_sev = float(d.loc[~lit, "claim_amount"].mean()) if (~lit).any() else None
    by_year = d.groupby("accident_year")["claim_amount"].mean().sort_index()
    sev_trend = float(np.exp(np.polyfit(np.arange(len(by_year)), np.log(by_year.to_numpy()), 1)[0]) - 1) if len(by_year) >= 3 else 0.0
    lag = d["report_lag_days"] if "report_lag_days" in d else None
    return {
        "closed_claims": int(len(closed)),
        "leakage_total": round(leak_total, 2),
        "leakage_rate_pct": round(leak_total / paid_total * 100, 2) if paid_total else 0.0,
        "overpaid_claim_count": int((over > 0).sum()),
        "rf_outlier_paid_count": int(rf_flag.sum()),
        "top_overpaid_claims": [{"claim_id": str(r["claim_id"]), "line_of_business": str(r["line_of_business"]), "paid_amount": round(float(r["paid_amount"]), 2),
                                 "predicted_severity": round(float(r["predicted_severity"]), 2), "overpaid": round(float(r["overpaid"]), 2)} for _, r in top.iterrows()],
        "litigation_rate_pct": round(float(lit.mean() * 100), 1),
        "litigated_mean_severity": round(lit_sev, 2) if lit_sev else None,
        "non_litigated_mean_severity": round(nonlit_sev, 2) if nonlit_sev else None,
        "litigation_severity_multiple": round(lit_sev / nonlit_sev, 2) if lit_sev and nonlit_sev else None,
        "severity_trend_pct_per_year": round(sev_trend * 100, 1),
        "assumed_cpi_pct": cpi * 100,
        "social_inflation_pct_per_year": round((sev_trend - cpi) * 100, 1),
        "mean_report_lag_days": round(float(lag.mean()), 1) if lag is not None else None,
        "late_reported_claims_over_90d": int((lag > 90).sum()) if lag is not None else None,
        "method": "leakage = paid in excess of reported incurred (overpayment); RF outlier = paid > 2x RandomForest-predicted severity; social inflation = log-linear severity trend minus CPI",
    }
