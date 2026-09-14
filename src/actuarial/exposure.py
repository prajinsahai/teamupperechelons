"""Exposure movement, concentration and emerging-risk signals (Risk Manager)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def exposure_movement(df: pd.DataFrame) -> dict[str, Any]:
    """Exposure and loss by year and by line; year-over-year change."""
    by_year = df.groupby("accident_year").agg(exposure=("exposure", "sum"), loss=("claim_amount", "sum"), claims=("claim_id", "count"))
    by_year["loss_ratio_to_exposure_pct"] = (by_year["loss"] / by_year["exposure"] * 100).round(2)
    by_year["exposure_yoy_pct"] = (by_year["exposure"].pct_change() * 100).round(1)
    by_line = df.groupby("line_of_business").agg(exposure=("exposure", "sum"), loss=("claim_amount", "sum"), claims=("claim_id", "count"))
    by_line["share_of_loss_pct"] = (by_line["loss"] / by_line["loss"].sum() * 100).round(1)
    return {
        "by_year": [{"accident_year": int(y), **{k: (None if pd.isna(v) else round(float(v), 2)) for k, v in r.items()}} for y, r in by_year.iterrows()],
        "by_line": [{"line_of_business": str(l), **{k: round(float(v), 2) for k, v in r.items()}} for l, r in by_line.iterrows()],
        "method": "sums by accident year and line; loss ratio to exposure",
    }


def concentration(df: pd.DataFrame) -> dict[str, Any]:
    """Herfindahl index of loss by line and by risk class; top contributors."""
    def hhi(series: pd.Series) -> float:
        s = series / series.sum()
        return float((s**2).sum())

    loss_by_line = df.groupby("line_of_business")["claim_amount"].sum().sort_values(ascending=False)
    loss_by_class = df.groupby("risk_class")["claim_amount"].sum().sort_values(ascending=False)
    top_claims = df.nlargest(5, "claim_amount")
    return {
        "hhi_by_line": round(hhi(loss_by_line), 3),
        "hhi_by_risk_class": round(hhi(loss_by_class), 3),
        "top_line": {"line_of_business": str(loss_by_line.index[0]), "share_pct": round(float(loss_by_line.iloc[0] / loss_by_line.sum() * 100), 1)},
        "top5_claims_share_pct": round(float(top_claims["claim_amount"].sum() / df["claim_amount"].sum() * 100), 1),
        "top5_claims": [{"claim_id": str(r["claim_id"]), "line_of_business": str(r["line_of_business"]), "claim_amount": round(float(r["claim_amount"]), 2)} for _, r in top_claims.iterrows()],
        "concentration_level": "high" if hhi(loss_by_line) > 0.4 else "moderate" if hhi(loss_by_line) > 0.25 else "diversified",
        "method": "HHI (sum of squared shares); > 0.4 high, > 0.25 moderate",
    }


def emerging_signals(df: pd.DataFrame, window: int = 2) -> dict[str, Any]:
    """Lines where recent frequency or severity is rising vs the earlier period."""
    latest = int(df["accident_year"].max())
    recent = df[df["accident_year"] > latest - window]
    earlier = df[df["accident_year"] <= latest - window]
    signals = []
    for lob in sorted(df["line_of_business"].unique(), key=str):
        r, e = recent[recent["line_of_business"] == lob], earlier[earlier["line_of_business"] == lob]
        ry = max(recent["accident_year"].nunique(), 1)
        ey = max(earlier["accident_year"].nunique(), 1)
        freq_r, freq_e = len(r) / ry, len(e) / ey
        sev_r = float(r["claim_amount"].mean()) if len(r) else 0.0
        sev_e = float(e["claim_amount"].mean()) if len(e) else 0.0
        freq_chg = (freq_r / freq_e - 1) * 100 if freq_e else 0.0
        sev_chg = (sev_r / sev_e - 1) * 100 if sev_e else 0.0
        level = "red" if (freq_chg > 30 or sev_chg > 30) else "amber" if (freq_chg > 10 or sev_chg > 10) else "green"
        signals.append({"line_of_business": str(lob), "frequency_change_pct": round(freq_chg, 1), "severity_change_pct": round(sev_chg, 1),
                        "recent_claims_per_year": round(freq_r, 1), "signal": level})
    signals.sort(key=lambda s: {"red": 0, "amber": 1, "green": 2}[s["signal"]])
    return {"window_years": window, "latest_year": latest, "signals": signals,
            "red_flags": [s["line_of_business"] for s in signals if s["signal"] == "red"],
            "method": f"last {window} years vs earlier years; >30% rise = red, >10% = amber"}
