"""Total Cost of Risk and retention optimisation (Reinsurance Manager).

TCoR = reinsurance premium + expected retained loss + cost of capital on retained volatility.
Reinsurance premium is priced as expected ceded loss x (1 + loading): an explicit assumption.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.actuarial.capital import simulate_aggregate

DEFAULT_LOADING = 0.35  # reinsurer margin over expected ceded loss
DEFAULT_COC = 0.08  # cost of capital on retained VaR99.5 - mean


def tcor_for_layer(df: pd.DataFrame, retention: float, limit: float, loading: float = DEFAULT_LOADING, coc: float = DEFAULT_COC) -> dict[str, Any]:
    sim = simulate_aggregate(df, retention, limit)
    net = sim["net_of_reinsurance"]
    premium = sim["expected_recovery"] * (1 + loading)
    capital_charge = (net["var_99_5"] - net["mean"]) * coc
    return {
        "retention": retention,
        "limit": limit,
        "reinsurance_premium": round(premium, 2),
        "expected_retained_loss": net["mean"],
        "retained_var_99_5": net["var_99_5"],
        "capital_charge": round(capital_charge, 2),
        "tcor": round(premium + net["mean"] + capital_charge, 2),
        "expected_recovery": sim["expected_recovery"],
    }


def retention_sweep(df: pd.DataFrame, limit: float, retentions: list[float] | None = None, loading: float = DEFAULT_LOADING, coc: float = DEFAULT_COC) -> dict[str, Any]:
    """Evaluate TCoR across candidate retentions; return the minimum."""
    sev_p90 = float(df["claim_amount"].quantile(0.9))
    retentions = retentions or sorted({round(sev_p90 * m, -3) for m in (0.25, 0.5, 1, 1.5, 2, 3, 5)} | {250_000, 500_000, 1_000_000, 2_000_000})
    rows = [tcor_for_layer(df, r, limit, loading, coc) for r in retentions if r > 0]
    best = min(rows, key=lambda r: r["tcor"])
    return {
        "limit": limit,
        "loading": loading,
        "cost_of_capital": coc,
        "sweep": rows,
        "optimal_retention": best["retention"],
        "optimal_tcor": best["tcor"],
        "method": "TCoR = premium (expected ceded x (1+loading)) + expected retained loss + CoC x (VaR99.5 - mean)",
    }


def compare_structures(df: pd.DataFrame, retention: float, limit: float, agg_deductible: float | None = None) -> dict[str, Any]:
    """Per-occurrence XoL vs. an aggregate stop-loss at the same expected recovery budget."""
    xol = tcor_for_layer(df, retention, limit)
    sim = simulate_aggregate(df)  # gross paths
    gross = sim["_gross_paths"]
    agg_ded = agg_deductible or float(np.percentile(gross, 75))
    agg_limit = float(np.percentile(gross, 99.5)) - agg_ded
    ceded = np.clip(gross - agg_ded, 0, agg_limit)
    net = gross - ceded
    premium = float(ceded.mean()) * (1 + DEFAULT_LOADING)
    v995 = float(np.percentile(net, 99.5))
    charge = (v995 - float(net.mean())) * DEFAULT_COC
    stop_loss = {
        "aggregate_deductible": round(agg_ded, 2),
        "aggregate_limit": round(agg_limit, 2),
        "reinsurance_premium": round(premium, 2),
        "expected_retained_loss": round(float(net.mean()), 2),
        "retained_var_99_5": round(v995, 2),
        "capital_charge": round(charge, 2),
        "tcor": round(premium + float(net.mean()) + charge, 2),
        "expected_recovery": round(float(ceded.mean()), 2),
    }
    return {
        "per_occurrence_xol": xol,
        "aggregate_stop_loss": stop_loss,
        "cheaper_structure": "aggregate_stop_loss" if stop_loss["tcor"] < xol["tcor"] else "per_occurrence_xol",
        "tcor_difference": round(abs(stop_loss["tcor"] - xol["tcor"]), 2),
        "method": "same pricing basis for both; stop-loss attaches at p75 of gross annual loss, limit to p99.5",
    }
