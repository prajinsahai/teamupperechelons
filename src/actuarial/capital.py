"""Aggregate loss simulation, VaR/TVaR, capital adequacy and retained-risk cost.

Monte Carlo: N years, claims ~ Poisson(lambda), each severity ~ lognormal(mu, sigma),
per-occurrence retention/limit applied per claim. Seeded -> deterministic.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.actuarial.frequency_severity import fit_frequency, fit_severity

SEED = 42
N_PATHS_FULL = 50_000
N_PATHS_THIN = 20_000
MAX_DRAWS = 4_000_000
_CACHE: dict[tuple, dict[str, Any]] = {}  # same data + layer -> same simulation (deterministic anyway)


def simulate_aggregate(
    df: pd.DataFrame,
    retention: float | None = None,
    limit: float | None = None,
    n_paths: int | None = None,
    seed: int = SEED,
) -> dict[str, Any]:
    """Annual aggregate loss distribution (gross, and net of a per-occurrence XoL layer if given)."""
    key = (id(df), len(df), float(df["claim_amount"].sum()), retention, limit, n_paths, seed)
    if key in _CACHE:
        return _CACHE[key]
    f = fit_frequency(df)
    s = fit_severity(df)
    lam, mu, sigma = f["poisson_lambda"], s["lognormal_mu"], s["lognormal_sigma"]
    thin = s["n"] < 200
    # bound total severity draws (~4M) so high-frequency books stay interactive:
    # 50k paths for lam<=80, scaling down to ~4.6k paths at lam~870.
    n = n_paths or min(N_PATHS_THIN if thin else N_PATHS_FULL, max(2_000, int(MAX_DRAWS / max(lam, 1.0))))
    rng = np.random.default_rng(seed)

    counts = rng.poisson(lam, size=n)
    total = counts.sum()
    sev = rng.lognormal(mu, sigma, size=int(total))
    if retention is not None:
        excess = np.clip(sev - retention, 0, None)
        ceded = np.clip(excess, 0, limit) if limit is not None else excess
        net_sev = sev - ceded
    else:
        ceded = np.zeros_like(sev)
        net_sev = sev
    idx = np.repeat(np.arange(n), counts)
    gross = np.bincount(idx, weights=sev, minlength=n)
    net = np.bincount(idx, weights=net_sev, minlength=n)
    recov = np.bincount(idx, weights=ceded, minlength=n)

    def stats(a: np.ndarray) -> dict[str, float]:
        v99 = float(np.percentile(a, 99))
        return {
            "mean": round(float(a.mean()), 2),
            "std": round(float(a.std()), 2),
            "p50": round(float(np.percentile(a, 50)), 2),
            "var_95": round(float(np.percentile(a, 95)), 2),
            "var_99": round(v99, 2),
            "var_99_5": round(float(np.percentile(a, 99.5)), 2),
            "tvar_99": round(float(a[a >= v99].mean()) if (a >= v99).any() else v99, 2),
            "max": round(float(a.max()), 2),
        }

    result = {
        "paths": int(n),
        "confidence_tier": "benchmark_thin_data" if thin else "full_monte_carlo",
        "frequency_lambda": lam,
        "severity_lognormal": {"mu": mu, "sigma": sigma},
        "retention": retention,
        "limit": limit,
        "gross": stats(gross),
        "net_of_reinsurance": stats(net) if retention is not None else None,
        "expected_recovery": round(float(recov.mean()), 2) if retention is not None else 0.0,
        "method": "Monte Carlo, Poisson frequency x lognormal severity, per-occurrence XoL applied per claim",
        "_gross_paths": gross,  # stripped before it reaches the LLM
        "_net_paths": net,
    }
    if len(_CACHE) > 64:
        _CACHE.clear()
    _CACHE[key] = result
    return result


def capital_adequacy(df: pd.DataFrame, capital_held: float, retention: float | None = None, limit: float | None = None) -> dict[str, Any]:
    """Probability that annual net loss exceeds capital; shortfall size; capital needed at 99.5%."""
    sim = simulate_aggregate(df, retention, limit)
    paths = sim["_net_paths"] if retention is not None else sim["_gross_paths"]
    breach = paths > capital_held
    shortfall = np.clip(paths - capital_held, 0, None)
    need_995 = float(np.percentile(paths, 99.5))
    basis = sim["net_of_reinsurance"] if retention is not None else sim["gross"]
    return {
        "capital_held": capital_held,
        "basis": "net_of_reinsurance" if retention is not None else "gross",
        "expected_annual_loss": basis["mean"],
        "shortfall_probability_pct": round(float(breach.mean() * 100), 3),
        "expected_shortfall_given_breach": round(float(shortfall[breach].mean()) if breach.any() else 0.0, 2),
        "capital_required_99_5": round(need_995, 2),
        "capital_surplus_or_deficit": round(capital_held - need_995, 2),
        "adequate_at_99_5": bool(capital_held >= need_995),
        "confidence_tier": sim["confidence_tier"],
        "method": "1-in-200 (99.5% VaR) capital test on simulated annual aggregate loss",
    }


def retained_risk_cost(df: pd.DataFrame, retention: float, limit: float, cost_of_capital: float = 0.08) -> dict[str, Any]:
    """Financial effect of what stays on the balance sheet under a given layer."""
    sim = simulate_aggregate(df, retention, limit)
    net = sim["net_of_reinsurance"]
    capital_for_retained = net["var_99_5"] - net["mean"]
    return {
        "retention": retention,
        "limit": limit,
        "expected_retained_loss": net["mean"],
        "retained_var_99_5": net["var_99_5"],
        "capital_backing_retained_risk": round(capital_for_retained, 2),
        "cost_of_capital_rate": cost_of_capital,
        "annual_cost_of_retained_risk": round(net["mean"] + capital_for_retained * cost_of_capital, 2),
        "expected_recovery": sim["expected_recovery"],
        "method": "expected retained loss + cost of capital on (VaR99.5 - mean) of retained distribution",
    }


def solvency_position(df: pd.DataFrame, capital_held: float, retention: float | None = None, limit: float | None = None) -> dict[str, Any]:
    """Solvency II-style view. SCR = 99.5% VaR of annual net loss minus expected loss (the
    unexpected-loss capital); own funds = capital held; SCR ratio = own funds / SCR.
    Warning below 120%, breach below 100%. Plus a simple liquidity cover of the 95% VaR."""
    sim = simulate_aggregate(df, retention, limit)
    basis = sim["net_of_reinsurance"] if retention is not None else sim["gross"]
    scr = max(basis["var_99_5"] - basis["mean"], 1.0)
    ratio = capital_held / scr * 100
    return {
        "own_funds": capital_held,
        "expected_annual_loss": basis["mean"],
        "var_99_5": basis["var_99_5"],
        "scr": round(scr, 2),
        "scr_ratio_pct": round(ratio, 1),
        "status": "breach" if ratio < 100 else "warning" if ratio < 120 else "adequate" if ratio < 200 else "strong",
        "surplus_over_scr": round(capital_held - scr, 2),
        "case_reserves_carried": round(float(df["reserve"].sum()), 2),
        "liquidity_cover_of_var95": round(capital_held / max(basis["var_95"], 1.0), 2),
        "basis": "net_of_reinsurance" if retention is not None else "gross",
        "confidence_tier": sim["confidence_tier"],
        "method": "SCR = VaR99.5 - mean of simulated annual loss; ratio = own funds / SCR; warn < 120%, breach < 100%",
    }


def public(result: dict[str, Any]) -> dict[str, Any]:
    """Strip private numpy arrays so the dict is JSON-serialisable for the LLM."""
    return {k: v for k, v in result.items() if not k.startswith("_")}
