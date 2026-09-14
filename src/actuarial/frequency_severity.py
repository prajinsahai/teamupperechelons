"""Frequency / severity fitting, pure premium, and stress tests. Pure Python + numpy."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def fit_frequency(df: pd.DataFrame) -> dict[str, Any]:
    """Claims per year: Poisson mean, and a negative-binomial check (variance/mean)."""
    per_year = df.groupby("accident_year").size()
    counts = per_year.to_numpy(dtype=float)
    mean = float(counts.mean())
    var = float(counts.var(ddof=1)) if len(counts) > 1 else mean
    dispersion = var / mean if mean else 1.0
    trend = 0.0
    if len(counts) >= 3:
        x = np.arange(len(counts))
        slope = np.polyfit(x, counts, 1)[0]
        trend = float(slope / mean) if mean else 0.0
    return {
        "years": int(len(counts)),
        "claims_per_year": {int(y): int(n) for y, n in per_year.items()},
        "poisson_lambda": round(mean, 3),
        "variance": round(var, 3),
        "dispersion_ratio": round(dispersion, 3),
        "distribution": "negative_binomial" if dispersion > 1.5 else "poisson",
        "annual_trend_pct": round(trend * 100, 1),
        "method": "Poisson MLE (lambda = mean count/year); NB flagged when var/mean > 1.5",
    }


def fit_severity(df: pd.DataFrame, column: str = "claim_amount") -> dict[str, Any]:
    """Lognormal fit on positive severities; empirical percentiles alongside."""
    x = df[column].to_numpy(dtype=float)
    x = x[x > 0]
    logx = np.log(x)
    mu, sigma = float(logx.mean()), float(logx.std(ddof=1)) if len(x) > 1 else 0.0
    ln_mean = float(np.exp(mu + sigma**2 / 2))
    return {
        "n": int(len(x)),
        "empirical_mean": round(float(x.mean()), 2),
        "empirical_median": round(float(np.median(x)), 2),
        "p90": round(float(np.percentile(x, 90)), 2),
        "p99": round(float(np.percentile(x, 99)), 2),
        "max": round(float(x.max()), 2),
        "lognormal_mu": round(mu, 4),
        "lognormal_sigma": round(sigma, 4),
        "lognormal_mean": round(ln_mean, 2),
        "lognormal_p99": round(float(np.exp(mu + 2.3263 * sigma)), 2),
        "tail": "heavy" if sigma > 1.0 else "moderate" if sigma > 0.6 else "light",
        "method": "lognormal MLE on log(claim_amount)",
    }


def pure_premium(df: pd.DataFrame, expense_load: float = 0.25, risk_margin: float = 0.10) -> dict[str, Any]:
    """Expected annual loss = frequency x severity; indicated premium with loads."""
    f = fit_frequency(df)
    s = fit_severity(df)
    expected = f["poisson_lambda"] * s["lognormal_mean"]
    indicated = expected * (1 + expense_load + risk_margin)
    return {
        "frequency_per_year": f["poisson_lambda"],
        "mean_severity": s["lognormal_mean"],
        "expected_annual_loss": round(expected, 2),
        "expense_load": expense_load,
        "risk_margin": risk_margin,
        "indicated_annual_premium": round(indicated, 2),
        "method": "pure premium = lambda x E[severity]; loaded for expenses and risk margin",
    }


def stress_test(df: pd.DataFrame, scenarios: dict[str, tuple[float, float]] | None = None) -> dict[str, Any]:
    """Scale frequency and severity; report expected loss and lognormal p99 per scenario."""
    scenarios = scenarios or {
        "base": (1.0, 1.0),
        "frequency_+25%": (1.25, 1.0),
        "severity_+30%": (1.0, 1.30),
        "combined_+25%_+30%": (1.25, 1.30),
        "catastrophe_freq_x2_sev_x1.5": (2.0, 1.5),
    }
    f = fit_frequency(df)
    s = fit_severity(df)
    base = f["poisson_lambda"] * s["lognormal_mean"]
    rows = []
    for name, (fm, sm) in scenarios.items():
        el = f["poisson_lambda"] * fm * s["lognormal_mean"] * sm
        rows.append({
            "scenario": name,
            "frequency_multiplier": fm,
            "severity_multiplier": sm,
            "expected_annual_loss": round(el, 2),
            "change_vs_base_pct": round((el / base - 1) * 100, 1) if base else 0.0,
        })
    return {"base_expected_annual_loss": round(base, 2), "scenarios": rows,
            "method": "deterministic scaling of fitted lambda and lognormal mean"}
