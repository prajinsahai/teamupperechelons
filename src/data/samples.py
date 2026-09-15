"""Four illustrative businesses with deliberately different risk profiles.

    .venv/Scripts/python -m src.data.samples        # (re)generate data/samples/<slug>/

Each sample = claims.csv (engine schema + litigated/status/report_lag_days columns),
policy.txt (a wording the Policy Analyst can read), profile.json (name, focus,
programme defaults). All synthetic, all seeded.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SAMPLES_DIR = Path("data/samples")
RISK_CLASSES = ["A", "B", "C", "D"]


@dataclass
class BusinessProfile:
    slug: str
    name: str
    tagline: str
    focus: str
    lines: dict[str, float]  # line -> share of claims
    years: list[int]
    claims_per_year: float
    year_multipliers: dict[int, float]  # volatility by year
    sev_median: float
    sev_sigma: float
    tail_months: int  # development horizon
    litigation_rate: float
    exposure_median: float
    limits: list[float]
    deductibles: list[float]
    programme: dict[str, float]  # retention, limit, capital_held
    quirks: dict[str, Any] = field(default_factory=dict)
    seed: int = 7


PROFILES: dict[str, BusinessProfile] = {
    "velocity": BusinessProfile(
        slug="velocity", name="Velocity Fleet & Freight",
        tagline="High frequency / low severity — a 2,400-truck logistics fleet.",
        focus="Claims leakage, fraud detection, attritional loss control.",
        lines={"auto_liability": 0.40, "physical_damage": 0.30, "cargo": 0.20, "workers_comp": 0.10},
        years=list(range(2018, 2025)), claims_per_year=850,
        year_multipliers={2018: 0.9, 2019: 0.95, 2020: 0.7, 2021: 1.0, 2022: 1.1, 2023: 1.15, 2024: 1.25},
        sev_median=6_500, sev_sigma=0.75, tail_months=30, litigation_rate=0.04,
        exposure_median=180_000, limits=[1e6, 2e6], deductibles=[5_000, 10_000, 25_000],
        programme={"retention": 250_000, "limit": 2_000_000, "capital_held": 2_000_000},
        quirks={"fraud_rate": 0.03, "duplicate_rate": 0.01},
    ),
    "nexus": BusinessProfile(
        slug="nexus", name="Nexus Cloud Sec",
        tagline="Low frequency / high severity — a cloud security provider with 900 enterprise tenants.",
        focus="Cyber catastrophe, systemic accumulation, reinsurance treaty design.",
        lines={"cyber_breach": 0.45, "business_interruption": 0.25, "tech_eo": 0.20, "d_and_o": 0.10},
        years=list(range(2018, 2025)), claims_per_year=20,
        year_multipliers={2018: 0.6, 2019: 0.8, 2020: 1.0, 2021: 1.3, 2022: 1.0, 2023: 1.2, 2024: 1.6},
        sev_median=420_000, sev_sigma=1.35, tail_months=36, litigation_rate=0.12,
        exposure_median=12_000_000, limits=[10e6, 25e6, 50e6], deductibles=[250_000, 500_000, 1_000_000],
        programme={"retention": 2_000_000, "limit": 25_000_000, "capital_held": 50_000_000},
        quirks={"catastrophes": [(2021, 28_000_000), (2024, 41_000_000)]},
    ),
    "aegis": BusinessProfile(
        slug="aegis", name="Aegis Healthcare",
        tagline="Medium frequency / high severity — a 14-hospital regional health system.",
        focus="Long-tail liabilities, IBNR reserving, litigation and social inflation.",
        lines={"medical_malpractice": 0.45, "general_liability": 0.20, "workers_comp": 0.25, "property": 0.10},
        years=list(range(2016, 2025)), claims_per_year=130,
        year_multipliers={y: 1.0 + 0.03 * (y - 2016) for y in range(2016, 2025)},
        sev_median=95_000, sev_sigma=1.15, tail_months=96, litigation_rate=0.28,
        exposure_median=4_500_000, limits=[2e6, 5e6, 10e6], deductibles=[50_000, 100_000, 250_000],
        programme={"retention": 1_000_000, "limit": 10_000_000, "capital_held": 20_000_000},
        quirks={"social_inflation": 0.07},
    ),
    "terrafirma": BusinessProfile(
        slug="terrafirma", name="TerraFirma Civil",
        tagline="Volatile / project-based — a civil contractor running 30-40 live sites.",
        focus="Weather delays, surety exposure, stress testing of lumpy project risk.",
        lines={"builders_risk": 0.30, "weather_delay": 0.25, "contractor_liability": 0.20, "equipment": 0.15, "surety": 0.10},
        years=list(range(2018, 2025)), claims_per_year=70,
        year_multipliers={2018: 0.8, 2019: 1.0, 2020: 0.5, 2021: 0.9, 2022: 1.9, 2023: 0.8, 2024: 1.3},
        sev_median=60_000, sev_sigma=1.05, tail_months=48, litigation_rate=0.10,
        exposure_median=2_500_000, limits=[2e6, 5e6, 10e6], deductibles=[25_000, 50_000, 100_000],
        programme={"retention": 500_000, "limit": 5_000_000, "capital_held": 5_500_000},
        quirks={"weather_years": [2022, 2024]},
    ),
}


def _make_claims(p: BusinessProfile) -> pd.DataFrame:
    rng = np.random.default_rng(p.seed)
    rows = []
    cid = 1
    lines, shares = list(p.lines), np.array(list(p.lines.values()))
    latest = max(p.years)
    for y in p.years:
        n = int(rng.poisson(p.claims_per_year * p.year_multipliers.get(y, 1.0)))
        line = rng.choice(lines, size=n, p=shares / shares.sum())
        risk = rng.choice(RISK_CLASSES, size=n, p=[0.3, 0.4, 0.2, 0.1])
        ridx = np.array([RISK_CLASSES.index(r) for r in risk])
        exposure = rng.lognormal(np.log(p.exposure_median), 0.5, size=n).round(-3)
        prev = rng.poisson(0.8 + 0.5 * ridx, size=n)
        ded = rng.choice(p.deductibles, size=n)
        lim = rng.choice(p.limits, size=n)
        # severity: lognormal with line and year effects
        line_mult = {l: m for l, m in zip(lines, np.linspace(0.7, 1.6, len(lines)))}
        mu = np.log(p.sev_median) + 0.2 * ridx + np.log([line_mult[l] for l in line])
        if "social_inflation" in p.quirks:
            mu = mu + np.log(1 + p.quirks["social_inflation"]) * (y - p.years[0])
        if "weather_years" in p.quirks and y in p.quirks["weather_years"]:
            mu = mu + np.where(line == "weather_delay", np.log(2.2), 0.0)
        amount = np.minimum(rng.lognormal(mu, p.sev_sigma, size=n), lim).round(0)
        age = (latest - y + 1) * 12
        dev = np.clip(rng.integers(max(1, age - 11), age + 1, size=n), 1, p.tail_months)
        dev_frac = np.clip(dev / (p.tail_months * 0.6), 0.05, 1.0)
        reported = (amount * rng.uniform(0.9, 1.12, size=n)).round(0)
        paid = (amount * dev_frac * rng.uniform(0.8, 1.0, size=n)).round(0)
        reserve = np.maximum(reported - paid, 0).round(0)
        litigated = rng.random(n) < p.litigation_rate * (1 + 0.5 * (ridx >= 2))
        status = np.where(dev_frac >= 0.95, "closed", "open")
        lag = rng.gamma(2.0, 12.0, size=n).round(0)
        for i in range(n):
            rows.append({
                "claim_id": f"{p.slug[:3].upper()}-{cid:05d}", "accident_year": y, "line_of_business": line[i],
                "risk_class": risk[i], "exposure": exposure[i], "previous_claims": int(prev[i]), "deductible": float(ded[i]),
                "coverage_limit": float(lim[i]), "development_month": int(dev[i]), "claim_amount": float(amount[i]),
                "reported_amount": float(reported[i]), "paid_amount": float(paid[i]), "reserve": float(reserve[i]),
                "litigated": bool(litigated[i]), "status": status[i], "report_lag_days": float(lag[i]),
            })
            cid += 1
    df = pd.DataFrame(rows)

    # business-specific quirks
    if "catastrophes" in p.quirks:
        for y, amt in p.quirks["catastrophes"]:
            idx = df.index[(df.accident_year == y) & (df.line_of_business == "cyber_breach")]
            if len(idx):
                i = idx[0]
                df.loc[i, ["claim_amount", "reported_amount"]] = amt, amt * 1.05
                df.loc[i, "paid_amount"] = amt * 0.55
                df.loc[i, "reserve"] = df.loc[i, "reported_amount"] - df.loc[i, "paid_amount"]
                df.loc[i, "coverage_limit"] = max(df.loc[i, "coverage_limit"], 50e6)
    if "fraud_rate" in p.quirks:
        k = int(len(df) * p.quirks["fraud_rate"])
        idx = rng.choice(df.index, size=k, replace=False)
        df.loc[idx, "paid_amount"] = df.loc[idx, "reported_amount"] * rng.uniform(1.6, 2.4, size=k)
        df.loc[idx, "report_lag_days"] = rng.uniform(120, 400, size=k).round(0)
        d = int(len(df) * p.quirks["duplicate_rate"])
        dup = df.sample(d, random_state=p.seed).copy()
        dup["claim_id"] = [f"{p.slug[:3].upper()}-D{i:04d}" for i in range(d)]
        df = pd.concat([df, dup], ignore_index=True)
    return df


POLICIES: dict[str, str] = {
    "velocity": """COMMERCIAL AUTO AND MOTOR CARRIER LIABILITY POLICY — VELOCITY FLEET & FREIGHT.
Limit of Liability: USD 2,000,000 each accident, combined single limit. Aggregate limit USD 10,000,000.
Cargo: sub-limit USD 250,000 per conveyance. Deductible: USD 10,000 each and every loss; USD 25,000 for physical damage to owned units.
Exclusions: This policy does not cover loss arising from war, nuclear risks, or the use of any vehicle while a driver is under the influence.
Not covered: theft of cargo from an unattended vehicle unless parked in a secured yard; wear and tear; mechanical breakdown; pollution.
Conditions: The insured shall notify the insurer within 14 days of any occurrence. Telematics data must be made available on request; failure to do so is a condition precedent to liability.
Fraud: Any claim that is fraudulent in whole or in part shall be void and the insurer may recover payments made.
Territory: United States, Canada and Mexico within 100 miles of the border. Policy period: 12 months from inception.
Endorsement 4: Hired and non-owned auto extension, sub-limit USD 1,000,000.""",
    "nexus": """CYBER, TECHNOLOGY ERRORS & OMISSIONS AND CONTINGENT BUSINESS INTERRUPTION POLICY — NEXUS CLOUD SEC.
Limit of Liability: USD 25,000,000 each claim and in the aggregate. Sub-limit: USD 5,000,000 for regulatory fines and penalties where insurable by law.
Business interruption is subject to a waiting period of 12 hours before cover attaches; dependent business interruption sub-limit USD 10,000,000.
Deductible: USD 500,000 each claim. Retention for systemic events affecting more than 50 tenants: USD 2,000,000.
Exclusions: This policy does not cover loss arising from war or state-sponsored cyber operations attributed by a competent authority; infrastructure failure of a public utility; unencrypted portable devices; prior known circumstances.
Not covered: betterment of systems; contractual penalties beyond those the insured would have owed absent contract; loss of intellectual property value.
Conditions: Notification within 72 hours of discovery of a security event is a condition precedent. Multi-factor authentication must be maintained on all privileged accounts (warranty).
Territory: worldwide excluding sanctioned countries. Policy period: 12 months. Endorsement 7: Extension for cryptojacking and ransomware negotiation costs, sub-limit USD 2,000,000.""",
    "aegis": """HOSPITAL PROFESSIONAL AND GENERAL LIABILITY POLICY (CLAIMS-MADE) — AEGIS HEALTHCARE.
Limit of Liability: USD 10,000,000 each claim; USD 30,000,000 aggregate. Sub-limit: USD 1,000,000 for sexual misconduct; USD 500,000 for regulatory defence.
Self-insured retention: USD 1,000,000 each claim inclusive of defence costs. Deductible for general liability: USD 100,000.
Exclusions: This policy does not cover punitive damages where uninsurable, criminal acts, claims arising from services outside the scope of licensure, or bodily injury from asbestos or pollution.
Not covered: claims first made after the policy period unless an extended reporting period is purchased; contractual liability; employment practices.
Conditions: Claims must be reported within the policy period or within 60 days thereafter. The insured shall maintain a peer review programme (warranty). Notification within 30 days of any circumstance likely to give rise to a claim.
Retroactive date: 1 January 2010. Territory: United States. Policy period: 12 months.
Endorsement 2: Batch clause — related claims arising from a single cause are treated as one claim first made when the earliest was reported.""",
    "terrafirma": """CONTRACTORS ALL RISKS, DELAY IN START-UP AND SURETY FACILITY — TERRAFIRMA CIVIL.
Limit of Liability: USD 5,000,000 each occurrence for physical damage; delay in start-up sub-limit USD 3,000,000 per project with a waiting period of 21 days.
Deductible: USD 50,000 each and every loss; USD 100,000 for named windstorm and flood; 5% of loss subject to minimum USD 250,000 for earthquake.
Exclusions: This policy does not cover loss arising from war, nuclear risks, faulty design (unless resulting damage), wear and tear, or contractual penalties for late completion not caused by insured physical damage.
Not covered: consequential loss other than delay in start-up; loss of a project already 90% complete without endorsement; surety bond calls arising from insolvency of the principal.
Conditions: The insured shall notify within 7 days of any occurrence and within 48 hours of any weather event exceeding 100 mm rainfall in 24 hours. Site security warranty: 24-hour manned security on sites over USD 20,000,000 contract value.
Surety: aggregate facility USD 60,000,000; single bond limit USD 15,000,000; indemnity by parent company.
Territory: United States and Canada. Policy period: 12 months. Endorsement 9: Off-site storage and transit extension, sub-limit USD 1,000,000.""",
}


def generate_all(out_dir: Path = SAMPLES_DIR) -> list[Path]:
    made = []
    for p in PROFILES.values():
        d = out_dir / p.slug
        d.mkdir(parents=True, exist_ok=True)
        df = _make_claims(p)
        df.to_csv(d / "claims.csv", index=False)
        (d / "policy.txt").write_text(POLICIES[p.slug], encoding="utf-8")
        meta = asdict(p)
        meta["year_multipliers"] = {str(k): v for k, v in p.year_multipliers.items()}
        meta["claim_count"] = int(len(df))
        meta["total_claim_amount"] = float(df["claim_amount"].sum())
        (d / "profile.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        made.append(d)
        print(f"{p.name:26s} {len(df):5,} claims  mean ${df.claim_amount.mean():>12,.0f}  max ${df.claim_amount.max():>13,.0f}  -> {d}")
    return made


def list_samples(base: Path = SAMPLES_DIR) -> dict[str, dict[str, Any]]:
    """slug -> profile.json contents, for the UI picker."""
    out = {}
    if base.exists():
        for d in sorted(base.iterdir()):
            pj = d / "profile.json"
            if pj.exists():
                out[d.name] = json.loads(pj.read_text(encoding="utf-8"))
    return out


def sample_files(slug: str, base: Path = SAMPLES_DIR) -> list[str]:
    d = base / slug
    return [str(d / "claims.csv"), str(d / "policy.txt")]


if __name__ == "__main__":
    generate_all()
