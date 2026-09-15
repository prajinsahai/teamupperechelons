"""The six analysis tracks. Each = a role prompt + deterministic tools over the loaded data.

The model never computes. It decides which tools to call, reads their JSON, and
writes the advice. Every tool returns JSON-serialisable dicts with a `method` key
so the narrative can cite how a figure was produced.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd
from langchain_core.tools import BaseTool, tool

from src.actuarial import capital as cap
from src.actuarial import development as dev
from src.actuarial import exposure as expo
from src.actuarial import tcor as tc
from src.actuarial.frequency_severity import fit_frequency, fit_severity, pure_premium, stress_test
from src.actuarial.ibnr import calculate_ibnr
from src.actuarial.reinsurance import DEFAULT_LIMIT, DEFAULT_RETENTION, calculate_reinsurance_recovery
from src.data.ingest import DataBundle
from src.ml.predict import anomaly_table, get_claims_intelligence
from src.policy import wording

COMMON_RULES = (
    "\n\nRules you must follow:\n"
    "0. Reason first inside a <scratchpad>...</scratchpad> block (which tools to call, what the numbers say, "
    "what conflicts). The scratchpad is stripped before display; the answer comes after it.\n"
    "1. Never calculate, estimate, or invent a number. Every figure you state must be copied "
    "verbatim from a tool result. If a figure is not in a tool result, call a tool or say you cannot say.\n"
    "2. Call the tools you need first (several if useful), then answer.\n"
    "3. State the confidence tier / data thinness when the tools report it; with few claims, hedge.\n"
    "4. Format money with thousands separators and units (e.g. $43.4M) but keep the digits exactly as given.\n"
    "5. Answer in this structure, in plain professional English, max ~180 words:\n"
    "   FINDINGS: 2-4 bullets with the key numbers.\n"
    "   RECOMMENDATION: 1-3 bullets, each an action.\n"
    "   CONFIDENCE & CAVEATS: one line.\n"
    "6. The tools' `method` fields tell you how a number was produced; cite the method briefly when it matters."
)


@dataclass
class Track:
    key: str
    name: str
    blurb: str
    system_prompt: str
    default_question: str
    example_questions: list[str]
    build_tools: Callable[[DataBundle, dict[str, Any]], list[BaseTool]]
    metadata: Callable[[DataBundle, dict[str, Any]], dict[str, Any]]
    needs: str = "claims"  # "claims" | "documents"


def _sanitise(x: Any) -> Any:
    """What the model may see: no private (`_`) keys, no numpy types, no NaN/inf (invalid JSON)."""
    if isinstance(x, dict):
        return {str(k): _sanitise(v) for k, v in x.items() if not str(k).startswith("_")}
    if isinstance(x, (list, tuple)):
        return [_sanitise(v) for v in x]
    if isinstance(x, bool):
        return x
    if hasattr(x, "item"):  # numpy scalar
        x = x.item()
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, (int, float, str)) or x is None:
        return x
    return str(x)


def _j(x: Any) -> str:
    return json.dumps(_sanitise(x), allow_nan=False)


def _need_claims(b: DataBundle) -> pd.DataFrame:
    if b.claims is None or b.claims.empty:
        raise ValueError("No claims data loaded. Upload a claims CSV or import from Supabase.")
    return b.claims


# ---------------------------------------------------------------- shared tools
def _profile_tool(b: DataBundle) -> BaseTool:
    @tool
    def describe_loaded_data() -> str:
        """What data is loaded: files, row counts, accident years, lines of business, normalisation assumptions."""
        return _j(b.profile())
    return describe_loaded_data


# ---------------------------------------------------------------- 1. AI Actuary
def _actuary_tools(b: DataBundle, p: dict[str, Any]) -> list[BaseTool]:
    df = _need_claims(b)

    @tool
    def frequency_model() -> str:
        """Fitted claim frequency: claims per year, Poisson lambda, overdispersion, trend."""
        return _j(fit_frequency(df))

    @tool
    def severity_model() -> str:
        """Fitted claim severity: lognormal parameters, empirical mean/p90/p99/max, tail heaviness."""
        return _j(fit_severity(df))

    @tool
    def ibnr_reserve() -> str:
        """IBNR by the loss-development-factor method, per accident year, with assumptions."""
        return _j(calculate_ibnr(df))

    @tool
    def pricing() -> str:
        """Pure premium and indicated annual premium (expense load + risk margin)."""
        return _j(pure_premium(df))

    @tool
    def stress_scenarios() -> str:
        """Expected annual loss under frequency/severity stress scenarios."""
        return _j(stress_test(df))

    @tool
    def capital_test() -> str:
        """1-in-200 capital adequacy on the simulated gross annual loss (uses capital_held from the sidebar)."""
        return _j(cap.public(cap.capital_adequacy(df, p["capital_held"])))

    @tool
    def loss_development_triangle() -> str:
        """Cumulative incurred triangle (accident year x 12-month development), age-to-age factors, chain-ladder ultimates and IBNR per year."""
        return _j(dev.loss_triangle(df))

    return [_profile_tool(b), frequency_model, severity_model, ibnr_reserve, loss_development_triangle, pricing, stress_scenarios, capital_test]


def _actuary_meta(b: DataBundle, p: dict[str, Any]) -> dict[str, Any]:
    df = _need_claims(b)
    f, s, i = fit_frequency(df), fit_severity(df), calculate_ibnr(df)
    return {"computed_frequency": f["poisson_lambda"], "computed_severity": s["lognormal_mean"],
            "computed_expected_loss": round(f["poisson_lambda"] * s["lognormal_mean"], 2), "computed_ibnr": i["ibnr"],
            "claim_count": int(len(df)), "data_years": f["years"],
            "confidence_tier": "full_monte_carlo" if len(df) >= 200 else "benchmark_thin_data"}


# ---------------------------------------------------------------- 2. AI Claims Analyst
def _claims_tools(b: DataBundle, p: dict[str, Any]) -> list[BaseTool]:
    df = _need_claims(b)

    @tool
    def claims_intelligence() -> str:
        """Severity trend (latest year vs prior), RF-predicted vs historical severity, Isolation-Forest anomaly counts."""
        ci = get_claims_intelligence(df)
        return _j({k: v for k, v in ci.items() if k != "anomalous_claim_ids"} | {"top_anomalous_claim_ids": ci["anomalous_claim_ids"][:10],
                   "method": "latest accident year mean severity vs prior 3-year mean; RandomForest predicted severity; IsolationForest bottom-5% anomaly cut"})

    @tool
    def anomalous_claims(top_n: int = 10) -> str:
        """The most anomalous claims (Isolation Forest) with their amounts and development month."""
        return _j({"rows": anomaly_table(df, top_n).to_dict(orient="records"), "top_n": top_n,
                   "method": "Isolation Forest, lowest anomaly score first (bottom 5% of the book)"})

    @tool
    def development_pattern() -> str:
        """Paid/reported ratios by development month and reserve ratios by accident year."""
        return _j(dev.development_pattern(df))

    @tool
    def adverse_development() -> str:
        """Accident years whose reserves are unusually high for their age (adverse development signal)."""
        return _j(dev.adverse_development(df))

    @tool
    def large_loss_indicators() -> str:
        """Early-development claims already at p90+ severity, and mostly-unpaid heavy reserves."""
        return _j(dev.large_loss_indicators(df))

    @tool
    def pattern_shift_by_line() -> str:
        """Frequency and severity change per line of business, latest year vs prior."""
        return _j(dev.claim_pattern_shift(df))

    @tool
    def leakage_and_litigation() -> str:
        """Claims leakage (paid above reported), RF-outlier payments, litigation rate and severity multiple, social inflation vs CPI, reporting lag."""
        return _j(dev.claims_leakage(df))

    return [_profile_tool(b), claims_intelligence, anomalous_claims, development_pattern, adverse_development, large_loss_indicators, pattern_shift_by_line, leakage_and_litigation]


def _claims_meta(b: DataBundle, p: dict[str, Any]) -> dict[str, Any]:
    df = _need_claims(b)
    ci = get_claims_intelligence(df)
    ad = dev.adverse_development(df)
    return {"computed_severity_change_pct": ci["severity_change_pct"], "computed_anomaly_count": ci["anomaly_count"],
            "computed_adverse_years": len(ad["flagged_accident_years"]), "claim_count": int(len(df))}


# ---------------------------------------------------------------- 3. AI Reinsurance Manager
def _reins_tools(b: DataBundle, p: dict[str, Any]) -> list[BaseTool]:
    df = _need_claims(b)
    r, l = p["retention"], p["limit"]

    @tool
    def current_layer_recovery() -> str:
        """Recovery under the current per-occurrence XoL layer, claims in/exhausting the layer, uncovered tail."""
        return _j(calculate_reinsurance_recovery(df, r, l))

    @tool
    def simulated_net_position() -> str:
        """Monte Carlo gross vs net annual loss distribution (mean, VaR95/99/99.5, TVaR) under the current layer."""
        return _j(cap.public(cap.simulate_aggregate(df, r, l)))

    @tool
    def retention_optimisation() -> str:
        """Total Cost of Risk across candidate retentions at the current limit; the TCoR-minimising retention."""
        return _j(tc.retention_sweep(df, l))

    @tool
    def alternative_structures() -> str:
        """Per-occurrence XoL vs aggregate stop-loss on the same pricing basis."""
        return _j(tc.compare_structures(df, r, l))

    @tool
    def attachment_sensitivity() -> str:
        """TCoR at the current retention with the limit halved and doubled."""
        return _j({"half_limit": tc.tcor_for_layer(df, r, l / 2), "current": tc.tcor_for_layer(df, r, l), "double_limit": tc.tcor_for_layer(df, r, l * 2),
                   "method": "TCoR = premium (expected ceded x 1.35) + expected retained loss + 8% cost of capital on (VaR99.5 - mean), at the current retention"})

    @tool
    def quota_share_vs_excess_of_loss() -> str:
        """Quota share (30% cession, 25% commission) versus the per-occurrence XoL layer on the same TCoR basis."""
        return _j(tc.quota_share_vs_xol(df, r, l))

    return [_profile_tool(b), current_layer_recovery, simulated_net_position, retention_optimisation, alternative_structures, attachment_sensitivity, quota_share_vs_excess_of_loss]


def _reins_meta(b: DataBundle, p: dict[str, Any]) -> dict[str, Any]:
    df = _need_claims(b)
    sw = tc.retention_sweep(df, p["limit"])
    rec = calculate_reinsurance_recovery(df, p["retention"], p["limit"])
    return {"computed_optimal_retention": sw["optimal_retention"], "computed_optimal_tcor": sw["optimal_tcor"],
            "computed_expected_recovery": rec["expected_recovery"], "current_retention": p["retention"], "current_limit": p["limit"]}


# ---------------------------------------------------------------- 4. AI Capital Manager
def _capital_tools(b: DataBundle, p: dict[str, Any]) -> list[BaseTool]:
    df = _need_claims(b)
    r, l, c = p["retention"], p["limit"], p["capital_held"]

    @tool
    def capital_adequacy_net() -> str:
        """Shortfall probability, expected shortfall and 99.5% capital requirement, net of the current reinsurance layer."""
        return _j(cap.capital_adequacy(df, c, r, l))

    @tool
    def capital_adequacy_gross() -> str:
        """Same capital test with no reinsurance (gross) - shows what the treaty is doing for solvency."""
        return _j(cap.capital_adequacy(df, c))

    @tool
    def retained_risk_economics() -> str:
        """Expected retained loss, capital backing retained risk, and annual cost of retained risk."""
        return _j(cap.retained_risk_cost(df, r, l))

    @tool
    def loss_distribution() -> str:
        """Simulated annual aggregate loss distribution, gross and net."""
        return _j(cap.public(cap.simulate_aggregate(df, r, l)))

    @tool
    def stress_scenarios() -> str:
        """Expected annual loss under frequency/severity stress scenarios."""
        return _j(stress_test(df))

    @tool
    def solvency_ratio() -> str:
        """Solvency II-style SCR (= VaR99.5 - expected loss), own funds / SCR ratio with warning < 120% and breach < 100%, liquidity cover."""
        return _j(cap.solvency_position(df, c, r, l))

    return [_profile_tool(b), solvency_ratio, capital_adequacy_net, capital_adequacy_gross, retained_risk_economics, loss_distribution, stress_scenarios]


def _capital_meta(b: DataBundle, p: dict[str, Any]) -> dict[str, Any]:
    df = _need_claims(b)
    ca = cap.capital_adequacy(df, p["capital_held"], p["retention"], p["limit"])
    sp = cap.solvency_position(df, p["capital_held"], p["retention"], p["limit"])
    return {"computed_shortfall_probability_pct": ca["shortfall_probability_pct"], "computed_capital_required_99_5": ca["capital_required_99_5"],
            "computed_scr_ratio_pct": sp["scr_ratio_pct"],
            # net simulated mean; the Actuary owns the canonical gross `computed_expected_loss`
            "computed_expected_loss_net": ca["expected_annual_loss"], "capital_held": p["capital_held"], "confidence_tier": ca["confidence_tier"]}


# ---------------------------------------------------------------- 5. AI Risk Manager
def _risk_tools(b: DataBundle, p: dict[str, Any]) -> list[BaseTool]:
    df = _need_claims(b)

    @tool
    def exposure_movement() -> str:
        """Exposure, loss and loss-to-exposure by year and by line; year-over-year exposure change."""
        return _j(expo.exposure_movement(df))

    @tool
    def concentration() -> str:
        """Loss concentration (HHI) by line and risk class; top-5 claims share."""
        return _j(expo.concentration(df))

    @tool
    def emerging_signals() -> str:
        """Lines with rising frequency or severity in the last 2 years (red/amber/green)."""
        return _j(expo.emerging_signals(df))

    @tool
    def stress_scenarios() -> str:
        """Expected annual loss under frequency/severity stress scenarios."""
        return _j(stress_test(df))

    @tool
    def anomalous_claims(top_n: int = 10) -> str:
        """The most anomalous claims (Isolation Forest)."""
        return _j({"rows": anomaly_table(df, top_n).to_dict(orient="records"), "top_n": top_n,
                   "method": "Isolation Forest, lowest anomaly score first (bottom 5% of the book)"})

    @tool
    def monte_carlo_var() -> str:
        """Simulated annual aggregate loss (mean, VaR95/99/99.5, TVaR99) gross and net of the current layer."""
        return _j(cap.public(cap.simulate_aggregate(df, p["retention"], p["limit"])))

    return [_profile_tool(b), exposure_movement, concentration, emerging_signals, monte_carlo_var, stress_scenarios, anomalous_claims]


def _risk_meta(b: DataBundle, p: dict[str, Any]) -> dict[str, Any]:
    df = _need_claims(b)
    es, co = expo.emerging_signals(df), expo.concentration(df)
    return {"computed_red_flags": len(es["red_flags"]), "computed_hhi_by_line": co["hhi_by_line"], "computed_top5_share_pct": co["top5_claims_share_pct"]}


# ---------------------------------------------------------------- 6. AI Policy Analyst
def _policy_tools(b: DataBundle, p: dict[str, Any]) -> list[BaseTool]:
    docs = b.documents
    if not docs:
        raise ValueError("No policy documents loaded. Upload one or more policy PDFs.")
    modeled: dict[str, float] = {}
    if b.claims is not None and not b.claims.empty:
        s = fit_severity(b.claims)
        sim = cap.simulate_aggregate(b.claims)
        modeled = {"severity_p99": s["p99"], "largest_claim": s["max"], "expected_annual_loss": sim["gross"]["mean"], "var_99_5": sim["gross"]["var_99_5"]}

    @tool
    def extract_policy_clauses(categories: list[str] | None = None) -> str:
        """Clauses by category: exclusions, deductibles, limits, waiting_periods, conditions, endorsements, territory_period. Omit categories for all."""
        return _j(wording.extract_clauses(docs, categories))

    @tool
    def search_policy(query: str) -> str:
        """Keyword search across the policy documents; returns matching sentences."""
        return _j(wording.search_wording(docs, query))

    @tool
    def coverage_gap_check() -> str:
        """Compare stated limits/deductibles/waiting periods and exclusions against the modeled exposure (if claims are loaded)."""
        return _j(wording.coverage_gaps(docs, modeled))

    return [_profile_tool(b), extract_policy_clauses, search_policy, coverage_gap_check]


def _policy_meta(b: DataBundle, p: dict[str, Any]) -> dict[str, Any]:
    if not b.documents:
        return {"documents": 0}
    cl = wording.extract_clauses(b.documents)
    return {"documents": len(b.documents), **{f"computed_{k}_clauses": v for k, v in cl["counts"].items()}}


# ---------------------------------------------------------------- registry
TRACKS: dict[str, Track] = {
    "actuary": Track(
        "actuary", "AI Actuary",
        "Loss forecasting, frequency and severity, reserving, IBNR, loss development, pricing, capital adequacy, and stress tests.",
        "You are the Lead Reserving Actuary. Your tone is highly mathematical, objective, and conservative. "
        "Prioritise IBNR: use the chain-ladder triangle for mature data and the loss-development-factor (Bornhuetter-Ferguson "
        "style pattern) reserve for volatile or thin data, and say which you relied on and why. Fit frequency and severity, "
        "indicate pricing, run stress tests, and check 1-in-200 capital." + COMMON_RULES,
        "Are our reserves adequate, what is the expected annual loss, and what would we charge for this risk?",
        [
            "Are our reserves adequate, what is the expected annual loss, and what would we charge for this risk?",
            "What is our expected annual loss and how confident should we be in it given the data we have?",
            "Is the IBNR reserve sufficient? Which accident years are still immature and drive it?",
            "How heavy is our severity tail, and what does that mean for pricing and reinsurance?",
            "What premium should we charge next year, and how does it move under a 25% frequency stress?",
            "Do we hold enough capital for a 1-in-200 year, and how much surplus or deficit is there?",
        ],
        _actuary_tools, _actuary_meta),
    "claims": Track(
        "claims", "AI Claims Analyst",
        "Finds changing claim patterns, adverse development, drivers, and early indicators of large-loss risk.",
        "You are the Senior Claims Analyst. Analyse claim frequency, severity trends and litigation rates to identify "
        "claims leakage, adjust for social inflation, and surface adverse development and early large-loss indicators, "
        "naming the lines and claims responsible." + COMMON_RULES,
        "What is changing in our claims experience and where is large-loss risk emerging?",
        [
            "What is changing in our claims experience and where is large-loss risk emerging?",
            "Which line of business is driving the severity increase this year, and by how much?",
            "Are any accident years showing adverse reserve development? Which ones and why?",
            "Which open claims look most likely to become large losses, and what should we review first?",
            "Which claims are anomalous and what do they have in common?",
            "Is the shift in claims frequency, severity, or both — and is it broad-based or concentrated?",
        ],
        _claims_tools, _claims_meta),
    "reinsurance": Track(
        "reinsurance", "AI Reinsurance Manager",
        "Models retentions, attachment points, limits, catastrophe exposure, transfer costs, and alternative program structures.",
        "You are the Head of Reinsurance. Analyse the financial impact of transferring risk to the secondary market "
        "(quota share vs excess of loss vs aggregate stop-loss), calculate attachment points and exhaustion, and recommend "
        "the structure that minimises Total Cost of Risk while protecting the tail." + COMMON_RULES,
        "Is our current retention and limit right, and would a different structure be cheaper?",
        [
            "Is our current retention and limit right, and would a different structure be cheaper?",
            "What retention minimises our Total Cost of Risk, and how much would we save versus today?",
            "How much of our loss actually gets recovered under the current layer, and how many claims exhaust it?",
            "Would an aggregate stop-loss protect us better than the per-occurrence excess-of-loss?",
            "If we halved or doubled the limit, what happens to premium, retained loss and TCoR?",
            "What is our net 1-in-100 and 1-in-200 loss after reinsurance, and is the tail adequately transferred?",
        ],
        _reins_tools, _reins_meta),
    "capital": Track(
        "capital", "AI Capital Manager",
        "Evaluates capital sufficiency, shortfall probability, and the financial effect of retained risk.",
        "You are the Chief Capital Manager. Focus strictly on the balance sheet, liquidity and regulatory solvency "
        "(Solvency II logic: SCR, own funds, SCR ratio). Issue an explicit WARNING if the SCR ratio is below 120% and a "
        "BREACH notice below 100%. Quantify shortfall probability and the cost of retained risk." + COMMON_RULES,
        "Is our capital sufficient for the retained risk, and what is retained risk costing us?",
        [
            "Is our capital sufficient for the retained risk, and what is retained risk costing us?",
            "What is the probability we exceed our capital in a year, and how big is the shortfall if we do?",
            "How much capital do we need at the 99.5% level, gross versus net of reinsurance?",
            "What is the annual cost of the risk we retain, including the capital that backs it?",
            "How much solvency protection is the reinsurance treaty actually buying us?",
            "Under a severity +30% or catastrophe scenario, does our capital still hold?",
        ],
        _capital_tools, _capital_meta),
    "risk": Track(
        "risk", "AI Risk Manager",
        "Continuously monitors risk changes, exposure movement, and emerging threat signals across the enterprise.",
        "You are the Enterprise Risk Manager. Use the Monte Carlo / VaR outputs to map aggregate exposure, monitor "
        "exposure movement and concentration, stress the baseline against black-swan scenarios, and flag emerging "
        "threats by line with a red/amber/green view." + COMMON_RULES,
        "Where is exposure moving, where are we concentrated, and which lines show emerging risk?",
        [
            "Where is exposure moving, where are we concentrated, and which lines show emerging risk?",
            "Which lines of business are red or amber on emerging frequency or severity, and what changed?",
            "How concentrated is our loss in one line or a few large claims, and is that a problem?",
            "How has exposure moved year over year, and is loss growing faster than exposure?",
            "What are the top five claims and what share of total loss do they represent?",
            "Which risks would hurt us most under a combined frequency and severity stress?",
        ],
        _risk_tools, _risk_meta),
    "policy": Track(
        "policy", "AI Policy Analyst",
        "Reads policies, contracts, endorsements, exclusions, deductibles, and coverage requirements to surface gaps.",
        "You are the Lead Policy Analyst. Analyse coverage triggers, exclusions, sub-limits, deductibles, waiting periods, "
        "warranties and notification conditions to determine how likely a loss is to actually be paid, and surface coverage "
        "gaps against the modelled exposure." + COMMON_RULES,
        "What does this policy exclude or limit, and where are the coverage gaps against our exposure?",
        [
            "What does this policy exclude or limit, and where are the coverage gaps against our exposure?",
            "What are the exclusions in this wording, and which of them matter for our loss history?",
            "What are the deductibles, waiting periods and sub-limits, and do they leave us under-covered?",
            "Is the stated limit of liability enough for our modelled 1-in-200 loss or our largest claim?",
            "What conditions or notification requirements could void a claim, and are we meeting them?",
            "Does the policy cover cyber, contingent business interruption or supplier failure?",
        ],
        _policy_tools, _policy_meta, needs="documents"),
}
