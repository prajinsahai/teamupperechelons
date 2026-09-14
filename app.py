"""bizmax — AI Actuary dashboard.

Run:  .venv/Scripts/python -m streamlit run app.py
"""

import os
from pathlib import Path
from typing import Any

# Engine/model paths are project-relative; make them work from any launch directory.
os.chdir(Path(__file__).resolve().parent)

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.actuarial.ibnr import calculate_ibnr
from src.actuarial.reinsurance import DEFAULT_LIMIT, DEFAULT_RETENTION, calculate_reinsurance_recovery
from src.llm.agent import api_key_available, run_recommendation
from src.llm.config import MODEL
from src.ml.predict import anomaly_table, get_claims_intelligence

load_dotenv()

DATA_PATH = Path("data/claims.csv")

st.set_page_config(page_title="bizmax · AI Actuary", page_icon="📐", layout="wide")


# ---------------------------------------------------------------- helpers
def money(x: float) -> str:
    if abs(x) >= 1e9:
        return f"${x / 1e9:,.2f}B"
    if abs(x) >= 1e6:
        return f"${x / 1e6:,.1f}M"
    if abs(x) >= 1e3:
        return f"${x / 1e3:,.0f}K"
    return f"${x:,.0f}"


def money_md(x: float) -> str:
    """money() for markdown contexts (captions, metric deltas) — Streamlit treats $…$ as LaTeX."""
    return money(x).replace("$", "\\$")


@st.cache_data
def load_claims(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def run_engine(df: pd.DataFrame, retention: float, limit: float) -> dict[str, Any]:
    """Everything deterministic, computed once per (data, treaty) combination."""
    ibnr = calculate_ibnr(df)
    recovery = calculate_reinsurance_recovery(df, retention=retention, limit=limit)
    intel = get_claims_intelligence(df)
    net_exposure = ibnr["ultimate_loss"] - recovery["expected_recovery"]
    return {
        "portfolio_loss": ibnr["portfolio_loss"],
        "ibnr": ibnr["ibnr"],
        "ultimate_loss": ibnr["ultimate_loss"],
        "expected_recovery": recovery["expected_recovery"],
        "net_exposure": round(net_exposure, 2),
        "ibnr_detail": ibnr,
        "reinsurance_detail": recovery,
        "claims_intelligence": intel,
    }


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Reinsurance treaty")
    st.caption("Per-occurrence excess of loss")
    retention = st.number_input("Retention ($)", min_value=0, value=int(DEFAULT_RETENTION), step=50_000, format="%d")
    limit = st.number_input("Layer limit ($)", min_value=100_000, value=int(DEFAULT_LIMIT), step=250_000, format="%d")
    st.markdown("---")
    st.caption(f"LLM: `{MODEL}` via langchain-anthropic")
    st.caption("Math: pure Python · ML: scikit-learn (2 models)")


# ---------------------------------------------------------------- header
st.title("bizmax · AI Actuary")
st.caption("AI-native insurance optimization for corporate risk · all figures illustrative, modeled estimates")

if not DATA_PATH.exists():
    st.error(f"`{DATA_PATH}` not found. Generate it with `python -m src.data.make_sample_claims`.")
    st.stop()

df = load_claims(DATA_PATH)
try:
    out = run_engine(df, float(retention), float(limit))
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

ibnr = out["ibnr_detail"]
rec = out["reinsurance_detail"]
intel = out["claims_intelligence"]

# ---------------------------------------------------------------- 1. portfolio overview
st.subheader("Portfolio overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Portfolio loss (reported)", money(out["portfolio_loss"]),
          help="Sum of reported incurred across all accident years")
c2.metric("IBNR", money(out["ibnr"]),
          help=f"{ibnr['method']} · ultimate {money(ibnr['ultimate_loss'])}")
c3.metric("Expected recovery", money(out["expected_recovery"]),
          help=f"{rec['method']} · {rec['claims_in_layer']} claims in layer, {rec['claims_exhausting_layer']} exhaust it")
c4.metric("Net exposure", money(out["net_exposure"]),
          help="Ultimate loss (reported + IBNR) minus expected reinsurance recovery")

st.caption(
    f"{intel['claim_count']:,} claims · accident years "
    f"{min(intel['severity_by_year'])}–{max(intel['severity_by_year'])} · valuation {ibnr['valuation_year']} year-end · "
    f"paid to date {money_md(ibnr['paid_to_date'])} · case reserves {money_md(ibnr['case_reserves'])}"
)

st.markdown("---")

# ---------------------------------------------------------------- 2. claims intelligence
st.subheader("Claims intelligence")
sev = intel["severity_change_pct"]
w1, w2, w3 = st.columns(3)
w1.metric(
    f"{'⚠️ ' if abs(sev) >= 10 else ''}Severity trend ({intel['latest_year']} vs prior 3-yr avg)",
    f"{sev:+.1f}%",
    delta=f"{money_md(intel['latest_year_mean_severity'])} vs {money_md(intel['prior_years_mean_severity'])} mean claim",
    delta_color="inverse",
)
w2.metric(
    "⚠️ Anomalous claims detected" if intel["anomaly_count"] else "Anomalous claims detected",
    f"{intel['anomaly_count']}",
    delta=f"{intel['anomaly_rate_pct']:.1f}% of book · {money_md(intel['anomalous_amount_total'])} incurred",
    delta_color="off",
)
w3.metric(
    "RF predicted vs historical mean severity",
    money(intel["predicted_mean_severity"]),
    delta=f"{money_md(intel['historical_mean_severity'])} historical",
    delta_color="off",
)

g1, g2 = st.columns([3, 2])
with g1:
    fig, ax = plt.subplots(figsize=(7, 3))
    years = list(intel["severity_by_year"].keys())
    vals = [v / 1e3 for v in intel["severity_by_year"].values()]
    colors = ["#b08a4a" if y == intel["latest_year"] else "#c9c4bb" for y in years]
    ax.bar(years, vals, color=colors, edgecolor="#8a6d3b", linewidth=0.6)
    ax.set_ylabel("Mean claim ($K)")
    ax.set_title("Mean claim severity by accident year", fontsize=11, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
with g2:
    st.markdown("**Top anomalous claims** (Isolation Forest)")
    tbl = anomaly_table(df, top_n=8)
    st.dataframe(
        tbl.style.format({c: "${:,.0f}" for c in ["claim_amount", "reported_amount", "paid_amount", "reserve"]}
                         | {"anomaly_score": "{:.3f}"}),
        hide_index=True,
        height=300,
    )

st.markdown("---")

# ---------------------------------------------------------------- 3. AI actuary recommendation
st.subheader("AI Actuary recommendation")
st.caption("Claude reads the engine and model outputs through a tool call and explains — it never computes a figure itself.")

llm_summary = {
    "portfolio_loss": out["portfolio_loss"],
    "ibnr": out["ibnr"],
    "ultimate_loss": out["ultimate_loss"],
    "expected_recovery": out["expected_recovery"],
    "net_exposure": out["net_exposure"],
    "reinsurance": {k: v for k, v in rec.items()},
    "ibnr_by_accident_year": ibnr["by_accident_year"],
    "ibnr_assumptions": ibnr["assumptions"],
    "claims_intelligence": {k: v for k, v in intel.items() if k != "anomalous_claim_ids"},
    "top_anomalous_claim_ids": intel["anomalous_claim_ids"][:10],
}

if not api_key_available():
    st.warning("`ANTHROPIC_API_KEY` is not set. Add it to a `.env` file in the project root to enable the recommendation.")
else:
    if st.button("Generate recommendation", type="primary"):
        with st.spinner(f"Asking {MODEL}…"):
            try:
                text, calls = run_recommendation(llm_summary)
                st.session_state["recommendation"] = text
                st.session_state["tool_calls"] = calls
            except Exception as e:  # surface the real error in the UI during the demo
                st.error(f"LLM call failed: {e}")

    if "recommendation" in st.session_state:
        st.info(st.session_state["recommendation"], icon="📐")
        calls = st.session_state.get("tool_calls", [])
        st.caption(f"Tool calls made: {', '.join(c['name'] for c in calls) or 'none'}")

with st.expander("What the AI can see (tool output)"):
    st.json(llm_summary, expanded=False)
