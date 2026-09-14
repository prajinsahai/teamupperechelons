"""bizmax — AI Actuary dashboard: six analysis tracks over uploaded / imported data.

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
from src.data.ingest import DataBundle, build_bundle, bundle_from_paths
from src.llm.agent import api_key_available, run_agent
from src.llm.config import model_name
from src.ml.predict import anomaly_table, get_claims_intelligence
from src.prism.tracing import analysis_run, new_session_id, tracing_enabled
from src.tracks.registry import TRACKS

load_dotenv()

SAMPLE_FILES = ["data/claims.csv", "data/sample_policy.txt"]
IMPORT_DIR = Path("data/imported")
DEFAULT_CAPITAL = 60_000_000

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
    """money() for markdown contexts — Streamlit treats $…$ as LaTeX."""
    return money(x).replace("$", "\\$")


@st.cache_data(show_spinner=False)
def _bundle_from_uploads(files: tuple[tuple[str, bytes], ...]) -> DataBundle:
    return build_bundle(list(files))


@st.cache_data(show_spinner=False)
def _bundle_from_disk(paths: tuple[str, ...]) -> DataBundle:
    return bundle_from_paths(list(paths))


@st.cache_data(show_spinner=False)
def run_overview(df: pd.DataFrame, retention: float, limit: float) -> dict[str, Any]:
    ibnr = calculate_ibnr(df)
    recovery = calculate_reinsurance_recovery(df, retention=retention, limit=limit)
    intel = get_claims_intelligence(df)
    return {
        "portfolio_loss": ibnr["portfolio_loss"], "ibnr": ibnr["ibnr"], "ultimate_loss": ibnr["ultimate_loss"],
        "expected_recovery": recovery["expected_recovery"],
        "net_exposure": round(ibnr["ultimate_loss"] - recovery["expected_recovery"], 2),
        "ibnr_detail": ibnr, "reinsurance_detail": recovery, "claims_intelligence": intel,
    }


# ---------------------------------------------------------------- sidebar: data + programme
with st.sidebar:
    st.markdown("### Data")
    source = st.radio("Source", ["Upload files", "Supabase import", "Sample data"], label_visibility="collapsed")

    bundle: DataBundle | None = None
    if source == "Upload files":
        ups = st.file_uploader("Claims / policies / treaties (CSV, XLSX) and policy wordings (PDF, TXT)",
                               type=["csv", "xlsx", "xls", "pdf", "txt"], accept_multiple_files=True)
        if ups:
            bundle = _bundle_from_uploads(tuple((u.name, u.getvalue()) for u in ups))
    elif source == "Supabase import":
        st.caption("Reads tables via the Supabase REST API into `data/imported/*.csv`, then loads them like uploads.")
        tables = st.text_input("Table names (comma-separated)", placeholder="claims, policies, treaties")
        if st.button("Import from Supabase"):
            from src.data.supabase_import import import_tables
            try:
                with st.spinner("Importing…"):
                    paths = import_tables([t.strip() for t in tables.split(",") if t.strip()])
                st.success("Imported: " + ", ".join(f"{k} ({v.name})" for k, v in paths.items()))
            except Exception as e:
                st.error(str(e))
        imported = sorted(str(p) for p in IMPORT_DIR.glob("*.csv")) if IMPORT_DIR.exists() else []
        if imported:
            chosen = st.multiselect("Imported tables to load", imported, default=imported)
            extra_docs = st.file_uploader("Add policy PDFs (optional)", type=["pdf", "txt"], accept_multiple_files=True)
            if chosen:
                bundle = _bundle_from_disk(tuple(chosen))
                if extra_docs:
                    docs = build_bundle([(u.name, u.getvalue()) for u in extra_docs])
                    bundle.documents.update(docs.documents); bundle.sources += docs.sources
        else:
            st.info("No imported tables yet.")
    else:
        bundle = _bundle_from_disk(tuple(SAMPLE_FILES))

    st.markdown("### Programme")
    retention = st.number_input("Retention ($)", min_value=0, value=int(DEFAULT_RETENTION), step=50_000, format="%d")
    limit = st.number_input("Layer limit ($)", min_value=100_000, value=int(DEFAULT_LIMIT), step=250_000, format="%d")
    capital_held = st.number_input("Capital held ($)", min_value=0, value=DEFAULT_CAPITAL, step=1_000_000, format="%d")
    params = {"retention": float(retention), "limit": float(limit), "capital_held": float(capital_held)}

    st.markdown("---")
    st.caption(f"LLM: `{model_name()}` via NVIDIA NIM")
    st.caption("Math: pure Python · ML: scikit-learn (2 models)")
    st.caption("PRISM tracing: " + ("on" if tracing_enabled() else "off (set PRISMTRACE_* in .env)"))


# ---------------------------------------------------------------- header
st.title("bizmax · AI Actuary")
st.caption("AI-native insurance optimization for corporate risk · all figures illustrative, modeled estimates")

if bundle is None:
    st.info("Upload files, import from Supabase, or pick Sample data in the sidebar.")
    st.stop()

with st.expander(f"Loaded data — {len(bundle.sources)} source(s)", expanded=False):
    st.write(bundle.profile())

has_claims = bundle.claims is not None and not bundle.claims.empty

# ---------------------------------------------------------------- tabs
tab_names = ["Overview"] + [t.name for t in TRACKS.values()]
tabs = st.tabs(tab_names)

# --- Overview: the deterministic dashboard
with tabs[0]:
    if not has_claims:
        st.warning("No claims table loaded — the overview needs claims data.")
    else:
        df = bundle.claims
        try:
            out = run_overview(df, params["retention"], params["limit"])
        except FileNotFoundError as e:
            st.error(str(e)); st.stop()
        ibnr, rec, intel = out["ibnr_detail"], out["reinsurance_detail"], out["claims_intelligence"]

        st.subheader("Portfolio overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Portfolio loss (reported)", money(out["portfolio_loss"]), help="Sum of reported incurred across all accident years")
        c2.metric("IBNR", money(out["ibnr"]), help=f"{ibnr['method']} · ultimate {money(ibnr['ultimate_loss'])}")
        c3.metric("Expected recovery", money(out["expected_recovery"]),
                  help=f"{rec['method']} · {rec['claims_in_layer']} claims in layer, {rec['claims_exhausting_layer']} exhaust it")
        c4.metric("Net exposure", money(out["net_exposure"]), help="Ultimate loss (reported + IBNR) minus expected reinsurance recovery")
        st.caption(f"{intel['claim_count']:,} claims · accident years {min(intel['severity_by_year'])}–{max(intel['severity_by_year'])} · "
                   f"valuation {ibnr['valuation_year']} year-end · paid to date {money_md(ibnr['paid_to_date'])} · case reserves {money_md(ibnr['case_reserves'])}")

        st.subheader("Claims intelligence")
        sev = intel["severity_change_pct"]
        w1, w2, w3 = st.columns(3)
        w1.metric(f"{'⚠️ ' if abs(sev) >= 10 else ''}Severity trend ({intel['latest_year']} vs prior 3-yr avg)", f"{sev:+.1f}%",
                  delta=f"{money_md(intel['latest_year_mean_severity'])} vs {money_md(intel['prior_years_mean_severity'])} mean claim", delta_color="inverse")
        w2.metric("⚠️ Anomalous claims detected" if intel["anomaly_count"] else "Anomalous claims detected", f"{intel['anomaly_count']}",
                  delta=f"{intel['anomaly_rate_pct']:.1f}% of book · {money_md(intel['anomalous_amount_total'])} incurred", delta_color="off")
        w3.metric("RF predicted vs historical mean severity", money(intel["predicted_mean_severity"]),
                  delta=f"{money_md(intel['historical_mean_severity'])} historical", delta_color="off")

        g1, g2 = st.columns([3, 2])
        with g1:
            fig, ax = plt.subplots(figsize=(7, 3))
            years = list(intel["severity_by_year"].keys())
            vals = [v / 1e3 for v in intel["severity_by_year"].values()]
            ax.bar(years, vals, color=["#b08a4a" if y == intel["latest_year"] else "#c9c4bb" for y in years], edgecolor="#8a6d3b", linewidth=0.6)
            ax.set_ylabel("Mean claim ($K)"); ax.set_title("Mean claim severity by accident year", fontsize=11, loc="left")
            ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=0.25)
            st.pyplot(fig, width="stretch"); plt.close(fig)
        with g2:
            st.markdown("**Top anomalous claims** (Isolation Forest)")
            tbl = anomaly_table(df, top_n=8)
            st.dataframe(tbl.style.format({c: "${:,.0f}" for c in ["claim_amount", "reported_amount", "paid_amount", "reserve"]} | {"anomaly_score": "{:.3f}"}),
                         hide_index=True, height=300)


# --- One tab per track
def render_track(track_key: str) -> None:
    track = TRACKS[track_key]
    st.markdown(f"**{track.name}** — {track.blurb}")
    ready = has_claims if track.needs == "claims" else bool(bundle.documents)
    if not ready:
        st.warning("Needs claims data." if track.needs == "claims" else "Needs at least one policy document (PDF/TXT).")
        return
    if not api_key_available():
        st.warning("`NVIDIA_API_KEY` is not set in `.env`."); return

    question = st.text_area("Question", value=track.default_question, key=f"q_{track_key}", height=70)
    state_key = f"result_{track_key}"
    if st.button(f"Run {track.name}", type="primary", key=f"run_{track_key}"):
        session_id = new_session_id(track_key)
        with st.spinner(f"{track.name} is working ({model_name()})…"):
            try:
                tools = track.build_tools(bundle, params)
                meta = {"track": track_key, **track.metadata(bundle, params)}
                with analysis_run(session_id, meta):
                    text, calls = run_agent(track.system_prompt, question, tools, run_name=f"{track_key}_track")
                st.session_state[state_key] = {"text": text, "calls": calls, "session": session_id, "meta": meta}
            except Exception as e:
                st.error(f"{track.name} failed: {e}")

    if state_key in st.session_state:
        r = st.session_state[state_key]
        st.info(r["text"] or "_(empty answer — try again)_", icon="📐")
        st.caption("Tools called: " + (", ".join(c["name"] for c in r["calls"]) or "none")
                   + (f" · PRISM session `{r['session']}`" if tracing_enabled() else ""))
        with st.expander("Computed figures sent to PRISM as truth (metadata)"):
            st.json(r["meta"])


for i, key in enumerate(TRACKS, start=1):
    with tabs[i]:
        render_track(key)
