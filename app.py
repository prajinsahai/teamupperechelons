"""bizmax AI Actuary operations workspace.

Run: .venv/Scripts/python -m streamlit run app.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

os.chdir(Path(__file__).resolve().parent)

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.actuarial import capital as cap
from src.actuarial import development as dev
from src.actuarial import exposure as expo
from src.actuarial import tcor as tc
from src.actuarial.frequency_severity import fit_frequency, fit_severity, stress_test
from src.actuarial.ibnr import calculate_ibnr
from src.actuarial.reinsurance import calculate_reinsurance_recovery
from src.charts import figures as F
from src.core.orchestrator import CoreResult, run_core
from src.core.synthesizer import strip_scratchpad
from src.data.ingest import DataBundle, build_bundle, bundle_from_paths
from src.data.samples import list_samples, sample_files
from src.llm.agent import api_key_available, run_agent
from src.llm.config import model_name
from src.ml.predict import anomaly_table, get_claims_intelligence
from src.policy import wording
from src.prism.tracing import analysis_run, new_session_id, tracing_enabled
from src.tracks.registry import TRACKS
from src.ui.magi import render_magi
from src.ui.theme import apply_theme, section_header, sidebar_brand, sidebar_system_card
from src.ui.workspace import (
    agent_network, clause_card, evidence_detail, metric_grid, panel_header,
    portfolio_bar, reinsurance_layers, risk_metric_list, workflow_trace,
    workspace_header,
)

load_dotenv()
IMPORT_DIR = Path("data/imported")
PAGES = ("Dashboard", "Claims Analysis", "Policy Intelligence", "Reinsurance", "Actuarial Engine", "Audit Trail")
NAV_LABELS = {
    "Dashboard": "▦  Dashboard",
    "Claims Analysis": "⌁  Claims Analysis",
    "Policy Intelligence": "▤  Policy Intelligence",
    "Reinsurance": "⬡  Reinsurance",
    "Actuarial Engine": "▦  Actuarial Engine",
    "Audit Trail": "◷  Audit Trail",
}

st.set_page_config(page_title="bizmax · AI Actuary", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")
apply_theme()


def money(value: float) -> str:
    value = float(value or 0)
    if abs(value) >= 1e9:
        return f"${value / 1e9:,.2f}B"
    if abs(value) >= 1e6:
        return f"${value / 1e6:,.1f}M"
    if abs(value) >= 1e3:
        return f"${value / 1e3:,.0f}K"
    return f"${value:,.0f}"


def md(text: str) -> str:
    return (text or "").replace("$", "\\$")


def show(fig) -> None:
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def sheet(frame: pd.DataFrame, name: str, money_cols: list[str] | None = None, height: int = 290) -> None:
    fmt = {column: "${:,.0f}" for column in (money_cols or []) if column in frame.columns}
    st.dataframe(frame.style.format(fmt) if fmt else frame, hide_index=True, height=height, width="stretch")
    st.download_button(f"Export {name}.csv", frame.to_csv(index=False).encode(), f"{name}.csv", "text/csv", key=f"download_{name}")


@st.cache_data(show_spinner=False)
def _bundle_from_uploads(files: tuple[tuple[str, bytes], ...]) -> DataBundle:
    return build_bundle(list(files))


@st.cache_data(show_spinner=False)
def _bundle_from_disk_versioned(paths: tuple[str, ...], _stamps: tuple[tuple[float, int], ...]) -> DataBundle:
    return bundle_from_paths(list(paths))


def _bundle_from_disk(paths: tuple[str, ...]) -> DataBundle:
    stamps = tuple((Path(path).stat().st_mtime, Path(path).stat().st_size) for path in paths)
    return _bundle_from_disk_versioned(paths, stamps)


@st.cache_data(show_spinner=False)
def engine(df: pd.DataFrame, retention: float, limit: float, capital_held: float) -> dict[str, Any]:
    """Everything shown by the workspace, computed deterministically per portfolio."""
    return {
        "ibnr": calculate_ibnr(df), "rec": calculate_reinsurance_recovery(df, retention, limit),
        "intel": get_claims_intelligence(df), "freq": fit_frequency(df), "sev": fit_severity(df),
        "tri": dev.loss_triangle(df), "stress": stress_test(df),
        "sim": cap.simulate_aggregate(df, retention, limit),
        "solv": cap.solvency_position(df, capital_held, retention, limit),
        "adq": cap.capital_adequacy(df, capital_held, retention, limit),
        "expo": expo.exposure_movement(df), "conc": expo.concentration(df),
        "emerg": expo.emerging_signals(df), "sweep": tc.retention_sweep(df, limit),
        "cmp": tc.compare_structures(df, retention, limit),
        "qs": tc.quota_share_vs_xol(df, retention, limit),
        "devp": dev.development_pattern(df), "adv": dev.adverse_development(df),
        "leak": dev.claims_leakage(df),
    }


def _merge_documents(base: DataBundle, uploads: list[Any]) -> DataBundle:
    if not uploads:
        return base
    docs = build_bundle([(upload.name, upload.getvalue()) for upload in uploads])
    return DataBundle(
        claims=base.claims, tables=dict(base.tables), documents={**base.documents, **docs.documents},
        notes=[*base.notes, *docs.notes], sources=[*base.sources, *docs.sources],
    )


# Sidebar navigation and inputs.
samples = list_samples()
with st.sidebar:
    sidebar_brand()
    selected_label = st.radio("Workspace", [NAV_LABELS[p] for p in PAGES], label_visibility="collapsed", key="workspace_navigation")
    page = next(name for name, label in NAV_LABELS.items() if label == selected_label)

    with st.expander("Portfolio & data", expanded=True):
        source = st.selectbox("Data source", ["Sample business", "Upload files", "Supabase import"])
        bundle: DataBundle | None = None
        profile: dict[str, Any] | None = None
        if source == "Sample business":
            slug = st.selectbox("Business", list(samples), format_func=lambda key: samples[key]["name"])
            profile = samples[slug]
            st.caption(f"{profile['tagline']}  \n**Focus:** {profile['focus']}")
            bundle = _bundle_from_disk(tuple(sample_files(slug)))
        elif source == "Upload files":
            uploads = st.file_uploader("Claims, treaties and wordings", type=["csv", "xlsx", "xls", "pdf", "txt"], accept_multiple_files=True)
            if uploads:
                bundle = _bundle_from_uploads(tuple((upload.name, upload.getvalue()) for upload in uploads))
        else:
            st.caption("Imports Supabase tables to local CSV before analysis.")
            tables = st.text_input("Table names", placeholder="claims, policies, treaties")
            if st.button("Import from Supabase", width="stretch"):
                from src.data.supabase_import import import_tables
                try:
                    with st.spinner("Importing tables…"):
                        paths = import_tables([item.strip() for item in tables.split(",") if item.strip()])
                    st.success("Imported " + ", ".join(paths))
                except Exception as exc:
                    st.error(str(exc))
            imported = sorted(str(path) for path in IMPORT_DIR.glob("*.csv")) if IMPORT_DIR.exists() else []
            if imported:
                chosen = st.multiselect("Imported tables", imported, default=imported)
                extra = st.file_uploader("Add policy files", type=["pdf", "txt"], accept_multiple_files=True)
                if chosen:
                    bundle = _merge_documents(_bundle_from_disk(tuple(chosen)), extra or [])
            else:
                st.info("No imported tables yet.")

    with st.expander("Programme settings", expanded=False):
        defaults = (profile or {}).get("programme", {"retention": 500_000, "limit": 4_500_000, "capital_held": 60_000_000})
        suffix = (profile or {}).get("slug", "custom")
        retention = st.number_input("Retention ($)", min_value=0, value=int(defaults["retention"]), step=50_000, key=f"ret_{suffix}")
        limit = st.number_input("Layer limit ($)", min_value=100_000, value=int(defaults["limit"]), step=250_000, key=f"lim_{suffix}")
        capital_held = st.number_input("Capital held ($)", min_value=0, value=int(defaults["capital_held"]), step=500_000, key=f"cap_{suffix}")
    params = {"retention": float(retention), "limit": float(limit), "capital_held": float(capital_held)}

    with st.expander("Analysis controls", expanded=False):
        operating_mode = st.selectbox("Mode", ["Auto — bizmax Core", "Manual — specialist"])
        llm_router = st.toggle("LLM router", value=True, help="Turn off for instant keyword routing.")
        deep_synthesis = st.toggle("Deep final synthesis", value=False, help="Off displays completed specialist reports immediately.")
    manual_mode = operating_mode.startswith("Manual")
    sidebar_system_card(model=model_name(), traced=tracing_enabled())


business_name = (profile or {}).get("name", "Uploaded risk portfolio" if source == "Upload files" else "Supabase risk portfolio")
focus = (profile or {}).get("focus", "Analysis based on the currently loaded files.")
claim_count = len(bundle.claims) if bundle is not None and bundle.claims is not None else 0
workspace_header(page, business=business_name, model=model_name(), traced=tracing_enabled(), claim_count=claim_count)

if bundle is None:
    section_header("DATA INTAKE", "Connect a portfolio", "Load samples, multiple files, or Supabase tables from the sidebar.")
    st.info("The operating workspace will populate as soon as data is loaded.")
    st.stop()

has_claims = bundle.claims is not None and not bundle.claims.empty
E: dict[str, Any] | None = None
if has_claims:
    try:
        E = engine(bundle.claims, **params)
    except Exception as exc:
        st.error(f"The deterministic engine could not process this claims table: {exc}")

portfolio_bar(business=business_name, focus=focus, source_count=len(bundle.sources), claim_count=claim_count, mode="Manual" if manual_mode else "Auto / Core")

CORE_EXAMPLES = [
    "Give me a full risk assessment of this business.",
    "Are our reserves adequate and is our capital position solvent?",
    "Is our reinsurance programme right for the tail we carry, and what would it cost to change it?",
    "Where is claims leakage or fraud costing us, and how much?",
    "What happens to us in a catastrophe year, and does the policy actually pay?",
    "Which lines are deteriorating and what should we do about them this quarter?",
]


def run_core_console(*, expanded: bool) -> None:
    title = "Run another Core analysis" if st.session_state.get("core_result") else "Start multi-agent analysis"
    with st.expander(title, expanded=expanded):
        if "core_q" not in st.session_state:
            st.session_state["core_q"] = CORE_EXAMPLES[0]

        def choose_example() -> None:
            st.session_state["core_q"] = st.session_state["core_example"]

        st.selectbox("Example questions", CORE_EXAMPLES, key="core_example", on_change=choose_example)
        question = st.text_area("Ask bizmax Core", key="core_q", height=82)
        run = st.button("Run multi-agent analysis", type="primary", disabled=not (has_claims and api_key_available()), width="stretch")
        if not has_claims:
            st.warning("Core needs a claims table.")
        elif not api_key_available():
            st.warning("`NVIDIA_API_KEY` is not set in `.env`.")
        if not run:
            return
        panel = st.empty()
        session_id = new_session_id("core")

        def on_update(phase: str, states: dict[str, str]) -> None:
            panel.html(render_magi(states, phase, session_id, "AUTO", ""))

        try:
            meta = {"mode": "auto", "question": question[:200], "synthesis_mode": "deep" if deep_synthesis else "rapid"}
            with st.spinner("Routing and running the selected specialists in parallel…"):
                with analysis_run(session_id, meta):
                    result = run_core(question, bundle, params, session_id, on_update=on_update, use_llm_router=llm_router, deep_synthesis=deep_synthesis)
            st.session_state["core_result"] = result
            panel.html(render_magi(result.statuses(), "complete", session_id, "AUTO", result.route.objective))
            st.rerun()
        except Exception as exc:
            st.error(f"Core run failed: {exc}")


def render_manual_track(track_key: str) -> None:
    track = TRACKS[track_key]
    ready = has_claims if track.needs == "claims" else bool(bundle.documents)
    section_header("SPECIALIST WORKSPACE", track.name, track.blurb)
    if not ready:
        st.warning("This specialist needs claims data." if track.needs == "claims" else "This specialist needs a PDF or TXT policy document.")
        return
    if not api_key_available():
        st.warning("`NVIDIA_API_KEY` is not set in `.env`.")
        return
    question_key = f"question_{track_key}"
    if question_key not in st.session_state:
        st.session_state[question_key] = track.default_question

    def choose_example() -> None:
        st.session_state[question_key] = st.session_state[f"example_{track_key}"]

    st.selectbox("Example questions", track.example_questions, key=f"example_{track_key}", on_change=choose_example)
    question = st.text_area("Question", key=question_key, height=78)
    state_key = f"result_{track_key}"
    if st.button(f"Run {track.name}", type="primary", key=f"run_{track_key}"):
        session_id = new_session_id(track_key)
        try:
            with st.spinner(f"{track.name} is analysing the loaded evidence…"):
                tools = track.build_tools(bundle, params)
                meta = {"track": track_key, **track.metadata(bundle, params)}
                with analysis_run(session_id, meta):
                    text, calls = run_agent(track.system_prompt, question, tools, run_name=f"{track_key}_track")
            st.session_state[state_key] = {"text": strip_scratchpad(text), "calls": calls, "session": session_id, "meta": meta}
        except Exception as exc:
            st.error(f"{track.name} failed: {exc}")
    if state_key in st.session_state:
        result = st.session_state[state_key]
        st.info(md(result["text"]) or "_(No answer was returned.)_", icon="📐")
        calls = ", ".join(f"{call['name']} ({call.get('seconds', 0):.1f}s)" for call in result["calls"]) or "none"
        st.caption("Tools: " + calls + (f" · PRISM session `{result['session']}`" if tracing_enabled() else ""))
        with st.expander("Computed truth sent to PRISM"):
            st.json(result["meta"])


def render_core_decision(result: CoreResult) -> None:
    synthesis = result.synthesis
    rapid = synthesis.get("_mode") == "rapid" or synthesis.get("_parse") == "rapid"
    panel_header("Rapid decision brief" if rapid else "Executive synthesis", "Completed specialist findings grounded in Python output.", badge="RAPID" if rapid else "DEEP", tone="green", icon="✦")
    with st.container(border=True):
        st.markdown(md(synthesis.get("executive_summary", "")) or "_(No synthesis returned.)_")
        friction = synthesis.get("strategic_risk_friction", "")
        if friction and friction.lower() not in ("none", "none identified"):
            st.warning(f"**Strategic Risk Friction** — {md(friction)}")
        grounding = result.grounding()
        phase = "rapid assembly" if rapid else "deep synthesis"
        st.caption(
            f"{result.route.source} router → {', '.join(TRACKS[key].name for key in result.route.active)} · "
            f"routing {result.timings.get('routing', 0):.1f}s · specialists {result.timings.get('processing', 0):.1f}s parallel · "
            f"{phase} {result.timings.get('synthesizing', 0):.2f}s · grounding {grounding['rate']:.0f}%"
            + (f" · PRISM `{result.session_id}`" if tracing_enabled() else "")
        )


def render_dashboard() -> None:
    if E is None:
        st.warning("Dashboard metrics need a valid claims table.")
        return
    ibnr, rec, solv = E["ibnr"], E["rec"], E["solv"]
    metric_grid([
        {"label": "Reported loss", "value": money(ibnr["portfolio_loss"]), "note": f"Paid {money(ibnr['paid_to_date'])}", "tone": "red", "icon": "↘"},
        {"label": "IBNR reserve", "value": money(ibnr["ibnr"]), "note": f"Ultimate {money(ibnr['ultimate_loss'])}", "tone": "cyan", "icon": "Σ"},
        {"label": "Expected recovery", "value": money(rec["expected_recovery"]), "note": f"{rec['claims_in_layer']} claims entered layer", "tone": "green", "icon": "⬡"},
        {"label": "Net ultimate exposure", "value": money(ibnr["ultimate_loss"] - rec["expected_recovery"]), "note": f"SCR ratio {solv['scr_ratio_pct']:.0f}% · {solv['status']}", "tone": "amber", "icon": "◇"},
    ])
    result: CoreResult | None = st.session_state.get("core_result")
    if result is None:
        panel_header("Multi-agent Analysis", "Ask one question; the router selects the needed specialists.", badge="READY", tone="cyan", icon="✦")
        run_core_console(expanded=True)
    else:
        render_core_decision(result)
    left, right = st.columns([2.05, 1], gap="medium")
    with left:
        panel_header("Agent Network", "Six specialists coordinated by bizmax Core.", badge=f"{len(result.route.active) if result else 0} ACTIVE", tone="green", icon="⌁")
        agent_network(result)
    with right:
        stats = E["sim"]["net_of_reinsurance"]
        panel_header("Risk Metrics", f"Monte Carlo · {E['sim']['paths']:,} simulations", badge=E["sim"]["confidence_tier"].replace("_", " ").upper(), tone="red", icon="◇")
        risk_metric_list([
            ("Expected annual loss", money(stats["mean"]), "mean"),
            ("VaR (95%)", money(stats["var_95"]), "95%"),
            ("VaR (99%)", money(stats["var_99"]), "99%"),
            ("VaR (99.5%)", money(stats["var_99_5"]), "SCR basis"),
            ("TVaR (99%)", money(stats["tvar_99"]), "tail"),
        ])
    panel_header("Workflow Trace", "How this portfolio moves from data to decision.", badge="PRISM" if tracing_enabled() else "LOCAL", tone="cyan", icon="◷")
    workflow_trace(result, source_count=len(bundle.sources), claim_count=claim_count, traced=tracing_enabled())
    if result is not None:
        run_core_console(expanded=False)


def render_claims() -> None:
    if E is None:
        st.warning("Claims Analysis needs a valid claims table.")
        return
    df, intel, leakage = bundle.claims, E["intel"], E["leak"]
    metric_grid([
        {"label": "Total claims", "value": f"{intel['claim_count']:,}", "note": f"Years {min(intel['severity_by_year'])}–{max(intel['severity_by_year'])}", "tone": "cyan", "icon": "▥"},
        {"label": "Average severity", "value": money(intel["historical_mean_severity"]), "note": f"Latest {intel['severity_change_pct']:+.1f}% vs prior", "tone": "red", "icon": "↗"},
        {"label": "Anomalies", "value": f"{intel['anomaly_count']}", "note": f"Isolation Forest · {intel['anomaly_rate_pct']:.1f}% of book", "tone": "amber", "icon": "⚠"},
        {"label": "Claims leakage", "value": f"{leakage['leakage_rate_pct']:.2f}%", "note": f"{leakage['overpaid_claim_count']} claims · {money(leakage['leakage_total'])}", "tone": "green", "icon": "⌁"},
    ])
    left, right = st.columns(2, gap="medium")
    with left:
        panel_header("Claim Severity Trend", "Average severity by accident year.", badge=f"{intel['severity_change_pct']:+.1f}%", tone="red", icon="↗")
        show(F.severity_trend(intel))
    with right:
        panel_header("Frequency and Severity", "Claim counts and mean claim size by year.", badge=E["freq"]["distribution"].upper(), tone="cyan", icon="⌁")
        show(F.frequency_severity_by_year(E["freq"], df))
    left, right = st.columns(2, gap="medium")
    with left:
        panel_header("Payment Development", "Paid-to-reported ratio by development age.", tone="green", icon="◉")
        show(F.development_curve(E["devp"]))
    with right:
        panel_header("Claims by Line", "Loss by line of business and accident year.", tone="amber", icon="▥")
        show(F.loss_by_line(df))
    panel_header("Anomalous Claims Detected", "Isolation Forest · lowest anomaly score first.", badge=f"{intel['anomaly_count']} FLAGGED", tone="amber", icon="⚠")
    anomalies = anomaly_table(df, min(20, max(10, intel["anomaly_count"])))
    sheet(anomalies, "anomalous_claims", ["claim_amount", "reported_amount", "paid_amount", "reserve"], 365)
    panel_header("Two-model Intelligence", "The only trained ML models in this project.", badge="2 MODELS", tone="green", icon="ML")
    metric_grid([
        {"label": "Severity model", "value": "Random Forest", "note": f"Predicted mean {money(intel['predicted_mean_severity'])}", "tone": "green", "icon": "RF"},
        {"label": "Anomaly model", "value": "Isolation Forest", "note": f"{intel['anomaly_count']} claims · bottom 5% cut", "tone": "amber", "icon": "IF"},
        {"label": "Litigation", "value": f"{leakage['litigation_rate_pct']:.1f}%", "note": f"Severity multiple {leakage['litigation_severity_multiple']:.2f}×", "tone": "red", "icon": "§"},
        {"label": "Social inflation", "value": f"{leakage['social_inflation_pct_per_year']:+.1f}%", "note": f"Less {leakage['assumed_cpi_pct']:.1f}% assumed CPI", "tone": "cyan", "icon": "Δ"},
    ])
    with st.expander("Ask the AI Claims Analyst", expanded=manual_mode):
        render_manual_track("claims")


def render_policy() -> None:
    if not bundle.documents:
        st.warning("Policy Intelligence needs at least one PDF or TXT policy document.")
        st.caption("Use Portfolio & data in the sidebar to add wordings.")
        return
    clauses = wording.extract_clauses(bundle.documents)
    total_clauses = sum(clauses["counts"].values())
    panel_header("Policy Document Intelligence", "Local regex and keyword retrieval: document → sentence → clause.", badge=f"{total_clauses} CLAUSES", tone="cyan", icon="▤")
    with st.container(border=True):
        search = st.text_input("Search policy wording", placeholder="coverage, limit, pollution, waiting period…")
        categories = ["all", *clauses["clauses"]]
        category = st.radio("Clause category", categories, horizontal=True, format_func=lambda value: value.replace("_", " ").title())
    rows: list[dict[str, Any]] = []
    if search.strip():
        result = wording.search_wording(bundle.documents, search)
        rows = [{"category": "keyword result", "document": hit["document"], "text": hit["text"], "amounts": re.findall(wording.MONEY, hit["text"], re.I)[:4]} for hit in result["hits"]]
    else:
        chosen = clauses["clauses"] if category == "all" else {category: clauses["clauses"][category]}
        rows = [{"category": cat, "document": hit["document"], "text": hit["text"], "amounts": hit["amounts"]} for cat, hits in chosen.items() for hit in hits]
    if not rows:
        st.info("No sentence matched this search and category.")
    else:
        labels = [f"{row['category'].replace('_', ' ').title()} · {row['document']} · {row['text'][:64]}…" for row in rows]
        selected_label = st.selectbox("Retrieved evidence", labels, label_visibility="collapsed")
        selected_index = labels.index(selected_label)
        selected = rows[selected_index]
        left, right = st.columns([1, 1.52], gap="medium")
        with left:
            st.caption(f"RETRIEVED CLAUSES · {len(rows)} RESULT{'S' if len(rows) != 1 else ''}")
            visible = [selected_index] + [index for index in range(len(rows)) if index != selected_index]
            for index in visible[:6]:
                row = rows[index]
                clause_card(row["category"], row["document"], row["text"], row["amounts"], selected=index == selected_index)
        with right:
            panel_header("Extracted Evidence", "Exact source sentence used by the Policy Analyst.", badge="RETRIEVED", tone="green", icon="§")
            evidence_detail(selected["category"], selected["document"], selected["text"], selected["amounts"])
            if E is not None:
                modeled = {"severity_p99": E["sev"]["p99"], "largest_claim": E["sev"]["max"], "expected_annual_loss": E["sim"]["gross"]["mean"], "var_99_5": E["sim"]["gross"]["var_99_5"]}
                gaps = wording.coverage_gaps(bundle.documents, modeled)
                with st.expander(f"Coverage gap checks · {gaps['gap_count']}"):
                    if gaps["gaps"]:
                        st.dataframe(pd.DataFrame(gaps["gaps"]), hide_index=True, width="stretch")
                    else:
                        st.success("No modeled limit or exclusion gap was identified by the deterministic checks.")
                    st.caption(gaps["method"])
    with st.expander("Ask the AI Policy Analyst", expanded=manual_mode):
        render_manual_track("policy")


def render_reinsurance() -> None:
    if E is None:
        st.warning("Reinsurance needs a valid claims table.")
        return
    rec = E["rec"]
    recovery_pct = 100 * rec["expected_recovery"] / rec["gross_loss"] if rec["gross_loss"] else 0
    metric_grid([
        {"label": "Gross loss", "value": money(rec["gross_loss"]), "note": f"Largest claim {money(rec['largest_gross_claim'])}", "tone": "red", "icon": "↘"},
        {"label": "Recovery", "value": money(rec["expected_recovery"]), "note": f"{recovery_pct:.1f}% of observed gross loss", "tone": "green", "icon": "⬡"},
        {"label": "Net loss", "value": money(rec["net_loss"]), "note": f"Retention {money(rec['retention'])}", "tone": "amber", "icon": "◇"},
        {"label": "Tail above layer", "value": money(rec["uncovered_above_layer"]), "note": f"{rec['claims_exhausting_layer']} claims exhausted", "tone": "cyan", "icon": "▤"},
    ])
    panel_header("Reinsurance Treaty Structure", rec["method"], badge="ACTIVE", tone="cyan", icon="⬡")
    reinsurance_layers(rec, money=money)
    left, right = st.columns(2, gap="medium")
    with left:
        panel_header("Observed Loss Flow", "Gross, retained, ceded and uncovered portions.", tone="red", icon="↘")
        show(F.reinsurance_flow(rec))
    with right:
        panel_header("Retention Economics", "Total Cost of Risk across attachment points.", badge=f"OPT {money(E['sweep']['optimal_retention'])}", tone="green", icon="Δ")
        show(F.tcor_curve(E["sweep"], params["retention"]))
    left, right = st.columns(2, gap="medium")
    with left:
        panel_header("Programme Structures", "XoL, aggregate stop-loss and quota share.", tone="amber", icon="▥")
        show(F.structures_bar(E["cmp"], E["qs"]))
    with right:
        panel_header("Recovery Calculation", "Observed claims under the selected layer.", badge="PYTHON", tone="green", icon="Σ")
        risk_metric_list([
            ("Gross observed loss", money(rec["gross_loss"]), "input"),
            ("Retention", money(rec["retention"]), "per claim"),
            ("Layer limit", money(rec["limit"]), "per claim"),
            ("Expected recovery", money(rec["expected_recovery"]), "ceded"),
            ("Net observed loss", money(rec["net_loss"]), "retained"),
        ])
    with st.expander("Retention sweep data"):
        sheet(pd.DataFrame(E["sweep"]["sweep"]), "retention_sweep", ["retention", "limit", "reinsurance_premium", "expected_retained_loss", "capital_charge", "tcor", "expected_recovery"], 300)
    with st.expander("Ask the AI Reinsurance Manager", expanded=manual_mode):
        render_manual_track("reinsurance")


def render_actuarial_engine() -> None:
    if E is None:
        st.warning("The Actuarial Engine needs a valid claims table.")
        return
    stats = E["sim"]["net_of_reinsurance"]
    metric_grid([
        {"label": "Mean annual loss", "value": money(stats["mean"]), "note": f"{E['sim']['paths']:,} seeded simulations", "tone": "green", "icon": "μ"},
        {"label": "VaR 95%", "value": money(stats["var_95"]), "note": "Net of current reinsurance", "tone": "cyan", "icon": "95"},
        {"label": "VaR 99.5%", "value": money(stats["var_99_5"]), "note": "Solvency capital basis", "tone": "amber", "icon": "99"},
        {"label": "TVaR 99%", "value": money(stats["tvar_99"]), "note": "Mean beyond the 99th percentile", "tone": "red", "icon": "T"},
    ])
    left, right = st.columns([2.05, 1], gap="medium")
    with left:
        panel_header("Monte Carlo Simulation", "Poisson frequency × lognormal severity × per-claim XoL.", badge=f"{E['sim']['paths']:,} SIMS", tone="cyan", icon="∷")
        show(F.aggregate_histogram(E["sim"], params["capital_held"]))
    with right:
        panel_header("Risk Measures", "From deterministic simulation output.", badge=E["sim"]["confidence_tier"].replace("_", " ").upper(), tone="red", icon="↗")
        risk_metric_list([
            ("Mean", money(stats["mean"]), "50% centre"),
            ("VaR (95%)", money(stats["var_95"]), "95%"),
            ("VaR (99%)", money(stats["var_99"]), "99%"),
            ("VaR (99.5%)", money(stats["var_99_5"]), "SCR"),
            ("TVaR (99%)", money(stats["tvar_99"]), "tail"),
        ])
    left, right = st.columns([2.05, 1], gap="medium")
    with left:
        panel_header("Loss Development Triangle", "Cumulative incurred loss by accident year and development age.", badge="CHAIN LADDER", tone="green", icon="Σ")
        show(F.loss_triangle_heatmap(E["tri"]))
    with right:
        panel_header("Age-to-Ultimate Factors", "Volume-weighted development pattern.", badge="CDF", tone="amber", icon="↘")
        factors = [(age, f"{factor:.4f}", f"to {age.split('-')[-1]}m") for age, factor in E["tri"]["age_to_age_factors"].items()]
        risk_metric_list(factors)
    with st.expander("Loss triangle values"):
        triangle = pd.DataFrame(E["tri"]["triangle"]).T.reset_index(names="accident_year")
        sheet(triangle, "loss_triangle", [column for column in triangle.columns if column != "accident_year"], 340)
    panel_header("Model Components", "Frequency, severity, reserving and solvency assumptions.", badge="DETERMINISTIC", tone="green", icon="∑")
    metric_grid([
        {"label": "Frequency", "value": f"λ {E['freq']['poisson_lambda']:.2f}", "note": f"{E['freq']['distribution']} · trend {E['freq']['annual_trend_pct']:+.1f}%/yr", "tone": "cyan", "icon": "λ"},
        {"label": "Severity", "value": money(E["sev"]["lognormal_mean"]), "note": f"σ {E['sev']['lognormal_sigma']:.2f} · {E['sev']['tail']} tail", "tone": "red", "icon": "Σ"},
        {"label": "Chain-ladder IBNR", "value": money(E["tri"]["total_chain_ladder_ibnr"]), "note": "Volume-weighted age-to-age factors", "tone": "green", "icon": "CL"},
        {"label": "SCR ratio", "value": f"{E['solv']['scr_ratio_pct']:.0f}%", "note": f"{E['solv']['status']} · own funds {money(E['solv']['own_funds'])}", "tone": "amber", "icon": "S"},
    ])
    left, right = st.columns(2, gap="medium")
    with left:
        panel_header("Reserve Development", "LDF IBNR and chain-ladder ultimate by year.", tone="green", icon="Σ")
        show(F.ibnr_by_year(E["ibnr"], E["tri"]))
    with right:
        panel_header("Stress Scenarios", "Frequency and severity multipliers applied in Python.", tone="red", icon="⚠")
        show(F.stress_bars(E["stress"]))
    with st.expander("Ask an actuarial, capital or risk specialist", expanded=manual_mode):
        specialist = st.selectbox("Specialist", ["actuary", "capital", "risk"], format_func=lambda key: TRACKS[key].name, key="actuarial_specialist")
        render_manual_track(specialist)


def render_audit() -> None:
    result: CoreResult | None = st.session_state.get("core_result")
    if result is None:
        metric_grid([
            {"label": "Core runs", "value": "0", "note": "Current browser session", "tone": "cyan", "icon": "◷"},
            {"label": "PRISM", "value": "Connected" if tracing_enabled() else "Offline", "note": "Live model observability", "tone": "green" if tracing_enabled() else "amber", "icon": "P"},
            {"label": "Claims loaded", "value": f"{claim_count:,}", "note": f"{len(bundle.sources)} source(s)", "tone": "cyan", "icon": "▥"},
            {"label": "Approval", "value": "Human required", "note": "The agent never binds cover", "tone": "amber", "icon": "✓"},
        ])
        panel_header("Workflow Trace", "No Core analysis has run in this browser session.", badge="READY", tone="cyan", icon="◷")
        workflow_trace(None, source_count=len(bundle.sources), claim_count=claim_count, traced=tracing_enabled())
        st.info("Run a question from Dashboard to populate the detailed audit trail.")
        return
    grounding = result.grounding()
    metric_grid([
        {"label": "Run status", "value": "Complete", "note": result.session_id, "tone": "green", "icon": "✓"},
        {"label": "Specialists", "value": f"{len(result.reports)}", "note": f"{sum(len(r.tool_calls) for r in result.reports.values())} tool calls", "tone": "cyan", "icon": "⌁"},
        {"label": "Grounding", "value": f"{grounding['rate']:.0f}%", "note": f"{grounding['checked']} figures checked", "tone": "green" if grounding["grounded"] else "amber", "icon": "G"},
        {"label": "Elapsed", "value": f"{sum(result.timings.values()):.1f}s", "note": f"Router {result.route.source} · {result.synthesis.get('_mode', 'deep')}", "tone": "amber", "icon": "◷"},
    ])
    panel_header("Current Model Run", result.question, badge="PRISM TRACED" if tracing_enabled() else "LOCAL", tone="green", icon="◷")
    agent_rows = []
    for key in result.route.active:
        report = result.reports.get(key)
        if report:
            agent_rows.append({"specialist": report.name, "status": "error" if report.error else "complete", "seconds": round(report.seconds, 2), "tool_calls": len(report.tool_calls), "tools": ", ".join(report.tool_names) or "none", "error": report.error})
    st.dataframe(pd.DataFrame(agent_rows), hide_index=True, width="stretch")
    panel_header("Detailed Trace", "Actual run stages and measured latency.", badge=result.session_id, tone="cyan", icon="▤")
    workflow_trace(result, source_count=len(bundle.sources), claim_count=claim_count, traced=tracing_enabled())
    panel_header("Specialist Evidence", "Model reports and deterministic tool results.", badge=f"{len(result.reports)} REPORTS", tone="green", icon="§")
    for key in result.route.active:
        report = result.reports.get(key)
        if not report:
            continue
        with st.expander(f"{report.name} · {report.seconds:.1f}s · {len(report.tool_calls)} tools" + (f" · {report.error}" if report.error else "")):
            st.markdown(md(report.text) or f"_{report.error or 'No output'}_")
            if report.tool_calls:
                st.dataframe(pd.DataFrame([{"tool": call["name"], "seconds": round(call.get("seconds", 0), 3), "arguments": str(call.get("args", {})), "result_preview": str(call.get("result", ""))[:240]} for call in report.tool_calls]), hide_index=True, width="stretch")
    with st.expander("Computed truth sent to PRISM"):
        st.json(result.metadata)
    with st.expander("Human review gate"):
        st.warning("Review the evidence before changing reserves, capital, policy wording, or reinsurance. This run is advisory and does not bind cover.")


if page == "Dashboard":
    render_dashboard()
elif page == "Claims Analysis":
    render_claims()
elif page == "Policy Intelligence":
    render_policy()
elif page == "Reinsurance":
    render_reinsurance()
elif page == "Actuarial Engine":
    render_actuarial_engine()
else:
    render_audit()
