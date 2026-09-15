"""bizmax — AI Actuary dashboard. Auto mode = bizmax Core (router -> parallel agents -> synthesis);
Manual mode = one track at a time.

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
import streamlit.components.v1 as components
from dotenv import load_dotenv

from src.actuarial import capital as cap
from src.actuarial import development as dev
from src.actuarial import exposure as expo
from src.actuarial import tcor as tc
from src.actuarial.frequency_severity import fit_frequency, fit_severity, stress_test
from src.actuarial.ibnr import calculate_ibnr
from src.actuarial.reinsurance import calculate_reinsurance_recovery
from src.charts import figures as F
from src.core.orchestrator import CoreResult, core_metadata, grounding_check, run_core
from src.data.ingest import DataBundle, build_bundle, bundle_from_paths
from src.data.samples import list_samples, sample_files
from src.llm.agent import api_key_available, run_agent
from src.llm.config import model_name
from src.ml.predict import anomaly_table, get_claims_intelligence
from src.policy import wording
from src.prism.tracing import analysis_run, new_session_id, tracing_enabled
from src.tracks.registry import TRACKS
from src.ui.magi import render_magi

load_dotenv()

IMPORT_DIR = Path("data/imported")
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
    return money(x).replace("$", "\\$")


def md(text: str) -> str:
    """LLM markdown for st.markdown/st.info: escape $ so Streamlit does not read $...$ as LaTeX."""
    return (text or "").replace("$", "\\$")


def show(fig) -> None:
    st.pyplot(fig, width="stretch"); plt.close(fig)


def sheet(df: pd.DataFrame, name: str, money_cols: list[str] | None = None, height: int = 260) -> None:
    fmt = {c: "${:,.0f}" for c in (money_cols or []) if c in df.columns}
    st.dataframe(df.style.format(fmt) if fmt else df, hide_index=True, height=height, width="stretch")
    st.download_button(f"Download {name}.csv", df.to_csv(index=False).encode(), f"{name}.csv", "text/csv", key=f"dl_{name}")


@st.cache_data(show_spinner=False)
def _bundle_from_uploads(files: tuple[tuple[str, bytes], ...]) -> DataBundle:
    return build_bundle(list(files))


@st.cache_data(show_spinner=False)
def _bundle_from_disk(paths: tuple[str, ...]) -> DataBundle:
    return bundle_from_paths(list(paths))


@st.cache_data(show_spinner=False)
def engine(df: pd.DataFrame, retention: float, limit: float, capital_held: float) -> dict[str, Any]:
    """Everything deterministic the dashboard shows, computed once per (data, programme)."""
    return {
        "ibnr": calculate_ibnr(df), "rec": calculate_reinsurance_recovery(df, retention, limit), "intel": get_claims_intelligence(df),
        "freq": fit_frequency(df), "sev": fit_severity(df), "tri": dev.loss_triangle(df), "stress": stress_test(df),
        "sim": cap.simulate_aggregate(df, retention, limit), "solv": cap.solvency_position(df, capital_held, retention, limit),
        "adq": cap.capital_adequacy(df, capital_held, retention, limit), "expo": expo.exposure_movement(df),
        "conc": expo.concentration(df), "emerg": expo.emerging_signals(df), "sweep": tc.retention_sweep(df, limit),
        "cmp": tc.compare_structures(df, retention, limit), "qs": tc.quota_share_vs_xol(df, retention, limit),
        "devp": dev.development_pattern(df), "adv": dev.adverse_development(df), "leak": dev.claims_leakage(df),
    }


# ---------------------------------------------------------------- sidebar
samples = list_samples()
with st.sidebar:
    st.markdown("### Mode")
    mode = st.radio("Mode", ["Auto — bizmax Core", "Manual — single track"], label_visibility="collapsed")
    auto = mode.startswith("Auto")

    st.markdown("### Data")
    source = st.radio("Source", ["Sample business", "Upload files", "Supabase import"], label_visibility="collapsed")
    bundle: DataBundle | None = None
    prof: dict[str, Any] | None = None
    if source == "Sample business":
        slug = st.selectbox("Business", list(samples), format_func=lambda s: samples[s]["name"])
        prof = samples[slug]
        st.caption(f"{prof['tagline']}  \n**Focus:** {prof['focus']}")
        bundle = _bundle_from_disk(tuple(sample_files(slug)))
    elif source == "Upload files":
        ups = st.file_uploader("Claims / policies / treaties (CSV, XLSX) and wordings (PDF, TXT)",
                               type=["csv", "xlsx", "xls", "pdf", "txt"], accept_multiple_files=True)
        if ups:
            bundle = _bundle_from_uploads(tuple((u.name, u.getvalue()) for u in ups))
    else:
        st.caption("Reads tables via the Supabase REST API into `data/imported/*.csv`.")
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
            extra = st.file_uploader("Add policy PDFs (optional)", type=["pdf", "txt"], accept_multiple_files=True)
            if chosen:
                bundle = _bundle_from_disk(tuple(chosen))
                if extra:
                    docs = build_bundle([(u.name, u.getvalue()) for u in extra])
                    bundle.documents.update(docs.documents); bundle.sources += docs.sources
        else:
            st.info("No imported tables yet.")

    st.markdown("### Programme")
    d = (prof or {}).get("programme", {"retention": 500_000, "limit": 4_500_000, "capital_held": 60_000_000})
    key_suffix = (prof or {}).get("slug", "custom")
    retention = st.number_input("Retention ($)", min_value=0, value=int(d["retention"]), step=50_000, format="%d", key=f"ret_{key_suffix}")
    limit = st.number_input("Layer limit ($)", min_value=100_000, value=int(d["limit"]), step=250_000, format="%d", key=f"lim_{key_suffix}")
    capital_held = st.number_input("Capital held ($)", min_value=0, value=int(d["capital_held"]), step=500_000, format="%d", key=f"cap_{key_suffix}")
    params = {"retention": float(retention), "limit": float(limit), "capital_held": float(capital_held)}

    st.markdown("---")
    st.caption(f"LLM: `{model_name()}` via NVIDIA NIM")
    st.caption("Math: pure Python · ML: scikit-learn (2 models)")
    st.caption("PRISM tracing: " + ("on" if tracing_enabled() else "off"))


# ---------------------------------------------------------------- header
st.title("bizmax · AI Actuary")
st.caption("AI-native insurance optimization for corporate risk · all figures illustrative, modeled estimates")
if bundle is None:
    st.info("Pick a sample business, upload files, or import from Supabase in the sidebar."); st.stop()
has_claims = bundle.claims is not None and not bundle.claims.empty
E = engine(bundle.claims, **params) if has_claims else None
with st.expander(f"Loaded data — {len(bundle.sources)} source(s)"):
    st.write(bundle.profile())


# ================================================================ AUTO MODE
def render_core_charts(res: CoreResult) -> None:
    """Charts for whichever agents the router activated — all from the engine."""
    if E is None:
        return
    active = set(res.route.active)
    if "actuary" in active:
        st.markdown("#### Actuary — reserving & pricing")
        c1, c2 = st.columns(2)
        with c1: show(F.ibnr_by_year(E["ibnr"], E["tri"]))
        with c2: show(F.frequency_severity_by_year(E["freq"], bundle.claims))
        show(F.loss_triangle_heatmap(E["tri"]))
    if "claims" in active:
        st.markdown("#### Claims Analyst — patterns, leakage, litigation")
        c1, c2 = st.columns(2)
        with c1: show(F.severity_trend(E["intel"]))
        with c2: show(F.development_curve(E["devp"]))
        show(F.leakage_bars(E["leak"]))
    if "reinsurance" in active:
        st.markdown("#### Reinsurance Manager — flow, retention, structures")
        show(F.reinsurance_flow(E["rec"]))
        c1, c2 = st.columns(2)
        with c1: show(F.tcor_curve(E["sweep"], params["retention"]))
        with c2: show(F.structures_bar(E["cmp"], E["qs"]))
    if "capital" in active:
        st.markdown("#### Capital Manager — solvency")
        c1, c2 = st.columns([1, 2])
        with c1: show(F.solvency_gauge(E["solv"]))
        with c2: show(F.aggregate_histogram(E["sim"], params["capital_held"]))
    if "risk" in active:
        st.markdown("#### Risk Manager — exposure & emerging signals")
        c1, c2 = st.columns(2)
        with c1: show(F.exposure_vs_loss(E["expo"]))
        with c2: show(F.emerging_signals_bars(E["emerg"]))
        show(F.stress_bars(E["stress"]))
    if "policy" in active and bundle.documents:
        st.markdown("#### Policy Analyst — wording")
        cl = wording.extract_clauses(bundle.documents)
        c1, c2 = st.columns([1, 2])
        with c1: show(F.clause_counts(cl))
        with c2:
            rows = [{"category": c, "document": h["document"], "clause": h["text"][:160], "amounts": ", ".join(h["amounts"])} for c, hs in cl["clauses"].items() for h in hs[:3]]
            st.dataframe(pd.DataFrame(rows), hide_index=True, height=260, width="stretch")


if auto:
    st.subheader("bizmax Core")
    st.caption("Ask anything about this business. The Core routes the question, runs the relevant agents in parallel, and synthesizes one answer. Every number comes from a Python tool.")
    examples = [
        "Give me a full risk assessment of this business.",
        "Are our reserves adequate and is our capital position solvent?",
        "Is our reinsurance programme right for the tail we carry, and what would it cost to change it?",
        "Where is claims leakage or fraud costing us, and how much?",
        "What happens to us in a catastrophe year, and does the policy actually pay?",
        "Which lines are deteriorating and what should we do about them this quarter?",
    ]
    if "core_q" not in st.session_state:
        st.session_state["core_q"] = examples[0]
    st.selectbox("Example questions", examples, key="core_pick", on_change=lambda: st.session_state.update(core_q=st.session_state["core_pick"]))
    question = st.text_area("Question", key="core_q", height=70)
    c1, c2 = st.columns([1, 3])
    with c1:
        run = st.button("Run bizmax Core", type="primary", disabled=not (has_claims and api_key_available()))
    with c2:
        llm_router = st.toggle("LLM router (off = keyword router, faster)", value=True)
    if not has_claims:
        st.warning("Needs claims data.")
    elif not api_key_available():
        st.warning("`NVIDIA_API_KEY` is not set in `.env`.")

    panel = st.empty()
    statuses = {k: "idle" for k in TRACKS}
    prev: CoreResult | None = st.session_state.get("core_result")
    if prev is not None and not run:
        statuses = {k: ("done" if k in prev.reports and prev.reports[k].text else "error" if k in prev.reports else "skipped") for k in TRACKS}
        panel.html(render_magi(statuses, "complete", prev.session_id, "AUTO", prev.route.objective))
    else:
        panel.html(render_magi(statuses, "idle", "", "AUTO"))

    if run:
        session_id = new_session_id("core")

        def on_update(phase: str, sts: dict[str, str]) -> None:
            panel.html(render_magi(sts, phase, session_id, "AUTO", ""))

        with st.spinner("bizmax Core is working…"):
            try:
                # route first (cheap) to know which tracks' computed_* to attach, then run inside the traced session
                meta = {"mode": "auto", "question": question[:200]}
                with analysis_run(session_id, meta):
                    res = run_core(question, bundle, params, session_id, on_update=on_update, use_llm_router=llm_router)
                res.metadata = core_metadata(bundle, params, res.route.active)
                st.session_state["core_result"] = res
                prev = res
                panel.html(render_magi({k: ("done" if k in res.reports and res.reports[k].text else "error" if k in res.reports else "skipped") for k in TRACKS},
                                       "complete", session_id, "AUTO", res.route.objective))
            except Exception as e:
                st.error(f"Core run failed: {e}")

    if prev is not None:
        res = prev
        syn = res.synthesis
        st.markdown("### Executive summary")
        st.markdown(md(syn.get("executive_summary", "")) or "_(empty)_")
        friction = syn.get("strategic_risk_friction", "")
        if friction and friction.lower() not in ("none identified", "none", ""):
            st.warning(f"**Strategic Risk Friction** — {md(friction)}")
        g = grounding_check(syn.get("executive_summary", ""), res.reports)
        rt = res.route
        st.caption(f"Routed by {rt.source} → {', '.join(TRACKS[k].name for k in rt.active)} · "
                   f"routing {res.timings.get('routing', 0):.0f}s · agents {res.timings.get('processing', 0):.0f}s (parallel) · synthesis {res.timings.get('synthesizing', 0):.0f}s · "
                   f"grounding {'OK' if g['grounded'] else 'CHECK ' + str(g['ungrounded'][:5])} ({g['numbers']} numbers)"
                   + (f" · PRISM session `{res.session_id}`" if tracing_enabled() else ""))
        if syn.get("key_figures"):
            st.dataframe(pd.DataFrame([{"figure": k, "value": v} for k, v in syn["key_figures"].items()]), hide_index=True, width="stretch")

        st.markdown("### Evidence")
        render_core_charts(res)

        st.markdown("### Sub-agent reports")
        for k in rt.active:
            r = res.reports.get(k)
            if r is None:
                continue
            with st.expander(f"{r.name} — {r.seconds:.0f}s · tools: {', '.join(r.tool_calls) or 'none'}" + (" · ERROR" if r.error else "")):
                st.markdown(md(r.text) or f"_{r.error or 'no output'}_")
        with st.expander("What PRISM received as truth (computed_* metadata)"):
            st.json(res.metadata)

# ================================================================ MANUAL MODE
else:
    tab_names = ["Overview"] + [t.name for t in TRACKS.values()]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        if E is None:
            st.warning("No claims table loaded — the overview needs claims data.")
        else:
            df = bundle.claims
            ibnr, rec, intel, solv = E["ibnr"], E["rec"], E["intel"], E["solv"]
            st.subheader("Portfolio overview")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Portfolio loss (reported)", money(ibnr["portfolio_loss"]))
            c2.metric("IBNR (LDF)", money(ibnr["ibnr"]), help=f"chain-ladder triangle IBNR {money(E['tri']['total_chain_ladder_ibnr'])}")
            c3.metric("Expected recovery", money(rec["expected_recovery"]), help=rec["method"])
            c4.metric("Net exposure", money(ibnr["ultimate_loss"] - rec["expected_recovery"]))
            c5.metric("SCR ratio", f"{solv['scr_ratio_pct']:.0f}%", delta=solv["status"], delta_color="normal" if solv["status"] in ("adequate", "strong") else "inverse")
            st.caption(f"{intel['claim_count']:,} claims · accident years {min(intel['severity_by_year'])}–{max(intel['severity_by_year'])} · "
                       f"λ {E['freq']['poisson_lambda']}/yr ({E['freq']['distribution']}) · severity σ {E['sev']['lognormal_sigma']} ({E['sev']['tail']} tail) · "
                       f"paid to date {money_md(ibnr['paid_to_date'])} · case reserves {money_md(ibnr['case_reserves'])}")

            st.subheader("Claims intelligence")
            sev = intel["severity_change_pct"]
            w1, w2, w3, w4 = st.columns(4)
            w1.metric(f"{'⚠️ ' if abs(sev) >= 10 else ''}Severity trend ({intel['latest_year']})", f"{sev:+.1f}%", delta=f"{money_md(intel['latest_year_mean_severity'])} vs {money_md(intel['prior_years_mean_severity'])}", delta_color="inverse")
            w2.metric("⚠️ Anomalous claims" if intel["anomaly_count"] else "Anomalous claims", f"{intel['anomaly_count']}", delta=f"{intel['anomaly_rate_pct']:.1f}% · {money_md(intel['anomalous_amount_total'])}", delta_color="off")
            w3.metric("Claims leakage", f"{E['leak']['leakage_rate_pct']}%", delta=f"{E['leak']['overpaid_claim_count']} overpaid · {money_md(E['leak']['leakage_total'])}", delta_color="off")
            w4.metric("Litigation rate", f"{E['leak']['litigation_rate_pct']}%", delta=f"severity x{E['leak']['litigation_severity_multiple']} · social inflation {E['leak']['social_inflation_pct_per_year']:+.1f}%/yr", delta_color="off")

            g1, g2, g3, g4, g5, g6 = st.tabs(["Reserving", "Claims", "Reinsurance", "Capital & risk", "Exposure", "Data sheets"])
            with g1:
                c1, c2 = st.columns(2)
                with c1: show(F.ibnr_by_year(ibnr, E["tri"]))
                with c2: show(F.frequency_severity_by_year(E["freq"], df))
                show(F.loss_triangle_heatmap(E["tri"]))
                sheet(pd.DataFrame(E["tri"]["by_accident_year"]), "chain_ladder_by_year", ["reported_to_date", "chain_ladder_ultimate", "ibnr"], 220)
            with g2:
                c1, c2 = st.columns(2)
                with c1: show(F.severity_trend(intel))
                with c2: show(F.development_curve(E["devp"]))
                c1, c2 = st.columns(2)
                with c1: show(F.leakage_bars(E["leak"]))
                with c2: show(F.loss_by_line(df))
                st.markdown("**Top anomalous claims** (Isolation Forest)")
                sheet(anomaly_table(df, 12), "anomalous_claims", ["claim_amount", "reported_amount", "paid_amount", "reserve"])
            with g3:
                show(F.reinsurance_flow(rec))
                c1, c2 = st.columns(2)
                with c1: show(F.tcor_curve(E["sweep"], params["retention"]))
                with c2: show(F.structures_bar(E["cmp"], E["qs"]))
                sheet(pd.DataFrame(E["sweep"]["sweep"]), "retention_sweep", ["retention", "limit", "reinsurance_premium", "expected_retained_loss", "retained_var_99_5", "capital_charge", "tcor", "expected_recovery"], 240)
            with g4:
                c1, c2 = st.columns([1, 2])
                with c1: show(F.solvency_gauge(solv))
                with c2: show(F.aggregate_histogram(E["sim"], params["capital_held"]))
                show(F.stress_bars(E["stress"]))
                a = E["adq"]
                st.caption(f"Shortfall probability {a['shortfall_probability_pct']}% · capital required at 99.5% {money_md(a['capital_required_99_5'])} · surplus/deficit {money_md(a['capital_surplus_or_deficit'])}")
            with g5:
                c1, c2 = st.columns(2)
                with c1: show(F.exposure_vs_loss(E["expo"]))
                with c2: show(F.emerging_signals_bars(E["emerg"]))
                sheet(pd.DataFrame(E["expo"]["by_line"]), "exposure_by_line", ["exposure", "loss"], 200)
            with g6:
                st.markdown("**Claims (filterable)**")
                lobs = sorted(df["line_of_business"].unique())
                f1, f2 = st.columns(2)
                pick_lob = f1.multiselect("Line of business", lobs, default=lobs)
                yrs = sorted(df["accident_year"].unique())
                pick_yr = f2.slider("Accident year", int(min(yrs)), int(max(yrs)), (int(min(yrs)), int(max(yrs))))
                view = df[df["line_of_business"].isin(pick_lob) & df["accident_year"].between(*pick_yr)]
                sheet(view, "claims_filtered", ["claim_amount", "reported_amount", "paid_amount", "reserve", "exposure"], 380)
                st.markdown("**IBNR by accident year (LDF)**")
                sheet(pd.DataFrame(ibnr["by_accident_year"]), "ibnr_by_year", ["reported", "ultimate", "ibnr"], 220)
                if bundle.documents:
                    st.markdown("**Policy clauses**")
                    cl = wording.extract_clauses(bundle.documents)
                    rows = [{"category": c, "document": h["document"], "clause": h["text"], "amounts": ", ".join(h["amounts"])} for c, hs in cl["clauses"].items() for h in hs]
                    sheet(pd.DataFrame(rows), "policy_clauses", None, 260)

    def render_track(track_key: str) -> None:
        track = TRACKS[track_key]
        st.markdown(f"**{track.name}** — {track.blurb}")
        ready = has_claims if track.needs == "claims" else bool(bundle.documents)
        if not ready:
            st.warning("Needs claims data." if track.needs == "claims" else "Needs at least one policy document (PDF/TXT)."); return
        if not api_key_available():
            st.warning("`NVIDIA_API_KEY` is not set in `.env`."); return
        q_key = f"q_{track_key}"
        if q_key not in st.session_state:
            st.session_state[q_key] = track.default_question

        def _pick(k: str = track_key) -> None:
            st.session_state[f"q_{k}"] = st.session_state[f"pick_{k}"]

        st.selectbox("Example questions", track.example_questions, key=f"pick_{track_key}", on_change=_pick)
        question = st.text_area("Question", key=q_key, height=70)
        state_key = f"result_{track_key}"
        if st.button(f"Run {track.name}", type="primary", key=f"run_{track_key}"):
            session_id = new_session_id(track_key)
            with st.spinner(f"{track.name} is working ({model_name()})…"):
                try:
                    tools = track.build_tools(bundle, params)
                    meta = {"track": track_key, **track.metadata(bundle, params)}
                    with analysis_run(session_id, meta):
                        text, calls = run_agent(track.system_prompt, question, tools, run_name=f"{track_key}_track")
                    from src.core.synthesizer import strip_scratchpad
                    st.session_state[state_key] = {"text": strip_scratchpad(text), "calls": calls, "session": session_id, "meta": meta}
                except Exception as e:
                    st.error(f"{track.name} failed: {e}")
        if state_key in st.session_state:
            r = st.session_state[state_key]
            st.info(md(r["text"]) or "_(empty answer — try again)_", icon="📐")
            st.caption("Tools called: " + (", ".join(c["name"] for c in r["calls"]) or "none") + (f" · PRISM session `{r['session']}`" if tracing_enabled() else ""))
            with st.expander("Computed figures sent to PRISM as truth (metadata)"):
                st.json(r["meta"])

    for i, key in enumerate(TRACKS, start=1):
        with tabs[i]:
            render_track(key)
