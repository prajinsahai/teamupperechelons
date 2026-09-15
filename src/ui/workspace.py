"""Pure presentation components for the post-analysis operations workspace.

These helpers only format values already produced by the deterministic engine or
the traced agent run. They do not calculate actuarial results.
"""

from __future__ import annotations

from html import escape
from typing import Any, Iterable

import streamlit as st


PAGE_META = {
    "Dashboard": ("Portfolio overview and multi-agent results", "GRID"),
    "Claims Analysis": ("ML anomaly detection and trend monitoring", "CLAIMS"),
    "Policy Intelligence": ("Document-grounded coverage evidence", "POLICY"),
    "Reinsurance": ("Treaty analysis and net exposure", "TREATY"),
    "Actuarial Engine": ("Deterministic calculations and simulations", "ENGINE"),
    "Audit Trail": ("Model governance and traceability", "AUDIT"),
}

AGENTS = (
    ("actuary", "AI Actuary", "Reserving, pricing and IBNR", "01"),
    ("claims", "AI Claims Analyst", "Severity, leakage and anomalies", "02"),
    ("reinsurance", "AI Reinsurance Manager", "Risk transfer and recovery", "03"),
    ("capital", "AI Capital Manager", "Solvency and capital adequacy", "04"),
    ("risk", "AI Risk Manager", "Stress, VaR and concentration", "05"),
    ("policy", "AI Policy Analyst", "Coverage clauses and exclusions", "06"),
)


def workspace_header(
    page: str,
    *,
    business: str,
    model: str,
    traced: bool,
    claim_count: int,
) -> None:
    subtitle, code = PAGE_META[page]
    engine_state = "Engine ready" if claim_count else "Awaiting claims"
    prism_state = "PRISM live" if traced else "PRISM offline"
    prism_class = "live" if traced else "offline"
    st.markdown(
        f"""
        <header class="bz-workspace-header">
          <div class="bz-workspace-title">
            <span class="bz-menu-glyph">≡</span>
            <div><div class="bz-page-code">{escape(code)}</div><h1>{escape(page)}</h1><p>{escape(subtitle)}</p></div>
          </div>
          <div class="bz-workspace-actions">
            <span class="bz-chip live"><i class="bz-dot"></i>{escape(engine_state)}</span>
            <span class="bz-chip {prism_class}">{escape(prism_state)}</span>
            <span class="bz-chip model">{escape(model)}</span>
            <span class="bz-portfolio">{escape(business)}</span>
          </div>
        </header>
        """,
        unsafe_allow_html=True,
    )


def portfolio_bar(*, business: str, focus: str, source_count: int, claim_count: int, mode: str) -> None:
    st.markdown(
        f"""
        <div class="bz-portfolio-bar">
          <div><span>ACTIVE PORTFOLIO</span><strong>{escape(business)}</strong></div>
          <p>{escape(focus)}</p>
          <div class="bz-portfolio-facts"><b>{claim_count:,}</b> claims</div>
          <div class="bz-portfolio-facts"><b>{source_count}</b> sources</div>
          <div class="bz-mode-pill">{escape(mode.upper())}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def panel_header(title: str, subtitle: str = "", *, badge: str = "", tone: str = "cyan", icon: str = "◇") -> None:
    badge_html = f'<span class="bz-panel-badge {escape(tone)}">{escape(badge)}</span>' if badge else ""
    st.markdown(
        f"""
        <div class="bz-panel-heading">
          <div class="bz-panel-icon {escape(tone)}">{escape(icon)}</div>
          <div><h3>{escape(title)}</h3><p>{escape(subtitle)}</p></div>
          {badge_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_grid(cards: Iterable[dict[str, str]]) -> None:
    blocks = []
    for card in cards:
        tone = escape(card.get("tone", "cyan"))
        blocks.append(
            f'<article class="bz-kpi-card {tone}"><div class="bz-kpi-top">'
            f'<span class="bz-kpi-icon">{escape(card.get("icon", "◇"))}</span>'
            f'<span>{escape(card["label"])}</span></div>'
            f'<strong>{escape(card["value"])}</strong>'
            f'<small>{escape(card.get("note", ""))}</small></article>'
        )
    st.markdown(f'<div class="bz-kpi-grid">{"".join(blocks)}</div>', unsafe_allow_html=True)


def risk_metric_list(metrics: Iterable[tuple[str, str, str]]) -> None:
    rows = "".join(
        f'<div class="bz-risk-row"><span>{escape(label)}<small>{escape(tag)}</small></span><b>{escape(value)}</b></div>'
        for label, value, tag in metrics
    )
    st.markdown(f'<div class="bz-risk-list">{rows}</div>', unsafe_allow_html=True)


def agent_network(result: Any | None) -> None:
    active = set(result.route.active) if result else set()
    statuses = result.statuses() if result else {}
    cards = []
    for key, name, role, number in AGENTS:
        report = result.reports.get(key) if result else None
        status = statuses.get(key, "idle")
        if status == "done":
            state, detail = "complete", f"{len(report.tool_calls)} tools · {report.seconds:.1f}s"
        elif status == "error":
            state, detail = "attention", (report.error or "No report returned")[:80]
        elif key in active:
            state, detail = "selected", "Selected by router"
        elif result:
            state, detail = "standby", "Not routed for this question"
        else:
            state, detail = "ready", "Ready for routing"
        cards.append(
            f'<article class="bz-agent-card {escape(status)}"><div class="bz-agent-icon">{number}</div>'
            f'<div class="bz-agent-copy"><strong>{escape(name)}</strong><span>{escape(role)}</span>'
            f'<small>{escape(detail)}</small></div><i></i><em>{escape(state)}</em></article>'
        )
    st.markdown(f'<div class="bz-agent-network">{"".join(cards)}</div>', unsafe_allow_html=True)


def workflow_trace(result: Any | None, *, source_count: int, claim_count: int, traced: bool) -> None:
    if result:
        tool_count = sum(len(r.tool_calls) for r in result.reports.values())
        grounding = result.grounding()
        synth_mode = "rapid assembly" if result.synthesis.get("_mode") == "rapid" else "deep synthesis"
        steps = [
            ("Data ingestion", f"{claim_count:,} claims from {source_count} source(s)", "ready"),
            ("Router", f"{result.route.source} selected {len(result.route.active)} specialists in {result.timings.get('routing', 0):.1f}s", "done"),
            ("Specialist analysis", f"{tool_count} deterministic tool calls · {result.timings.get('processing', 0):.1f}s parallel", "done"),
            ("Decision assembly", f"{synth_mode} · {result.timings.get('synthesizing', 0):.2f}s", "done"),
            ("Grounding and observability", f"{grounding['rate']:.0f}% across {grounding['checked']} checked figures · {'PRISM traced' if traced else 'local only'}", "done" if grounding["grounded"] else "attention"),
        ]
    else:
        steps = [
            ("Data ingestion", f"{claim_count:,} claims from {source_count} source(s)", "ready"),
            ("Router", "Waiting for a Core question", "waiting"),
            ("Specialist analysis", "Runs selected specialists in parallel", "waiting"),
            ("Decision assembly", "Rapid grounded assembly is the default", "waiting"),
            ("Grounding and observability", "PRISM captures the live run" if traced else "PRISM credentials are not loaded", "waiting"),
        ]
    rows = "".join(
        f'<div class="bz-trace-step {state}"><span>{i}</span><div><strong>{escape(title)}</strong><p>{escape(detail)}</p></div><i></i></div>'
        for i, (title, detail, state) in enumerate(steps, 1)
    )
    st.markdown(f'<div class="bz-trace-list">{rows}</div>', unsafe_allow_html=True)


def reinsurance_layers(rec: dict[str, Any], *, money) -> None:
    gross = float(rec["gross_loss"])
    recovery = float(rec["expected_recovery"])
    net = float(rec["net_loss"])
    retained_share = max(8.0, min(78.0, 100.0 * net / gross if gross else 0.0))
    ceded_share = max(4.0, min(92.0, 100.0 * recovery / gross if gross else 0.0))
    st.markdown(
        f"""
        <div class="bz-layer-stack">
          <div class="bz-layer-label"><strong>Retention</strong><span>{escape(money(rec['retention']))} attachment</span></div>
          <div class="bz-layer-track"><div class="bz-layer retained" style="width:{retained_share:.1f}%">Net retained {escape(money(net))}</div></div>
          <div class="bz-layer-label"><strong>Per-occurrence XoL</strong><span>{escape(money(rec['limit']))} limit · {rec['claims_in_layer']} claims entered</span></div>
          <div class="bz-layer-track"><div class="bz-layer ceded" style="margin-left:{min(retained_share,70):.1f}%;width:{min(100-retained_share,ceded_share):.1f}%">Ceded {escape(money(recovery))}</div></div>
          <div class="bz-layer-label"><strong>Above layer</strong><span>{rec['claims_exhausting_layer']} claims exhausted · {escape(money(rec['uncovered_above_layer']))} uncovered</span></div>
          <div class="bz-layer-track"><div class="bz-layer available">Available capacity above attachment</div></div>
          <div class="bz-layer-flow"><div><span>GROSS LOSS</span><b>{escape(money(gross))}</b></div><em>→</em><div><span>RECOVERY</span><b class="green">{escape(money(recovery))}</b></div><em>→</em><div><span>NET LOSS</span><b class="amber">{escape(money(net))}</b></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def clause_card(category: str, document: str, text: str, amounts: list[str], *, selected: bool = False) -> None:
    amount = " · ".join(amounts) if amounts else "No monetary amount parsed"
    cls = " selected" if selected else ""
    st.markdown(
        f'<article class="bz-clause-card{cls}"><div><span>{escape(category.replace("_", " ").title())}</span>'
        f'<small>{escape(document)}</small></div><p>{escape(text)}</p><footer>{escape(amount)}</footer></article>',
        unsafe_allow_html=True,
    )


def evidence_detail(category: str, document: str, text: str, amounts: list[str]) -> None:
    amount = ", ".join(amounts) if amounts else "No amount in this clause"
    st.markdown(
        f"""
        <div class="bz-evidence-detail">
          <div class="bz-evidence-head"><span>DOCUMENT EVIDENCE</span><b>{escape(category.replace('_', ' ').title())}</b></div>
          <blockquote>{escape(text)}</blockquote>
          <div class="bz-evidence-grid"><div><span>SOURCE</span><b>{escape(document)}</b></div><div><span>PARSED AMOUNTS</span><b>{escape(amount)}</b></div></div>
          <footer>Document → sentence → clause category → deterministic evidence → AI reasoning</footer>
        </div>
        """,
        unsafe_allow_html=True,
    )
