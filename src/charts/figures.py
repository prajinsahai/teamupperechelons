"""matplotlib figures for the dashboard. Every function takes ENGINE OUTPUT, never LLM text.

Palette: dark panel with the MAGI cyan/green accents so charts sit next to the Core panel.
"""

from __future__ import annotations

from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

matplotlib.use("Agg")

BG, FG, GRID = "#0b1220", "#cfe3f5", "#1f2a3d"
CYAN, GREEN, ORANGE, RED, MUTED = "#00E5FF", "#22c55e", "#f59e0b", "#ef4444", "#4b5a73"


def _fig(w: float = 7.0, h: float = 3.2) -> tuple[Figure, Any]:
    fig, ax = plt.subplots(figsize=(w, h), facecolor=BG)
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors=FG, labelsize=8)
    ax.xaxis.label.set_color(FG); ax.yaxis.label.set_color(FG); ax.title.set_color(FG)
    ax.grid(axis="y", color=GRID, alpha=0.6, linewidth=0.6)
    return fig, ax


def _m(x: float) -> str:
    return f"${x / 1e6:.1f}M" if abs(x) >= 1e6 else f"${x / 1e3:.0f}K" if abs(x) >= 1e3 else f"${x:.0f}"


# ---------------------------------------------------------------- actuary
def loss_triangle_heatmap(tri: dict[str, Any]) -> Figure:
    years = list(tri["triangle"].keys()); buckets = tri["buckets"]
    mat = np.array([[tri["triangle"][y].get(b) if tri["triangle"][y].get(b) is not None else np.nan for b in buckets] for y in years], dtype=float)
    fig, ax = _fig(7.5, 0.35 * len(years) + 1.4)
    ax.grid(False)
    im = ax.imshow(mat / 1e6, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(buckets))); ax.set_xticklabels([f"{b}m" for b in buckets])
    ax.set_yticks(range(len(years))); ax.set_yticklabels(years)
    for i in range(len(years)):
        for j in range(len(buckets)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j] / 1e6:.1f}", ha="center", va="center", fontsize=7, color="white" if mat[i, j] < np.nanmax(mat) * 0.6 else "black")
    ax.set_title("Cumulative incurred development triangle ($M) — chain-ladder", fontsize=10, loc="left")
    ax.set_xlabel("development month"); ax.set_ylabel("accident year")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.ax.tick_params(colors=FG, labelsize=7)
    fig.tight_layout(); return fig


def ibnr_by_year(ibnr: dict[str, Any], tri: dict[str, Any] | None = None) -> Figure:
    rows = ibnr["by_accident_year"]
    years = [r["accident_year"] for r in rows]
    rep = np.array([r["reported"] for r in rows]) / 1e6
    ib = np.array([r["ibnr"] for r in rows]) / 1e6
    fig, ax = _fig()
    ax.bar(years, rep, color=MUTED, label="reported to date")
    ax.bar(years, ib, bottom=rep, color=CYAN, label="IBNR (LDF)")
    if tri:
        cl = {u["accident_year"]: u["chain_ladder_ultimate"] / 1e6 for u in tri["by_accident_year"]}
        ax.plot(years, [cl.get(y, np.nan) for y in years], color=ORANGE, marker="o", ms=4, lw=1.2, label="chain-ladder ultimate")
    ax.set_ylabel("$M"); ax.set_title("Reported vs IBNR by accident year", fontsize=10, loc="left")
    ax.legend(fontsize=7, facecolor=BG, labelcolor=FG, edgecolor=GRID); fig.tight_layout(); return fig


def frequency_severity_by_year(freq: dict[str, Any], df: pd.DataFrame) -> Figure:
    years = sorted(freq["claims_per_year"]); counts = [freq["claims_per_year"][y] for y in years]
    sev = df.groupby("accident_year")["claim_amount"].mean().reindex(years).to_numpy() / 1e3
    fig, ax = _fig()
    ax.bar(years, counts, color=MUTED, label="claims (count)")
    ax.set_ylabel("claims / year")
    ax2 = ax.twinx(); ax2.plot(years, sev, color=GREEN, marker="o", ms=4, lw=1.4, label="mean severity ($K)")
    ax2.tick_params(colors=FG, labelsize=8); ax2.set_ylabel("mean severity ($K)", color=FG)
    for sp in ax2.spines.values(): sp.set_color(GRID)
    ax.set_title(f"Frequency and severity by year — λ={freq['poisson_lambda']}/yr, dispersion {freq['dispersion_ratio']}", fontsize=10, loc="left")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, facecolor=BG, labelcolor=FG, edgecolor=GRID, loc="upper left"); fig.tight_layout(); return fig


def stress_bars(stress: dict[str, Any]) -> Figure:
    rows = stress["scenarios"]
    fig, ax = _fig(7, 2.8)
    names = [r["scenario"].replace("_", " ") for r in rows]; vals = [r["expected_annual_loss"] / 1e6 for r in rows]
    cols = [MUTED if r["scenario"] == "base" else ORANGE if r["change_vs_base_pct"] < 60 else RED for r in rows]
    ax.barh(names, vals, color=cols)
    for i, r in enumerate(rows):
        ax.text(vals[i], i, f"  {_m(r['expected_annual_loss'])} ({r['change_vs_base_pct']:+.0f}%)", va="center", fontsize=7, color=FG)
    ax.set_xlabel("expected annual loss ($M)"); ax.set_title("Stress scenarios", fontsize=10, loc="left"); ax.grid(axis="x", color=GRID, alpha=0.6)
    fig.tight_layout(); return fig


# ---------------------------------------------------------------- capital / risk
def aggregate_histogram(sim: dict[str, Any], capital_held: float | None = None) -> Figure:
    paths = sim.get("_net_paths") if sim.get("retention") is not None else sim.get("_gross_paths")
    if paths is None:
        paths = sim["_gross_paths"]
    stats = sim["net_of_reinsurance"] if sim.get("retention") is not None else sim["gross"]
    fig, ax = _fig(7.5, 3.4)
    hi = np.percentile(paths, 99.7)
    ax.hist(np.clip(paths, 0, hi) / 1e6, bins=70, color=MUTED, edgecolor=BG)
    for v, c, lab in [(stats["mean"], GREEN, "mean"), (stats["var_95"], CYAN, "VaR 95"), (stats["var_99_5"], ORANGE, "VaR 99.5"), (stats["tvar_99"], RED, "TVaR 99")]:
        ax.axvline(v / 1e6, color=c, lw=1.2, ls="--"); ax.text(v / 1e6, ax.get_ylim()[1] * 0.9, f" {lab}\n {_m(v)}", color=c, fontsize=7, va="top")
    if capital_held:
        ax.axvline(capital_held / 1e6, color="white", lw=1.4); ax.text(capital_held / 1e6, ax.get_ylim()[1] * 0.55, f" capital\n {_m(capital_held)}", color="white", fontsize=7, va="top")
    basis = "net of reinsurance" if sim.get("retention") is not None else "gross"
    ax.set_xlabel("annual aggregate loss ($M)"); ax.set_ylabel("simulated years")
    ax.set_title(f"Monte Carlo annual loss — {sim['paths']:,} paths, {basis} ({sim['confidence_tier']})", fontsize=10, loc="left")
    fig.tight_layout(); return fig


def solvency_gauge(sp: dict[str, Any]) -> Figure:
    ratio = float(sp["scr_ratio_pct"]); cap = 300.0
    fig = plt.figure(figsize=(4.2, 2.6), facecolor=BG); ax = fig.add_subplot(111, polar=True); ax.set_facecolor(BG)
    ax.set_theta_zero_location("W"); ax.set_theta_direction(-1); ax.set_thetamin(0); ax.set_thetamax(180)
    ax.set_yticklabels([]); ax.set_xticks([]); ax.grid(False); ax.spines["polar"].set_visible(False)
    bands = [(0, 100, RED), (100, 120, ORANGE), (120, 200, GREEN), (200, cap, CYAN)]
    for lo, hi_, c in bands:
        th = np.linspace(np.pi * lo / cap, np.pi * hi_ / cap, 40)
        ax.fill_between(th, 0.72, 1.0, color=c, alpha=0.85)
    ang = np.pi * min(ratio, cap) / cap
    ax.plot([ang, ang], [0, 0.95], color="white", lw=2.5); ax.plot([ang], [0], marker="o", color="white", ms=6)
    ax.set_ylim(0, 1)
    fig.text(0.5, 0.18, f"SCR ratio {ratio:.0f}%", ha="center", color=FG, fontsize=13, fontweight="bold")
    fig.text(0.5, 0.06, f"{sp['status'].upper()} · own funds {_m(sp['own_funds'])} vs SCR {_m(sp['scr'])}", ha="center", color=FG, fontsize=8)
    return fig


def exposure_vs_loss(expo: dict[str, Any]) -> Figure:
    rows = expo["by_year"]; years = [r["accident_year"] for r in rows]
    ex = np.array([r["exposure"] or 0 for r in rows]) / 1e6; ls = np.array([r["loss"] or 0 for r in rows]) / 1e6
    fig, ax = _fig()
    ax.plot(years, ex, color=MUTED, marker="s", ms=4, lw=1.3, label="exposure ($M)")
    ax2 = ax.twinx(); ax2.bar(years, ls, color=CYAN, alpha=0.7, label="loss ($M)")
    ax2.tick_params(colors=FG, labelsize=8); [sp.set_color(GRID) for sp in ax2.spines.values()]
    ax.set_ylabel("exposure ($M)"); ax2.set_ylabel("loss ($M)", color=FG)
    ax.set_title("Exposure movement vs loss by year", fontsize=10, loc="left")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, facecolor=BG, labelcolor=FG, edgecolor=GRID, loc="upper left"); fig.tight_layout(); return fig


def emerging_signals_bars(es: dict[str, Any]) -> Figure:
    rows = es["signals"]; names = [r["line_of_business"] for r in rows]
    fig, ax = _fig(7, 0.42 * len(rows) + 1.2)
    y = np.arange(len(rows))
    ax.barh(y - 0.2, [r["frequency_change_pct"] for r in rows], height=0.38, color=MUTED, label="frequency Δ%")
    ax.barh(y + 0.2, [r["severity_change_pct"] for r in rows], height=0.38, color=[{"red": RED, "amber": ORANGE, "green": GREEN}[r["signal"]] for r in rows], label="severity Δ% (RAG)")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8); ax.axvline(0, color=FG, lw=0.6)
    ax.set_xlabel("% change, last 2 years vs earlier"); ax.set_title("Emerging signals by line", fontsize=10, loc="left"); ax.grid(axis="x", color=GRID, alpha=0.6)
    ax.legend(fontsize=7, facecolor=BG, labelcolor=FG, edgecolor=GRID); fig.tight_layout(); return fig


def loss_by_line(df: pd.DataFrame) -> Figure:
    piv = df.pivot_table(index="accident_year", columns="line_of_business", values="claim_amount", aggfunc="sum").fillna(0) / 1e6
    fig, ax = _fig()
    bottom = np.zeros(len(piv)); palette = [CYAN, GREEN, ORANGE, "#a78bfa", "#f472b6", MUTED, RED]
    for i, col in enumerate(piv.columns):
        ax.bar(piv.index, piv[col], bottom=bottom, color=palette[i % len(palette)], label=col); bottom += piv[col].to_numpy()
    ax.set_ylabel("$M"); ax.set_title("Loss by line of business and accident year", fontsize=10, loc="left")
    ax.legend(fontsize=7, facecolor=BG, labelcolor=FG, edgecolor=GRID, ncol=2); fig.tight_layout(); return fig


# ---------------------------------------------------------------- reinsurance
def reinsurance_flow(rec: dict[str, Any]) -> Figure:
    """Where the gross loss goes: retained below retention, ceded to the layer, uncovered above it."""
    gross = rec["gross_loss"]; ceded = rec["expected_recovery"]; above = rec["uncovered_above_layer"]; retained = gross - ceded - above
    fig, ax = _fig(7.5, 3.0); ax.grid(False); ax.set_axis_off()
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.add_patch(plt.Rectangle((0.3, 2.5), 1.8, 5, color=MUTED)); ax.text(1.2, 5.0, f"GROSS\n{_m(gross)}", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    parts = [("retained\n(below retention)", retained, GREEN, 8.2), (f"ceded to layer\n{_m(rec['limit'])} xs {_m(rec['retention'])}", ceded, CYAN, 5.0), ("uncovered\n(above layer)", above, RED, 1.8)]
    for label, val, col, yc in parts:
        h = max(val / gross * 5, 0.15)
        ax.add_patch(plt.Polygon([[2.1, 2.5 + (yc - 1.5) * 0.0 + 0], [2.1, 7.5], [6.2, yc + h / 2], [6.2, yc - h / 2]], color=col, alpha=0.18))
        ax.add_patch(plt.Rectangle((6.2, yc - h / 2), 1.8, h, color=col))
        ax.text(8.1, yc, f"{label}\n{_m(val)} ({val / gross * 100:.0f}%)", va="center", fontsize=8, color=FG)
    ax.set_title(f"Reinsurance flow — {rec['claims_in_layer']} claims in layer, {rec['claims_exhausting_layer']} exhaust it", fontsize=10, loc="left", color=FG)
    fig.tight_layout(); return fig


def tcor_curve(sweep: dict[str, Any], current_retention: float | None = None) -> Figure:
    rows = sorted(sweep["sweep"], key=lambda r: r["retention"])
    x = [r["retention"] / 1e6 for r in rows]
    fig, ax = _fig()
    ax.plot(x, [r["tcor"] / 1e6 for r in rows], color=CYAN, marker="o", ms=4, lw=1.6, label="TCoR")
    ax.plot(x, [r["reinsurance_premium"] / 1e6 for r in rows], color=ORANGE, lw=1.1, ls="--", label="reinsurance premium")
    ax.plot(x, [r["expected_retained_loss"] / 1e6 for r in rows], color=GREEN, lw=1.1, ls="--", label="expected retained loss")
    ax.plot(x, [r["capital_charge"] / 1e6 for r in rows], color=MUTED, lw=1.1, ls="--", label="capital charge")
    ax.axvline(sweep["optimal_retention"] / 1e6, color=CYAN, lw=0.8, alpha=0.6); ax.text(sweep["optimal_retention"] / 1e6, ax.get_ylim()[1] * 0.95, f" optimal {_m(sweep['optimal_retention'])}", color=CYAN, fontsize=7, va="top")
    if current_retention:
        ax.axvline(current_retention / 1e6, color="white", lw=0.8, alpha=0.6); ax.text(current_retention / 1e6, ax.get_ylim()[1] * 0.8, f" current {_m(current_retention)}", color="white", fontsize=7, va="top")
    ax.set_xlabel("retention ($M)"); ax.set_ylabel("$M / year"); ax.set_title("Total Cost of Risk across retentions", fontsize=10, loc="left")
    ax.legend(fontsize=7, facecolor=BG, labelcolor=FG, edgecolor=GRID); fig.tight_layout(); return fig


def structures_bar(cmp: dict[str, Any], qs: dict[str, Any] | None = None) -> Figure:
    items = [("per-occurrence XoL", cmp["per_occurrence_xol"]), ("aggregate stop-loss", cmp["aggregate_stop_loss"])]
    if qs:
        items.append(("quota share 30%", qs["quota_share"]))
    fig, ax = _fig(7, 2.8)
    names = [n for n, _ in items]; keys = [("reinsurance_premium", ORANGE), ("expected_retained_loss", GREEN), ("capital_charge", MUTED)]
    left = np.zeros(len(items))
    for k, c in keys:
        vals = np.array([it[k] for _, it in items]) / 1e6
        ax.barh(names, vals, left=left, color=c, label=k.replace("_", " ")); left += vals
    for i, (_, it) in enumerate(items):
        ax.text(left[i], i, f"  TCoR {_m(it['tcor'])}", va="center", fontsize=8, color=FG)
    ax.set_xlabel("$M / year"); ax.set_title("Programme structures compared (TCoR components)", fontsize=10, loc="left"); ax.grid(axis="x", color=GRID, alpha=0.6)
    ax.legend(fontsize=7, facecolor=BG, labelcolor=FG, edgecolor=GRID, loc="lower right"); fig.tight_layout(); return fig


# ---------------------------------------------------------------- claims
def severity_trend(intel: dict[str, Any]) -> Figure:
    years = list(intel["severity_by_year"]); vals = [v / 1e3 for v in intel["severity_by_year"].values()]
    fig, ax = _fig()
    ax.bar(years, vals, color=[ORANGE if y == intel["latest_year"] else MUTED for y in years])
    ax.axhline(intel["prior_years_mean_severity"] / 1e3, color=GREEN, ls="--", lw=1, label="prior 3-yr mean")
    ax.set_ylabel("mean claim ($K)"); ax.set_title(f"Mean severity by accident year — latest {intel['severity_change_pct']:+.1f}% vs prior", fontsize=10, loc="left")
    ax.legend(fontsize=7, facecolor=BG, labelcolor=FG, edgecolor=GRID); fig.tight_layout(); return fig


def development_curve(dp: dict[str, Any]) -> Figure:
    pts = sorted(dp["paid_to_reported_by_dev_month"].items())
    fig, ax = _fig(7, 2.8)
    ax.plot([k for k, _ in pts], [v for _, v in pts], color=CYAN, marker="o", ms=4, lw=1.5)
    ax.set_ylim(0, 1.05); ax.set_xlabel("development month"); ax.set_ylabel("paid / reported")
    ax.set_title("Payment development pattern", fontsize=10, loc="left"); fig.tight_layout(); return fig


def leakage_bars(lk: dict[str, Any]) -> Figure:
    rows = lk["top_overpaid_claims"]
    fig, ax = _fig(7, 0.35 * len(rows) + 1.2)
    names = [f"{r['claim_id']} · {r['line_of_business']}" for r in rows]
    ax.barh(names, [r["paid_amount"] / 1e3 for r in rows], color=RED, label="paid")
    ax.barh(names, [(r["paid_amount"] - r["overpaid"]) / 1e3 for r in rows], color=MUTED, label="reported")
    ax.invert_yaxis(); ax.set_xlabel("$K"); ax.tick_params(axis="y", labelsize=7)
    ax.set_title(f"Top overpaid claims — leakage {lk['leakage_rate_pct']}% of paid ({lk['overpaid_claim_count']} claims)", fontsize=10, loc="left"); ax.grid(axis="x", color=GRID, alpha=0.6)
    ax.legend(fontsize=7, facecolor=BG, labelcolor=FG, edgecolor=GRID, loc="lower right"); fig.tight_layout(); return fig


# ---------------------------------------------------------------- policy
def clause_counts(cl: dict[str, Any]) -> Figure:
    counts = cl["counts"]
    fig, ax = _fig(6, 2.6)
    ax.barh(list(counts), list(counts.values()), color=CYAN); ax.grid(axis="x", color=GRID, alpha=0.6)
    ax.set_title("Clauses found by category", fontsize=10, loc="left"); fig.tight_layout(); return fig
