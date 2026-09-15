"""Replay the fixed cohort through bizmax Core and score it — the before/after evidence.

    .venv/Scripts/python -m tests.cohort.replay --tag baseline [--repeat 2] [--keyword-router] [--only nex_full,vel_leakage]

Each case x repeat is one traced PRISM session (session id `cohort-<tag>-<case>-r<n>`).
Scores per run: routed agents vs expected (containment + Jaccard), per-agent
status/seconds/timeouts, synthesis parse mode, grounding of the executive summary
against every sub-agent's tool results (src/eval/grounding.py), computed_* equality vs the
golden snapshot (determinism), wall time. Writes tests/cohort/results/<tag>.json and .md.
Runs sequentially: NIM rate limits, and a stable latency baseline.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.core.orchestrator import run_core
from src.data.ingest import bundle_from_paths
from src.data.samples import list_samples, sample_files
from src.prism.tracing import analysis_run
from src.tracks.registry import TRACKS

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
GOLDEN = HERE / "golden"


def load_cases(only: str | None) -> list[dict[str, Any]]:
    cases = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))
    if only:
        keep = {s.strip() for s in only.split(",")}
        cases = [c for c in cases if c["id"] in keep]
    return cases


def golden_truth(slug: str) -> dict[str, Any]:
    """The computed_* values the tracks would send, derived from the golden engine snapshot."""
    g = json.loads((GOLDEN / f"{slug}.json").read_text(encoding="utf-8"))
    return {
        "computed_expected_loss": round(g["freq"]["poisson_lambda"] * g["sev"]["lognormal_mean"], 2),
        "computed_ibnr": g["ibnr"]["ibnr"],
        "computed_frequency": g["freq"]["poisson_lambda"],
        "computed_severity": g["sev"]["lognormal_mean"],
        "computed_scr_ratio_pct": g["solv"]["scr_ratio_pct"],
        "computed_capital_required_99_5": g["adq"]["capital_required_99_5"],
        "computed_optimal_retention": g["sweep"]["optimal_retention"],
        "computed_severity_change_pct": g["intel"]["severity_change_pct"],
        "computed_anomaly_count": g["intel"]["anomaly_count"],
    }


def score_one(case: dict[str, Any], res, truth: dict[str, Any], wall: float) -> dict[str, Any]:
    routed = set(res.route.active)
    expected = set(case["expected_agents"]) | {"actuary"}
    contained = expected <= routed
    jaccard = len(expected & routed) / len(expected | routed)
    g = res.grounding()
    meta = res.metadata
    mismatches = {k: (meta.get(k), v) for k, v in truth.items() if k in meta and meta.get(k) != v}
    return {
        "case": case["id"], "business": case["business"], "question": case["question"],
        "routed": sorted(routed), "route_source": res.route.source, "route_error": res.route.error,
        "expected": sorted(expected), "routing_contained": contained, "routing_jaccard": round(jaccard, 3),
        "agents": {k: {"ok": bool(r.text and not r.error), "seconds": round(r.seconds, 1), "error": r.error, "tools": r.tool_names}
                   for k, r in res.reports.items()},
        "timed_out": res.timed_out,
        "synthesis_parse": res.synthesis.get("_parse"),
        "friction": bool((res.synthesis.get("strategic_risk_friction") or "").strip().lower() not in ("", "none", "none identified")),
        "grounding_rate": g["rate"], "figures_checked": g["checked"], "ungrounded": g["ungrounded"],
        "determinism_ok": not mismatches, "computed_mismatches": mismatches,
        "timings": {k: round(v, 1) for k, v in res.timings.items()}, "wall_s": round(wall, 1),
        "session_id": res.session_id,
        "summary": res.synthesis.get("executive_summary", ""),
    }


def summarise(tag: str, runs: list[dict[str, Any]]) -> str:
    n = len(runs)
    if not n:
        return f"# {tag}\n\nno runs\n"
    pct = lambda xs: f"{100 * sum(xs) / n:.0f}%"
    walls = sorted(r["wall_s"] for r in runs)
    p = lambda q: walls[min(n - 1, int(q * n))]
    agent_runs = [a for r in runs for a in r["agents"].values()]
    rows = [
        ("Runs (cases x repeats)", str(n)),
        ("Grounding rate (mean)", f"{statistics.mean(r['grounding_rate'] for r in runs):.1f}%"),
        ("Runs fully grounded", pct([not r["ungrounded"] for r in runs])),
        ("Ungrounded figures (total)", str(sum(len(r["ungrounded"]) for r in runs))),
        ("Routing: expected agents present", pct([r["routing_contained"] for r in runs])),
        ("Routing: mean Jaccard", f"{statistics.mean(r['routing_jaccard'] for r in runs):.2f}"),
        ("Routing via LLM", pct([r["route_source"] == "llm" for r in runs])),
        ("Synthesis parsed as JSON", pct([r["synthesis_parse"] == "json" for r in runs])),
        ("Agent runs OK", f"{100 * sum(a['ok'] for a in agent_runs) / max(len(agent_runs), 1):.0f}% of {len(agent_runs)}"),
        ("Runs with a timeout", pct([bool(r["timed_out"]) for r in runs])),
        ("Determinism (computed_* == golden)", pct([r["determinism_ok"] for r in runs])),
        ("Wall time p50 / p90", f"{p(0.5):.0f}s / {p(0.9):.0f}s"),
    ]
    md = [f"# cohort `{tag}`", "", "| metric | value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in rows]
    md += ["", "| case | routed | grounding | ungrounded | parse | wall |", "|---|---|---|---|---|---|"]
    for r in runs:
        md.append(f"| {r['case']} | {', '.join(r['routed'])} | {r['grounding_rate']}% | {', '.join(r['ungrounded'][:3]) or '—'} | {r['synthesis_parse']} | {r['wall_s']}s |")
    return "\n".join(md) + "\n"


def main() -> int:
    load_dotenv(".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--keyword-router", action="store_true")
    ap.add_argument("--deep-synthesis", action="store_true", help="benchmark the optional extra NIM synthesis call")
    ap.add_argument("--only", default=None, help="comma-separated case ids")
    args = ap.parse_args()

    cases = load_cases(args.only)
    samples = list_samples()
    bundles = {slug: bundle_from_paths(sample_files(slug)) for slug in {c["business"] for c in cases}}
    RESULTS.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    started = time.time()
    for rep in range(1, args.repeat + 1):
        for case in cases:
            slug = case["business"]
            sid = f"cohort-{args.tag}-{case['id']}-r{rep}"
            t0 = time.time()
            with analysis_run(sid, {"mode": "auto", "cohort": args.tag, "case": case["id"], "repeat": rep}, blocking=True):
                res = run_core(case["question"], bundles[slug], samples[slug]["programme"], sid,
                               use_llm_router=not args.keyword_router, deep_synthesis=args.deep_synthesis)
            rec = score_one(case, res, golden_truth(slug), time.time() - t0)
            runs.append(rec)
            print(f"[{time.time()-started:5.0f}s] {case['id']:22s} r{rep} route={rec['route_source']:8s} {','.join(rec['routed']):40s} "
                  f"grounding={rec['grounding_rate']:5.1f}% parse={rec['synthesis_parse']:9s} det={'ok' if rec['determinism_ok'] else 'DIFF'} wall={rec['wall_s']:.0f}s"
                  + (f" TIMEOUT {rec['timed_out']}" if rec["timed_out"] else ""))
            (RESULTS / f"{args.tag}.json").write_text(json.dumps({"tag": args.tag, "runs": runs}, indent=1), encoding="utf-8")  # incremental
    (RESULTS / f"{args.tag}.md").write_text(summarise(args.tag, runs), encoding="utf-8")
    print(f"\nwrote {RESULTS / (args.tag + '.json')} and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
