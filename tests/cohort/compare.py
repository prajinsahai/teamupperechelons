"""Before/after table for two cohort tags — the slide.

    .venv/Scripts/python -m tests.cohort.compare baseline fix1
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


def metrics(tag: str) -> dict[str, float]:
    runs = json.loads((RESULTS / f"{tag}.json").read_text(encoding="utf-8"))["runs"]
    n = len(runs)
    agent_runs = [a for r in runs for a in r["agents"].values()]
    walls = sorted(r["wall_s"] for r in runs)
    return {
        "runs": n,
        "grounding_rate_mean_pct": statistics.mean(r["grounding_rate"] for r in runs),
        "runs_fully_grounded_pct": 100 * sum(not r["ungrounded"] for r in runs) / n,
        "ungrounded_figures": sum(len(r["ungrounded"]) for r in runs),
        "routing_expected_present_pct": 100 * sum(r["routing_contained"] for r in runs) / n,
        "routing_jaccard_mean": statistics.mean(r["routing_jaccard"] for r in runs),
        "synthesis_json_pct": 100 * sum(r["synthesis_parse"] == "json" for r in runs) / n,
        "agent_ok_pct": 100 * sum(a["ok"] for a in agent_runs) / max(len(agent_runs), 1),
        "runs_with_timeout_pct": 100 * sum(bool(r["timed_out"]) for r in runs) / n,
        "determinism_pct": 100 * sum(r["determinism_ok"] for r in runs) / n,
        "wall_p50_s": walls[n // 2],
        "wall_p90_s": walls[min(n - 1, int(0.9 * n))],
    }


LABELS = {
    "runs": ("Runs", "{:.0f}"), "grounding_rate_mean_pct": ("Grounding rate (mean)", "{:.1f}%"),
    "runs_fully_grounded_pct": ("Runs fully grounded", "{:.0f}%"), "ungrounded_figures": ("Ungrounded figures", "{:.0f}"),
    "routing_expected_present_pct": ("Routing: expected agents present", "{:.0f}%"), "routing_jaccard_mean": ("Routing: Jaccard", "{:.2f}"),
    "synthesis_json_pct": ("Synthesis parsed as JSON", "{:.0f}%"), "agent_ok_pct": ("Agent runs OK", "{:.0f}%"),
    "runs_with_timeout_pct": ("Runs with a timeout", "{:.0f}%"), "determinism_pct": ("Determinism (computed_* == golden)", "{:.0f}%"),
    "wall_p50_s": ("Wall time p50", "{:.0f}s"), "wall_p90_s": ("Wall time p90", "{:.0f}s"),
}
HIGHER_IS_BETTER = {"grounding_rate_mean_pct", "runs_fully_grounded_pct", "routing_expected_present_pct", "routing_jaccard_mean",
                    "synthesis_json_pct", "agent_ok_pct", "determinism_pct"}


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__); return 2
    a, b = sys.argv[1], sys.argv[2]
    ma, mb = metrics(a), metrics(b)
    lines = [f"| metric | {a} | {b} | Δ |", "|---|---|---|---|"]
    for key, (label, fmt) in LABELS.items():
        va, vb = ma[key], mb[key]
        d = vb - va
        if key == "runs":
            arrow = ""
        elif d == 0:
            arrow = "="
        else:
            better = (d > 0) if key in HIGHER_IS_BETTER else (d < 0)
            arrow = ("▲ " if d > 0 else "▼ ") + fmt.format(abs(d)) + (" better" if better else " worse")
        lines.append(f"| {label} | {fmt.format(va)} | {fmt.format(vb)} | {arrow} |")
    out = "\n".join(lines)
    print(out)
    (RESULTS / f"compare_{a}_vs_{b}.md").write_text(out + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
