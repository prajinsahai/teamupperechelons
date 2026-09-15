"""Run bizmax Core from the command line (traced to PRISM as one session).

    .venv/Scripts/python -m src.core.run nexus "Is our reinsurance right for a cyber catastrophe?"
    .venv/Scripts/python -m src.core.run velocity "Where is claims leakage costing us?" --keyword-router
"""

from __future__ import annotations

import argparse
import json
import time

from dotenv import load_dotenv

from src.core.orchestrator import run_core
from src.data.ingest import bundle_from_paths
from src.data.samples import list_samples, sample_files
from src.prism.tracing import analysis_run, new_session_id
from src.tracks.registry import TRACKS


def main() -> int:
    load_dotenv(".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("business", help="sample slug (velocity | nexus | aegis | terrafirma) or a claims file path")
    ap.add_argument("question")
    ap.add_argument("--keyword-router", action="store_true")
    ap.add_argument("--deep-synthesis", action="store_true", help="run the optional extra NIM synthesis call")
    args = ap.parse_args()

    samples = list_samples()
    if args.business in samples:
        prof = samples[args.business]; files = sample_files(args.business); params = dict(prof["programme"])
        print(f"business: {prof['name']} — {prof['tagline']}")
    else:
        files = [args.business]; params = {"retention": 500_000.0, "limit": 4_500_000.0, "capital_held": 60_000_000.0}
    bundle = bundle_from_paths(files)
    session_id = new_session_id("core")
    t0 = time.time()

    def on_update(phase: str, sts: dict[str, str]) -> None:
        lit = [k for k, s in sts.items() if s in ("running", "done")]
        print(f"  [{time.time()-t0:5.0f}s] {phase:12s} " + " ".join(f"{k}:{s}" for k, s in sts.items() if k in lit))

    with analysis_run(session_id, {"mode": "auto", "question": args.question[:200]}, blocking=True):
        res = run_core(args.question, bundle, params, session_id, on_update=on_update,
                       use_llm_router=not args.keyword_router, deep_synthesis=args.deep_synthesis)

    print(f"\nROUTE ({res.route.source}): {[TRACKS[k].name for k in res.route.active]} — {res.route.objective}")
    if res.route.error:
        print("  note:", res.route.error)
    for k, r in res.reports.items():
        print(f"\n--- {r.name} ({r.seconds:.0f}s, tools {r.tool_names}){' ERROR: ' + r.error if r.error else ''}\n{r.text[:1200]}")
    syn = res.synthesis
    g = res.grounding()
    print(f"\n=== EXECUTIVE SUMMARY (parse={syn.get('_parse')}, grounding {g['rate']}% of {g['checked']} figures{', ungrounded ' + str(g['ungrounded']) if g['ungrounded'] else ''})")
    if res.timed_out:
        print("timed out:", res.timed_out)
    print(syn.get("executive_summary", ""))
    print("\nfriction:", syn.get("strategic_risk_friction"))
    print("key figures:", json.dumps(syn.get("key_figures", {}), indent=1)[:800])
    print(f"\ntimings: {res.timings} | session {session_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
