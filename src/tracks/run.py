"""Run one or more tracks from the command line (traced to PRISM, one session per track).

    .venv/Scripts/python -m src.tracks.run actuary claims --files data/claims.csv data/sample_policy.txt
    .venv/Scripts/python -m src.tracks.run all
"""

from __future__ import annotations

import argparse
import json
import re
import time

from dotenv import load_dotenv

from src.data.ingest import bundle_from_paths
from src.llm.agent import run_agent
from src.prism.tracing import analysis_run, new_session_id
from src.tracks.registry import TRACKS

DEFAULT_FILES = ["data/claims.csv", "data/sample_policy.txt"]


def grounding_check(text: str, tool_outputs: list[str]) -> dict:
    """Mechanical hallucination check: every number in the answer must appear in some tool output."""
    haystack = " ".join(tool_outputs).replace(",", "")
    nums = re.findall(r"\d[\d,]*\.?\d*", text)
    missing = []
    for n in nums:
        clean = n.replace(",", "").rstrip(".")
        if not clean or clean in ("1", "2", "3", "4", "5", "1.0", "200"):  # bullets / trivial
            continue
        if clean not in haystack:
            missing.append(n)
    return {"numbers": len(nums), "ungrounded": missing, "grounded": not missing}


def main() -> int:
    load_dotenv(".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("tracks", nargs="+", help="track keys or 'all'")
    ap.add_argument("--files", nargs="*", default=DEFAULT_FILES)
    ap.add_argument("--question", default=None)
    ap.add_argument("--retention", type=float, default=500_000)
    ap.add_argument("--limit", type=float, default=4_500_000)
    ap.add_argument("--capital", type=float, default=60_000_000)
    args = ap.parse_args()

    keys = list(TRACKS) if args.tracks == ["all"] else args.tracks
    bundle = bundle_from_paths(args.files)
    params = {"retention": args.retention, "limit": args.limit, "capital_held": args.capital}
    print("loaded:", bundle.sources)

    for key in keys:
        track = TRACKS[key]
        session_id = new_session_id(key)
        t0 = time.time()
        tools = track.build_tools(bundle, params)
        meta = {"track": key, **track.metadata(bundle, params)}
        outputs: list[str] = []
        # wrap tools to capture outputs for the grounding check
        for t in tools:
            orig = t.func
            def _wrapped(*a, _orig=orig, **k):
                r = _orig(*a, **k); outputs.append(str(r)); return r
            t.func = _wrapped
        with analysis_run(session_id, meta, blocking=True):
            text, calls = run_agent(track.system_prompt, args.question or track.default_question, tools, run_name=f"{key}_track")
        g = grounding_check(text, outputs)
        print(f"\n=== {track.name}  ({time.time()-t0:.0f}s, session {session_id})")
        print("tools:", [c["name"] for c in calls])
        print("grounding:", "OK" if g["grounded"] else f"UNGROUNDED {g['ungrounded']}", f"({g['numbers']} numbers)")
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
