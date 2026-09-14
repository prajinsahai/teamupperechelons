"""Emit one live PRISM trace from the real app path, without the Streamlit UI.

    .venv/Scripts/python -m src.prism.smoke

Runs the deterministic engine on data/claims.csv, opens one PRISM session, and
invokes the `get_actuarial_summary` tool through LangChain with the PRISM
callback attached (a real tool span). If NVIDIA_API_KEY is set it also runs
the full recommendation loop (model spans + tool spans). Flushes, then
prints the session id to look for in the dashboard.
"""

from __future__ import annotations

import json
import sys

import pandas as pd
from dotenv import load_dotenv

from src.actuarial.ibnr import calculate_ibnr
from src.actuarial.reinsurance import calculate_reinsurance_recovery
from src.llm.agent import api_key_available, make_summary_tool, run_recommendation
from src.ml.predict import get_claims_intelligence
from src.prism.tracing import analysis_run, build_metadata, callbacks, get_handler, new_session_id, tracing_enabled


def main() -> int:
    load_dotenv()
    if not tracing_enabled():
        print("PRISMTRACE_API_KEY / PRISMTRACE_PROJECT_ID not set — see .env.example", file=sys.stderr)
        return 2

    df = pd.read_csv("data/claims.csv")
    ibnr = calculate_ibnr(df)
    rec = calculate_reinsurance_recovery(df)
    intel = get_claims_intelligence(df)
    out = {
        "portfolio_loss": ibnr["portfolio_loss"],
        "ibnr": ibnr["ibnr"],
        "ultimate_loss": ibnr["ultimate_loss"],
        "expected_recovery": rec["expected_recovery"],
        "net_exposure": round(ibnr["ultimate_loss"] - rec["expected_recovery"], 2),
        "reinsurance_detail": rec,
        "claims_intelligence": intel,
    }
    summary = {k: v for k, v in out.items() if k not in ("reinsurance_detail", "claims_intelligence")}
    summary["reinsurance"] = dict(rec)
    summary["claims_intelligence"] = {k: v for k, v in intel.items() if k != "anomalous_claim_ids"}

    session_id = new_session_id("smoke")
    handler = get_handler()
    print(f"PRISM host: {handler.endpoint}  project: {handler.project_id}  session: {session_id}")

    with analysis_run(session_id, build_metadata(out), blocking=True):  # script exits right after
        # Real app tool, real LangChain invoke, PRISM callback attached.
        tool = make_summary_tool(summary)
        result = tool.invoke({}, config={"callbacks": callbacks()})
        print(f"tool span: {tool.name} -> {len(result)} chars")

        if api_key_available():
            text, calls = run_recommendation(summary)
            print(f"model spans: recommendation generated via {[c['name'] for c in calls]}")
            print(json.dumps({"recommendation": text}, indent=2))
        else:
            print("NVIDIA_API_KEY not set — tool span only, no model span.")

    print(f"flushed. Look for session '{session_id}' in the PRISM dashboard.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
