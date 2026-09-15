"""bizmax Core: route -> run the selected tracks in parallel -> synthesize.

One PRISM session per question. Each phase is its own traced run inside that session
(router, one per sub-agent, synthesizer), so the trajectory in PRISM reads like the
MAGI panel: which nodes lit up, what each said, how they were merged.

Threads do not inherit contextvars, so the PRISM session and the ambient computed_*
metadata are carried into each worker with `contextvars.copy_context()`.
"""

from __future__ import annotations

import contextvars
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable

from src.core.router import KEY_TO_AGENT, Route, route
from src.core.synthesizer import strip_scratchpad, synthesize
from src.data.ingest import DataBundle
from src.llm.agent import run_agent
from src.tracks.registry import TRACKS

Status = str  # "idle" | "running" | "done" | "error" | "skipped"
OnUpdate = Callable[[str, dict[str, Status]], None]  # (phase, node statuses)


@dataclass
class AgentReport:
    key: str
    name: str
    text: str
    tool_calls: list[str]
    seconds: float
    error: str = ""


@dataclass
class CoreResult:
    session_id: str
    question: str
    route: Route
    reports: dict[str, AgentReport] = field(default_factory=dict)
    synthesis: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)


def _run_one(key: str, bundle: DataBundle, params: dict[str, Any], question: str) -> AgentReport:
    track = TRACKS[key]
    t0 = time.time()
    try:
        tools = track.build_tools(bundle, params)
        text, calls = run_agent(track.system_prompt, question, tools, run_name=f"{key}_track")
        return AgentReport(key, track.name, strip_scratchpad(text), [c["name"] for c in calls], time.time() - t0)
    except Exception as e:  # one failing agent must not sink the run
        return AgentReport(key, track.name, "", [], time.time() - t0, error=str(e)[:300])


def run_core(
    question: str,
    bundle: DataBundle,
    params: dict[str, Any],
    session_id: str,
    on_update: OnUpdate | None = None,
    use_llm_router: bool = True,
    max_workers: int = 4,
) -> CoreResult:
    """Full auto-mode run. Call inside `analysis_run(session_id, ...)` so it is traced."""
    notify = on_update or (lambda phase, st: None)
    statuses: dict[str, Status] = {k: "idle" for k in TRACKS}
    has_docs = bool(bundle.documents)
    result = CoreResult(session_id=session_id, question=question, route=Route([], "", ""))

    # 1. route
    notify("routing", statuses)
    t0 = time.time()
    rt = route(question, has_docs, use_llm=use_llm_router)
    result.route = rt
    result.timings["routing"] = time.time() - t0
    for k in rt.active:
        statuses[k] = "running"
    for k in TRACKS:
        if k not in rt.active:
            statuses[k] = "skipped"
    notify("processing", statuses)

    # 2. parallel sub-agents (contextvars copied so PRISM session + metadata follow)
    t0 = time.time()
    ctx = contextvars.copy_context()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(ctx.copy().run, _run_one, k, bundle, params, question): k for k in rt.active}
        for fut in as_completed(futures):
            k = futures[fut]
            rep = fut.result()
            result.reports[k] = rep
            statuses[k] = "error" if rep.error or not rep.text else "done"
            notify("processing", statuses)
    result.timings["processing"] = time.time() - t0

    # 3. synthesize
    notify("synthesizing", statuses)
    t0 = time.time()
    good = {KEY_TO_AGENT[k]: r.text for k, r in result.reports.items() if r.text}
    if good:
        try:
            result.synthesis = synthesize(question, rt.objective, good)
        except Exception as e:
            first = next(iter(good.values()))
            result.synthesis = {"executive_summary": first, "active_agents_cited": list(good), "strategic_risk_friction": "",
                                "key_figures": {}, "_parse": "fallback", "_error": str(e)[:300]}
    else:
        result.synthesis = {"executive_summary": "No sub-agent produced a report.", "active_agents_cited": [], "strategic_risk_friction": "",
                            "key_figures": {}, "_parse": "none"}
    result.timings["synthesizing"] = time.time() - t0
    notify("complete", statuses)
    return result


def core_metadata(bundle: DataBundle, params: dict[str, Any], active: list[str]) -> dict[str, Any]:
    """Union of the active tracks' computed_* truth values, for the PRISM session."""
    meta: dict[str, Any] = {"mode": "auto", "active_tracks": ",".join(active)}
    for k in active:
        try:
            for mk, mv in TRACKS[k].metadata(bundle, params).items():
                meta[mk if mk.startswith("computed_") else f"{k}_{mk}"] = mv
        except Exception:
            pass
    return meta


def grounding_check(text: str, reports: dict[str, AgentReport], tool_outputs: list[str] | None = None) -> dict[str, Any]:
    """Every number in the synthesis must appear in some sub-agent report (or tool output)."""
    hay = " ".join([r.text for r in reports.values()] + (tool_outputs or [])).replace(",", "")
    nums = re.findall(r"\d[\d,]*\.?\d*", text)
    missing = [n for n in nums if n.replace(",", "").rstrip(".") not in hay and n.replace(",", "").rstrip(".") not in {"1", "2", "3", "4", "5", "200"}]
    return {"numbers": len(nums), "ungrounded": missing, "grounded": not missing}
