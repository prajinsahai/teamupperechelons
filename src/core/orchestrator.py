"""bizmax Core: route -> run the selected tracks in parallel -> synthesize.

One PRISM session per question. Each phase is its own traced run inside that session
(router, one per sub-agent, synthesizer), so the trajectory in PRISM reads like the
MAGI panel: which nodes lit up, what each said, how they were merged.

Threads do not inherit contextvars, so the PRISM session and the ambient metadata are
carried into each worker with `contextvars.copy_context()`. Inside the worker each
sub-agent opens `with_metadata(its computed_*)` so its spans carry its own truth values;
the synthesizer span carries the union.

Budgets: every sub-agent gets AGENT_BUDGET_S. Agents that miss it are reported as
timed out and synthesis proceeds on what arrived. (Their threads finish in the
background; the LLM client timeout bounds how long that takes.)
"""

from __future__ import annotations

import contextvars
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable

from src.core.router import KEY_TO_AGENT, Route, route
from src.core.synthesizer import rapid_synthesis, strip_scratchpad, synthesize
from src.data.ingest import DataBundle
from src.eval.grounding import grounding
from src.llm.agent import run_agent
from src.prism.tracing import with_metadata
from src.tracks.registry import TRACKS

AGENT_BUDGET_S = 180.0  # per parallel batch; a hung NIM call must not freeze the UI

Status = str  # "idle" | "running" | "done" | "error" | "skipped"
OnUpdate = Callable[[str, dict[str, Status]], None]  # (phase, node statuses)


@dataclass
class AgentReport:
    key: str
    name: str
    text: str
    tool_calls: list[dict[str, Any]]  # {name, args, result, seconds}
    seconds: float
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)  # this agent's computed_* (as sent to PRISM)

    @property
    def tool_names(self) -> list[str]:
        return [c["name"] for c in self.tool_calls]


@dataclass
class CoreResult:
    session_id: str
    question: str
    route: Route
    reports: dict[str, AgentReport] = field(default_factory=dict)
    synthesis: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)  # union sent with the synthesizer span
    metadata_errors: list[str] = field(default_factory=list)
    timed_out: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    def statuses(self) -> dict[str, Status]:
        out: dict[str, Status] = {}
        for k in TRACKS:
            if k not in self.route.active:
                out[k] = "skipped"
            elif k in self.reports and self.reports[k].text and not self.reports[k].error:
                out[k] = "done"
            else:
                out[k] = "error"
        return out

    def grounding(self) -> dict[str, Any]:
        """Executive summary numbers vs every sub-agent's tool results and report text."""
        records = [c for r in self.reports.values() for c in r.tool_calls]
        return grounding(self.synthesis.get("executive_summary", ""), records, [r.text for r in self.reports.values()])


def track_metadata(key: str, bundle: DataBundle, params: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """One track's computed_* truth values, or ({}, error)."""
    try:
        return {"track": key, **TRACKS[key].metadata(bundle, params)}, ""
    except Exception as e:  # a metadata failure must not sink the agent
        return {"track": key}, f"{key}: {e}"[:200]


def _run_one(key: str, bundle: DataBundle, params: dict[str, Any], question: str) -> AgentReport:
    track = TRACKS[key]
    t0 = time.time()
    meta, meta_err = track_metadata(key, bundle, params)
    try:
        tools = track.build_tools(bundle, params)
        with with_metadata(meta):  # this agent's spans carry its own computed_*
            text, calls = run_agent(track.system_prompt, question, tools, run_name=f"{key}_track")
        rep = AgentReport(key, track.name, strip_scratchpad(text), calls, time.time() - t0, metadata=meta)
        if meta_err:
            rep.error = ""  # the run itself succeeded; surface the metadata problem separately
            rep.metadata["metadata_error"] = meta_err
        return rep
    except Exception as e:  # one failing agent must not sink the run
        return AgentReport(key, track.name, "", [], time.time() - t0, error=str(e)[:300], metadata=meta)


def core_metadata(bundle: DataBundle, params: dict[str, Any], active: list[str], reports: dict[str, AgentReport] | None = None) -> tuple[dict[str, Any], list[str]]:
    """Union of the active tracks' computed_* (reusing what the agents already computed)."""
    meta: dict[str, Any] = {"mode": "auto", "active_tracks": ",".join(active)}
    errors: list[str] = []
    for k in active:
        if reports and k in reports and reports[k].metadata:
            m = dict(reports[k].metadata)
        else:
            m, err = track_metadata(k, bundle, params)
            if err:
                errors.append(err)
        for mk, mv in m.items():
            if mk in ("track", "metadata_error"):
                continue
            meta[mk if mk.startswith("computed_") else f"{k}_{mk}"] = mv
    return meta, errors


def run_core(
    question: str,
    bundle: DataBundle,
    params: dict[str, Any],
    session_id: str,
    on_update: OnUpdate | None = None,
    use_llm_router: bool = True,
    max_workers: int = 4,
    agent_budget_s: float = AGENT_BUDGET_S,
    deep_synthesis: bool = False,
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
    for k in TRACKS:
        statuses[k] = "running" if k in rt.active else "skipped"
    notify("processing", statuses)

    # 2. parallel sub-agents, bounded by a budget (contextvars copied so the PRISM session follows)
    t0 = time.time()
    ctx = contextvars.copy_context()
    pool = ThreadPoolExecutor(max_workers=max_workers)
    futures = {pool.submit(ctx.copy().run, _run_one, k, bundle, params, question): k for k in rt.active}
    try:
        for fut in as_completed(futures, timeout=agent_budget_s):
            k = futures[fut]
            rep = fut.result()
            result.reports[k] = rep
            statuses[k] = "error" if rep.error or not rep.text else "done"
            notify("processing", statuses)
    except FuturesTimeout:
        for fut, k in futures.items():
            if k not in result.reports:
                result.reports[k] = AgentReport(k, TRACKS[k].name, "", [], agent_budget_s, error=f"timed out after {agent_budget_s:.0f}s")
                result.timed_out.append(k)
                statuses[k] = "error"
        notify("processing", statuses)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)  # abandoned workers finish in the background
    result.timings["processing"] = time.time() - t0

    # 3. synthesize (span carries the union of computed_* truth values)
    notify("synthesizing" if deep_synthesis else "assembling", statuses)
    t0 = time.time()
    result.metadata, result.metadata_errors = core_metadata(bundle, params, rt.active, result.reports)
    result.metadata["synthesis_mode"] = "deep" if deep_synthesis else "rapid"
    good = {KEY_TO_AGENT[k]: r.text for k, r in result.reports.items() if r.text}
    if good:
        if deep_synthesis:
            try:
                with with_metadata(result.metadata):
                    result.synthesis = synthesize(question, rt.objective, good)
            except Exception as e:
                result.synthesis = rapid_synthesis(question, good)
                result.synthesis["_error"] = f"Deep synthesis unavailable; rapid brief used: {e}"[:300]
        else:
            result.synthesis = rapid_synthesis(question, good)
    else:
        result.synthesis = {"executive_summary": "No sub-agent produced a report.", "active_agents_cited": [], "strategic_risk_friction": "",
                            "key_figures": {}, "_parse": "none"}
    result.timings["synthesizing"] = time.time() - t0
    notify("complete", statuses)
    return result
