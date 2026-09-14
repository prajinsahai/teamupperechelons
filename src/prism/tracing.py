"""PRISM (Block Convey) live tracing for the AI Actuary.

One `PRISMtraceCallbackHandler` for the whole process (it owns an HTTP pool);
one `prismtrace.session(...)` per analysis run so every span of that run —
tool call, model call, chain — lands in a single trajectory.

The engine's computed figures are attached to every span as metadata
(`computed_expected_loss`, `computed_optimal_retention`, ...) so a PRISM
evaluator can mechanically compare what the engine computed against what the
narrative said. The stock handler does not forward LangChain's per-run
`metadata`, so `ActuaryTraceHandler` merges an ambient dict in at span start.

Env vars: PRISMTRACE_API_KEY, PRISMTRACE_PROJECT_ID, PRISMTRACE_HOST.
Tracing is optional: with no key set, `get_handler()` returns None and the app
runs untraced.
"""

from __future__ import annotations

import atexit
import contextvars
import os
import uuid
from contextlib import contextmanager
from typing import Any, Iterator, Optional

import prismtrace
from prismtrace import PRISMtraceCallbackHandler

AGENT_NAME = "ai-actuary"

_run_metadata: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "ai_actuary_run_metadata", default={}
)
_handler: Optional["ActuaryTraceHandler"] = None


class ActuaryTraceHandler(PRISMtraceCallbackHandler):
    """Stock handler + ambient per-run metadata on every span."""

    def _start_span(self, name: str, span_type: str, run_id: str, *args: Any, **kwargs: Any) -> None:
        ambient = _run_metadata.get()
        if ambient:
            kwargs["metadata"] = {**ambient, **(kwargs.get("metadata") or {})}
        super()._start_span(name, span_type, run_id, *args, **kwargs)


def tracing_enabled() -> bool:
    return bool(os.environ.get("PRISMTRACE_API_KEY")) and bool(os.environ.get("PRISMTRACE_PROJECT_ID"))


def get_handler() -> Optional[ActuaryTraceHandler]:
    """Process-wide handler, built once. None when PRISM env vars are absent."""
    global _handler
    if _handler is None and tracing_enabled():
        # key/project/host from env. 30s: the default 10s timed out against a cold backend
        # and the SDK fails open, silently dropping the batch.
        _handler = ActuaryTraceHandler(agent_name=AGENT_NAME, timeout=30)
        atexit.register(_handler.close)  # close() flushes first
    return _handler


def callbacks() -> list[ActuaryTraceHandler]:
    """Value for `config={"callbacks": ...}` on any LangChain invoke."""
    h = get_handler()
    return [h] if h else []


def new_session_id(prefix: str = "analysis") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def build_metadata(engine_output: dict[str, Any]) -> dict[str, Any]:
    """The computed_* truth values PRISM evaluators compare the narrative against."""
    intel = engine_output.get("claims_intelligence", {})
    return {
        "agent_id": AGENT_NAME,
        "computed_expected_loss": float(engine_output.get("ultimate_loss", 0.0)),
        "computed_portfolio_loss": float(engine_output.get("portfolio_loss", 0.0)),
        "computed_ibnr": float(engine_output.get("ibnr", 0.0)),
        "computed_expected_recovery": float(engine_output.get("expected_recovery", 0.0)),
        "computed_net_exposure": float(engine_output.get("net_exposure", 0.0)),
        "computed_optimal_retention": float(engine_output.get("reinsurance_detail", {}).get("retention", 0.0)),
        "claim_count": int(intel.get("claim_count", 0)),
        "data_years": int(len(intel.get("severity_by_year", {}))),
        "confidence_tier": "full_monte_carlo" if int(intel.get("claim_count", 0)) > 200 else "benchmark",
        "severity_change_pct": float(intel.get("severity_change_pct", 0.0)),
        "anomaly_count": int(intel.get("anomaly_count", 0)),
    }


@contextmanager
def analysis_run(session_id: str, metadata: dict[str, Any]) -> Iterator[str]:
    """Group everything inside under one PRISM session, with computed_* metadata on each span."""
    token = _run_metadata.set(dict(metadata))
    try:
        with prismtrace.session(session_id) as sid:
            yield sid
    finally:
        _run_metadata.reset(token)
        h = get_handler()
        if h:
            h.flush()  # the run is complete; send it now rather than at exit
