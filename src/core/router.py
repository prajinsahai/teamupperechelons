"""Master router: which sub-agents does this question need?

LLM routing with strict JSON output (the blueprint's Master Router prompt), with a
deterministic keyword router as the fallback so the system always routes. The
Actuary is locked in; 1-3 additional agents are chosen.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm.config import make_llm
from src.prism.tracing import callbacks as prism_callbacks

AGENT_KEYS = {"Actuary": "actuary", "Claims_Analyst": "claims", "Reinsurance_Manager": "reinsurance",
              "Capital_Manager": "capital", "Risk_Manager": "risk", "Policy_Analyst": "policy"}
KEY_TO_AGENT = {v: k for k, v in AGENT_KEYS.items()}

ROUTER_PROMPT = """You are the central orchestration engine for an elite corporate risk and insurance AI. You have access to six specialized sub-agents: Actuary, Claims_Analyst, Reinsurance_Manager, Capital_Manager, Risk_Manager, and Policy_Analyst.
Your Task: Analyze the incoming user query and determine which sub-agents must be activated to provide a comprehensive financial answer.
Strict Routing Rule: The Actuary is the foundational quantitative engine of this platform. It is permanently locked into the active state and cannot be cut from any routing decision. Select 1 to 3 additional agents based on the query's focus.
Agent remits: Claims_Analyst = claim patterns, leakage, fraud, litigation, adverse development, social inflation. Reinsurance_Manager = retentions, attachment points, limits, treaties, quota share, excess of loss, stop-loss, transfer cost. Capital_Manager = capital, solvency, SCR, liquidity, shortfall, balance sheet. Risk_Manager = exposure movement, concentration, emerging risks, stress tests, VaR, black swans. Policy_Analyst = policy wording, exclusions, sub-limits, deductibles, waiting periods, coverage gaps, probability of payout.
Output Format: You must output ONLY a valid JSON object in the following format, with no prose before or after.
{"active_nodes": ["Actuary", "Risk_Manager", "Capital_Manager"], "synthesized_objective": "Brief summary of the required analysis."}"""

KEYWORDS: dict[str, list[str]] = {
    "claims": ["claim", "leakage", "fraud", "litigat", "adverse", "social inflation", "severity trend", "frequency trend", "anomal", "large loss", "duplicate", "overpaid"],
    "reinsurance": ["reinsur", "retention", "attachment", "limit", "treaty", "quota", "excess of loss", "xol", "stop-loss", "stop loss", "cede", "layer", "transfer", "tcor", "cost of risk"],
    "capital": ["capital", "solven", "scr", "liquidity", "shortfall", "balance sheet", "own funds", "surplus", "1-in-200", "1 in 200"],
    "risk": ["exposure", "concentrat", "emerging", "stress", "black swan", "scenario", "var", "value at risk", "catastroph", "weather", "accumulat", "monte carlo"],
    "policy": ["policy", "wording", "exclu", "sub-limit", "sublimit", "deductible", "waiting period", "coverage", "cover", "endorsement", "warranty", "payout", "trigger", "clause"],
}


@dataclass
class Route:
    active: list[str]  # track keys, actuary first
    objective: str
    source: str  # "llm" | "keywords"
    raw: str = ""
    error: str = ""


def keyword_route(question: str, has_documents: bool) -> Route:
    q = question.lower()
    scores = {k: sum(q.count(w) for w in ws) for k, ws in KEYWORDS.items()}
    if not has_documents:
        scores["policy"] = 0
    ranked = [k for k, v in sorted(scores.items(), key=lambda kv: -kv[1]) if v > 0][:3]
    if not ranked:
        ranked = ["risk"]  # a generic "how are we doing" question
    return Route(active=["actuary"] + ranked, objective=f"Keyword routing on: {', '.join(ranked)}", source="keywords")


def _parse_route(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in router output")
    return json.loads(m.group(0))


def route(question: str, has_documents: bool, use_llm: bool = True) -> Route:
    """LLM route with strict validation; keyword fallback on any failure."""
    fallback = keyword_route(question, has_documents)
    if not use_llm:
        return fallback
    try:
        model = os.environ.get("NVIDIA_ROUTER_MODEL")  # optional fast model
        llm = make_llm(temperature=0.0, model=model, max_tokens=2000)  # reasoning tokens count; 400 returned nothing
        # PRISM: traced as its own run inside the caller's session (see src/prism/tracing.py)
        resp = llm.invoke([SystemMessage(ROUTER_PROMPT), HumanMessage(question)],
                          config={"callbacks": prism_callbacks(), "run_name": "core_router"})
        raw = resp.content if isinstance(resp.content, str) else str(resp.content)
        data = _parse_route(raw)
        nodes = [n for n in data.get("active_nodes", []) if n in AGENT_KEYS]
        keys = ["actuary"] + [AGENT_KEYS[n] for n in nodes if AGENT_KEYS[n] != "actuary"]
        if not has_documents:
            keys = [k for k in keys if k != "policy"]
        keys = list(dict.fromkeys(keys))[:4]  # actuary + up to 3
        if len(keys) < 2:
            keys = fallback.active
        return Route(active=keys, objective=str(data.get("synthesized_objective", ""))[:300], source="llm", raw=raw[:500])
    except Exception as e:
        fallback.error = f"router LLM failed, used keywords: {e}"[:200]
        return fallback
