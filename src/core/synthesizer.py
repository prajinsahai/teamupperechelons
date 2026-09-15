"""Executive Synthesizer: merges the sub-agent reports into one C-suite summary.

The blueprint's prompt, with one deliberate change: chart payloads are NOT produced
by the LLM. Charts render from the deterministic engine (src/charts), so a chart can
never show a hallucinated number. The synthesizer returns the narrative, the agents it
cited, any "Strategic Risk Friction", and the key figures it relied on.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm.config import make_llm
from src.prism.tracing import callbacks as prism_callbacks

SYNTH_PROMPT = """<system_role>
You are the bizmax Core Executive Synthesizer, an elite financial orchestration AI. You act as the Chief Risk Officer summarizing highly technical actuarial and risk data for a C-suite audience.
</system_role>

<objective>
You will receive raw analytical reports from up to 4 specialized sub-agents, plus the user's question. Synthesize these reports into a single cohesive executive summary that answers the question.
</objective>

<strict_rules>
1. NO HALLUCINATIONS: Use ONLY the exact numbers provided in the sub-agent reports. Never compute new numbers.
2. TONE: Ruthlessly objective, highly technical, and institutional.
3. CONFLICT RESOLUTION: If sub-agents provide conflicting perspectives, explicitly highlight this tension as a "Strategic Risk Friction".
4. FORMATTING: Use Markdown for the text summary. Use bolding for key financial figures. Structure: a one-paragraph answer, then "Key findings" bullets, then "Recommended actions" bullets, then "Strategic Risk Friction" (or "None identified"), then "Confidence & caveats".
5. LENGTH: 120-220 words. Be direct and do not repeat the reports.
</strict_rules>

<output_schema>
Output ONLY a valid JSON object matching this exact schema. Do not include scratchpad text or markdown code block formatting.
{
  "executive_summary": "The full markdown-formatted synthesized response.",
  "active_agents_cited": ["Agent1", "Agent2"],
  "strategic_risk_friction": "One or two sentences, or 'None identified'.",
  "key_figures": {"label": "value as it appeared in a report", "...": "..."}
}
</output_schema>"""


def strip_scratchpad(text: str) -> str:
    return re.sub(r"<scratchpad>.*?</scratchpad>", "", text, flags=re.S | re.I).strip()


def parse_synthesis(text: str) -> dict[str, Any]:
    """Tolerant JSON extraction: strip scratchpad and code fences, take the outermost object."""
    body = strip_scratchpad(text)
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", body.strip(), flags=re.S)
    start, end = body.find("{"), body.rfind("}")
    if start == -1:
        return {"executive_summary": body, "active_agents_cited": [], "strategic_risk_friction": "", "key_figures": {}, "_parse": "no-json"}
    try:
        if end <= start:
            raise json.JSONDecodeError("truncated", body, len(body))
        data = json.loads(body[start:end + 1])
    except json.JSONDecodeError:
        # Truncated or malformed JSON (long answers hit the token cap): recover the
        # executive_summary string as far as it got, then the other fields if present.
        m = re.search(r'"executive_summary"\s*:\s*"(.*?)(?:"\s*,\s*"active_agents_cited"|$)', body, re.S)
        summary = m.group(1) if m else body
        summary = re.sub(r'"\s*,?\s*"?\w*$', "", summary).rstrip("\\")  # drop a dangling key fragment
        summary = summary.replace("\\n", "\n").replace('\\"', '"')
        agents = re.findall(r'"active_agents_cited"\s*:\s*\[(.*?)\]', body, re.S)
        cited = re.findall(r'"([^"]+)"', agents[0]) if agents else []
        fr = re.search(r'"strategic_risk_friction"\s*:\s*"(.*?)"', body, re.S)
        return {"executive_summary": summary.strip(), "active_agents_cited": cited,
                "strategic_risk_friction": fr.group(1) if fr else "", "key_figures": {}, "_parse": "recovered"}
    data.setdefault("executive_summary", ""); data.setdefault("active_agents_cited", [])
    data.setdefault("strategic_risk_friction", ""); data.setdefault("key_figures", {})
    data["_parse"] = "json"
    return data


_SECTION_NAMES = {
    "FINDINGS": "findings",
    "RECOMMENDATION": "recommendation",
    "RECOMMENDATIONS": "recommendation",
    "CONFIDENCE & CAVEATS": "confidence",
    "CONFIDENCE AND CAVEATS": "confidence",
}


def _heading(line: str) -> str | None:
    clean = re.sub(r"[#*_`]", "", line).strip().rstrip(":").strip().upper()
    return _SECTION_NAMES.get(clean)


def _section_lines(text: str, wanted: str, limit: int) -> list[str]:
    """Extract complete report bullets without changing any model-supplied figures."""
    active = False
    out: list[str] = []
    for raw in strip_scratchpad(text).splitlines():
        line = raw.strip()
        section = _heading(line)
        if section is not None:
            active = section == wanted
            continue
        if not active or not line:
            continue
        if wanted == "findings" and ("data loaded:" in line.lower() or "claims.csv" in line.lower() and "rows" in line.lower()):
            continue
        out.append(line if line.startswith(("-", "*")) else f"- {line}")
        if len(out) >= limit:
            break
    return out


def _fallback_lines(text: str, limit: int = 2) -> list[str]:
    lines: list[str] = []
    for raw in strip_scratchpad(text).splitlines():
        line = raw.strip()
        if not line or _heading(line) is not None:
            continue
        lines.append(line if line.startswith(("-", "*")) else f"- {line}")
        if len(lines) >= limit:
            break
    return lines


def rapid_synthesis(question: str, reports: dict[str, str]) -> dict[str, Any]:
    """Build an immediate grounded brief from completed model reports, with no extra LLM call.

    The sub-agents have already interpreted the tool output. This function only selects
    their complete findings and recommendations, so it adds no actuarial calculations or
    invented figures and normally completes in under a millisecond.
    """
    blocks = [
        "The routed specialists completed their tool-grounded analysis. "
        "Their decision points are consolidated below.",
    ]
    for name, report in reports.items():
        findings = _section_lines(report, "findings", 2)
        actions = _section_lines(report, "recommendation", 1)
        selected = findings + actions
        if not selected:
            selected = _fallback_lines(report)
        blocks.extend((f"### {name.replace('_', ' ')}", "\n".join(selected) or "- No displayable report was returned."))
    return {
        "executive_summary": "\n\n".join(blocks),
        "active_agents_cited": list(reports),
        "strategic_risk_friction": "",
        "key_figures": {},
        "_parse": "rapid",
        "_mode": "rapid",
        "_question": question,
        "_raw": "",
    }


def synthesize(question: str, objective: str, reports: dict[str, str]) -> dict[str, Any]:
    """Optional deep synthesis, bounded so it cannot hold the UI for several minutes."""
    llm = make_llm(
        temperature=0.1,
        max_tokens=4096,
        request_timeout=45,
        max_retries=0,
    )
    parts = [f"USER QUESTION:\n{question}", f"ROUTER OBJECTIVE:\n{objective or '(none)'}"]
    for name, text in reports.items():
        parts.append(f"=== REPORT FROM {name} ===\n{text}")
    # PRISM: traced as its own run inside the caller's session (see src/prism/tracing.py)
    config = {"callbacks": prism_callbacks(), "run_name": "core_synthesizer"}
    resp = llm.invoke([SystemMessage(SYNTH_PROMPT), HumanMessage("\n\n".join(parts))], config=config)
    raw = resp.content if isinstance(resp.content, str) else str(resp.content)
    out = parse_synthesis(raw)
    if not out["executive_summary"].strip():
        out = rapid_synthesis(question, reports)
        out["_error"] = "Deep synthesizer returned an empty response; rapid brief used."
    else:
        out["_mode"] = "deep"
    out["_raw"] = raw
    return out
