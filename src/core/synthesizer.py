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
5. LENGTH: 180-320 words.
</strict_rules>

<processing_pipeline>
Before generating your final response, you MUST think through your synthesis inside a <scratchpad> block.
1. List the active agents.
2. Extract key metrics.
3. Note conflicts.
4. Draft narrative flow.
</processing_pipeline>

<output_schema>
After your <scratchpad>, output ONLY a valid JSON object matching this exact schema. Do not include markdown code block formatting.
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


def synthesize(question: str, objective: str, reports: dict[str, str]) -> dict[str, Any]:
    """reports: {agent display name: report text (scratchpad already stripped)}."""
    llm = make_llm(temperature=0.15, max_tokens=7000)  # scratchpad + ~300-word JSON; 3000 truncated
    parts = [f"USER QUESTION:\n{question}", f"ROUTER OBJECTIVE:\n{objective or '(none)'}"]
    for name, text in reports.items():
        parts.append(f"=== REPORT FROM {name} ===\n{text}")
    # PRISM: traced as its own run inside the caller's session (see src/prism/tracing.py)
    config = {"callbacks": prism_callbacks(), "run_name": "core_synthesizer"}
    resp = llm.invoke([SystemMessage(SYNTH_PROMPT), HumanMessage("\n\n".join(parts))], config=config)
    raw = resp.content if isinstance(resp.content, str) else str(resp.content)
    out = parse_synthesis(raw)
    if not out["executive_summary"].strip():
        # reasoning budget exhausted: one nudge without the scratchpad requirement
        resp = llm.invoke([SystemMessage(SYNTH_PROMPT), HumanMessage("\n\n".join(parts)),
                           HumanMessage("Output the JSON object now. Skip the scratchpad.")],
                          config={**config, "run_name": "core_synthesizer_retry"})
        raw = resp.content if isinstance(resp.content, str) else str(resp.content)
        out = parse_synthesis(raw)
    out["_raw"] = raw
    return out
