"""One LLM (NVIDIA NIM via langchain-openai) with one tool and a plain manual tool-calling loop.

No agent framework. The LLM reads engine/ML outputs through `get_actuarial_summary`
and never computes anything itself (CLAUDE.md Core Rule 1). Every model and tool
invoke carries the PRISM callback so the whole run is traced.
"""

import json
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, tool

from src.llm.config import SYSTEM_PROMPT, USER_PROMPT, api_key_available, make_llm  # noqa: F401  (re-exported)
from src.prism.tracing import callbacks as prism_callbacks

MAX_TOOL_ROUNDS = 5


def make_summary_tool(summary: dict[str, Any]) -> BaseTool:
    """Build a tool that exposes the already-computed engine + ML outputs to the LLM."""

    @tool
    def get_actuarial_summary() -> str:
        """Return the deterministic actuarial engine outputs (portfolio loss, IBNR,
        reinsurance recovery, net exposure) and the ML claims-intelligence outputs
        (severity trend, anomaly counts) for the loaded claims portfolio, as JSON.
        All figures are computed in Python; quote them exactly."""
        return json.dumps(summary, default=float)

    return get_actuarial_summary


def run_recommendation(summary: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Run the tool loop and return (recommendation_text, tool_calls_made)."""
    summary_tool = make_summary_tool(summary)
    tools_by_name = {summary_tool.name: summary_tool}

    llm = make_llm().bind_tools(list(tools_by_name.values()))

    # PRISM: every model call and tool call is traced. The caller opens the
    # session (see src/prism/tracing.py) so all spans share one trajectory.
    config = {"callbacks": prism_callbacks()}

    messages: list[BaseMessage] = [SystemMessage(SYSTEM_PROMPT), HumanMessage(USER_PROMPT)]
    calls_made: list[dict[str, Any]] = []

    for _ in range(MAX_TOOL_ROUNDS):
        response: AIMessage = llm.invoke(messages, config=config)
        messages.append(response)
        if not response.tool_calls:
            text = _text_of(response)
            if not text and calls_made:
                # Seen with NIM reasoning models: budget spent thinking, no text emitted.
                # One nudge, same traced session, then give up honestly.
                messages.append(HumanMessage("Write the 3-sentence recommendation now, using only the tool figures."))
                text = _text_of(llm.invoke(messages, config=config))
            return text, calls_made
        for call in response.tool_calls:
            result = tools_by_name[call["name"]].invoke(call["args"], config=config)
            calls_made.append({"name": call["name"], "args": call["args"]})
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

    raise RuntimeError("LLM did not finish within the tool-call limit")


def _text_of(msg: AIMessage) -> str:
    if isinstance(msg.content, str):
        return msg.content.strip()
    return "".join(
        block.get("text", "") for block in msg.content if isinstance(block, dict) and block.get("type") == "text"
    ).strip()
