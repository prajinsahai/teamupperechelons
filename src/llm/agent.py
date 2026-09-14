"""One LLM (NVIDIA NIM via langchain-openai), a track's tools, and a plain manual tool loop.

No agent framework. The model chooses tools, reads their JSON, and writes the advice;
it never computes (CLAUDE.md Core Rule 1). The whole loop is ONE LangChain chain so
PRISM sees a single trace with the model and tool calls nested under it — the caller
opens the session (src/prism/tracing.py). Child invokes inherit the callbacks from the
root run; do not pass callbacks again inside, or spans double up.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import chain
from langchain_core.tools import BaseTool, tool

from src.llm.config import SYSTEM_PROMPT, USER_PROMPT, api_key_available, make_llm  # noqa: F401  (re-exported)
from src.prism.tracing import callbacks as prism_callbacks

MAX_TOOL_ROUNDS = 8


def make_summary_tool(summary: dict[str, Any]) -> BaseTool:
    """Single-tool wrapper used by the overview recommendation and the smoke test."""

    @tool
    def get_actuarial_summary() -> str:
        """Return the deterministic actuarial engine outputs (portfolio loss, IBNR,
        reinsurance recovery, net exposure) and the ML claims-intelligence outputs
        (severity trend, anomaly counts) for the loaded claims portfolio, as JSON.
        All figures are computed in Python; quote them exactly."""
        return json.dumps(summary, default=float)

    return get_actuarial_summary


def run_agent(
    system_prompt: str,
    question: str,
    tools: list[BaseTool],
    run_name: str = "ai_actuary_recommendation",
) -> tuple[str, list[dict[str, Any]]]:
    """Run the tool loop with the given role/tools. Returns (answer_text, tool_calls_made)."""
    tools_by_name = {t.name: t for t in tools}
    llm = make_llm().bind_tools(tools)

    @chain
    def _loop(_: Any) -> dict[str, Any]:
        messages: list[BaseMessage] = [SystemMessage(system_prompt), HumanMessage(question)]
        calls_made: list[dict[str, Any]] = []
        for _round in range(MAX_TOOL_ROUNDS):
            response: AIMessage = llm.invoke(messages)
            messages.append(response)
            if not response.tool_calls:
                text = _text_of(response)
                if not text and calls_made:
                    # Seen with NIM reasoning models: budget spent thinking, no text emitted.
                    messages.append(HumanMessage("Write the answer now in the required structure, using only the tool figures."))
                    text = _text_of(llm.invoke(messages))
                return {"text": text, "tool_calls": calls_made}
            for call in response.tool_calls:
                t = tools_by_name.get(call["name"])
                if t is None:
                    result = json.dumps({"error": f"unknown tool {call['name']}"})
                else:
                    try:
                        result = t.invoke(call["args"])
                    except Exception as e:  # tool errors go back to the model, not up the stack
                        result = json.dumps({"error": str(e)})
                calls_made.append({"name": call["name"], "args": call["args"]})
                messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        raise RuntimeError("LLM did not finish within the tool-call limit")

    out = _loop.with_config(run_name=run_name).invoke({}, config={"callbacks": prism_callbacks()})
    return out["text"], out["tool_calls"]


def run_recommendation(summary: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Overview recommendation: the original single-tool flow."""
    return run_agent(SYSTEM_PROMPT, USER_PROMPT, [make_summary_tool(summary)])


def _text_of(msg: AIMessage) -> str:
    if isinstance(msg.content, str):
        return msg.content.strip()
    return "".join(
        block.get("text", "") for block in msg.content if isinstance(block, dict) and block.get("type") == "text"
    ).strip()
