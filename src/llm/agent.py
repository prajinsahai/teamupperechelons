"""Claude via langchain-anthropic with one tool and a plain manual tool-calling loop.

No agent framework. The LLM reads engine/ML outputs through `get_actuarial_summary`
and never computes anything itself (CLAUDE.md Core Rule 1).
"""

import json
import os
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, tool

from src.llm.config import MAX_TOKENS, MODEL, SYSTEM_PROMPT, USER_PROMPT

MAX_TOOL_ROUNDS = 5


def api_key_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


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

    llm = ChatAnthropic(model=MODEL, max_tokens=MAX_TOKENS).bind_tools(list(tools_by_name.values()))

    messages: list[BaseMessage] = [SystemMessage(SYSTEM_PROMPT), HumanMessage(USER_PROMPT)]
    calls_made: list[dict[str, Any]] = []

    for _ in range(MAX_TOOL_ROUNDS):
        response: AIMessage = llm.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            return _text_of(response), calls_made
        for call in response.tool_calls:
            result = tools_by_name[call["name"]].invoke(call["args"])
            calls_made.append({"name": call["name"], "args": call["args"]})
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

    raise RuntimeError("LLM did not finish within the tool-call limit")


def _text_of(msg: AIMessage) -> str:
    if isinstance(msg.content, str):
        return msg.content.strip()
    return "".join(
        block.get("text", "") for block in msg.content if isinstance(block, dict) and block.get("type") == "text"
    ).strip()
