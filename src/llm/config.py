"""Single source of truth for the LLM provider, model and prompts.

Provider: NVIDIA NIM through its OpenAI-compatible endpoint
(https://integrate.api.nvidia.com/v1). Env: NVIDIA_API_KEY, NVIDIA_BASE_URL,
optional NVIDIA_MODEL.
"""

import os

from langchain_core.language_models import BaseChatModel

DEFAULT_MODEL = "meta/muse-glimmer-30b"
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
MAX_TOKENS = 4096  # NIM models spend hidden reasoning tokens; 1024 truncated to empty output


def model_name() -> str:
    return os.environ.get("NVIDIA_MODEL", DEFAULT_MODEL)


def api_key_available() -> bool:
    return bool(os.environ.get("NVIDIA_API_KEY"))


def make_llm(temperature: float = 0.1, model: str | None = None, max_tokens: int | None = None) -> BaseChatModel:
    """NVIDIA NIM chat model. Imported lazily so the dashboard loads without the key.

    `model` / `max_tokens` overrides exist for the router (small, fast) and the
    synthesizer (long output); everything else uses the defaults.
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model or model_name(),
        api_key=os.environ["NVIDIA_API_KEY"],
        base_url=os.environ.get("NVIDIA_BASE_URL", DEFAULT_BASE_URL),
        max_tokens=max_tokens or MAX_TOKENS,
        temperature=temperature,
        timeout=240,  # tool-heavy turns under 4-way concurrency exceeded 180s; 600 froze the UI on a hang
        max_retries=1,  # never retry a 4-minute call on stage
    )


SYSTEM_PROMPT = (
    "You are an AI Actuary. Analyze the provided metrics (severity spikes, IBNR, "
    "reinsurance limits) and write a concise, professional 3-sentence recommendation on "
    "whether reserves need adjustment and how reinsurance is protecting the portfolio.\n\n"
    "Rules: never calculate, estimate or invent a number yourself. Every figure you cite "
    "must come verbatim from the get_actuarial_summary tool. Call the tool first, then answer."
)

USER_PROMPT = (
    "Review the current portfolio and give your recommendation. "
    "Read the metrics with the tool before writing."
)
