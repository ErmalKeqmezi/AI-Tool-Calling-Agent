"""Pure formatting helpers for the Streamlit UI (no Streamlit import, so they're easy to test).

They turn the agent's structured output into short, user-safe text. Raw exception details
stay in the backend logs; the user only sees friendly messages.
"""

from __future__ import annotations

import json
from typing import Any

_UNREACHABLE = "⚠️ Unable to contact the AI model. Please try again."

# AgentReply.error -> message shown instead of the raw reply text
LLM_ERROR_MESSAGES = {
    "LLMTimeoutError": _UNREACHABLE,
    "LLMUnavailableError": _UNREACHABLE,
    "LLMError": _UNREACHABLE,
    "LLMRateLimitError": "⚠️ The AI model is busy right now (rate limited). Please wait a moment and try again.",
    "LLMAuthError": "⚠️ The AI model rejected the API key. Check ANTHROPIC_API_KEY in your .env file.",
    "LLMBadRequestError": "⚠️ I couldn't process that request.",
    "LLMInvalidResponseError": "⚠️ I couldn't process that request.",
    "unexpected": "⚠️ Something went wrong while handling your request. Please try again.",
}


def friendly_error(error: str | None) -> str | None:
    """User-facing text for an agent-level error, or None if the reply text is fine to show."""
    if error is None:
        return None
    return LLM_ERROR_MESSAGES.get(error)  # max_iterations / max_tokens: the reply text already explains


def tool_status(call: dict[str, Any]) -> tuple[str, str]:
    """(icon, label) for a finished tool call."""
    result = call.get("result") or {}
    if result.get("success"):
        return "✓", "Completed"
    if (result.get("error") or {}).get("type") == "permission_denied":
        return "⊘", "Declined"
    return "✗", "Failed"


def tool_failure_message(display_name: str, result: dict[str, Any]) -> str:
    """Short explanation of a failed tool call. Never includes a stack trace."""
    error = result.get("error") or {}
    error_type = error.get("type", "")
    if error_type == "execution_error":
        detail = "An internal error occurred inside the tool."  # details are in the server log
    else:
        detail = error.get("message") or "No details available."
    return f"⚠️ The {display_name.lower()} tool failed to return a result. {detail}"


def summarize_arguments(arguments: Any, limit: int = 70) -> str:
    """'expression: 25 * 17' style one-liner for the collapsed tool header."""
    if isinstance(arguments, dict):
        text = ", ".join(f"{k}: {_short(v)}" for k, v in arguments.items())
    else:
        text = _short(arguments)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _short(value: Any) -> str:
    if isinstance(value, str):
        return value.replace("\n", " ")
    return json.dumps(value, ensure_ascii=False, default=str)


def escape_markdown(text: str) -> str:
    """Streamlit renders $...$ as LaTeX; '$5 and $10' should stay plain text."""
    return text.replace("$", "\\$")
