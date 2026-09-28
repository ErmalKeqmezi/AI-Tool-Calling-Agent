"""Thin wrapper around the Anthropic Messages API.

The rest of the app never touches the SDK directly. It talks to `LLMClient.chat()`, which
returns a provider-neutral `LLMResponse` and raises only our own `LLMError` subclasses.
That keeps the agent loop readable and makes the LLM trivially fakeable in tests.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

import anthropic

from app.config.settings import Settings

logger = logging.getLogger("agent.llm")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LLMError(Exception):
    """Base class for every failure talking to the LLM."""

    retryable = False


class LLMAuthError(LLMError):
    """Missing / invalid API key or no permission."""


class LLMTimeoutError(LLMError):
    retryable = True


class LLMRateLimitError(LLMError):
    retryable = True


class LLMUnavailableError(LLMError):
    """Network failure, 5xx, or overloaded."""

    retryable = True


class LLMBadRequestError(LLMError):
    """The request itself was rejected (bad model id, malformed messages, ...)."""


class LLMInvalidResponseError(LLMError):
    """The API answered, but not with something we can use."""


# ---------------------------------------------------------------------------
# Normalized response
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    """One tool request from the LLM: which tool, with what arguments."""

    id: str
    name: str
    arguments: Any  # usually a dict - but the executor must not assume that


@dataclass
class LLMResponse:
    text: str  # concatenated text blocks ("" when the model only called tools)
    tool_calls: list[ToolCall]
    stop_reason: str | None
    # The assistant content exactly as returned, to append to history unchanged.
    # (It may contain thinking blocks that must be sent back as-is.)
    content: list[Any] = field(default_factory=list)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class ChatModel(Protocol):
    """Anything with this method can drive the agent (the real client or a test fake)."""

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
    ) -> LLMResponse: ...


# ---------------------------------------------------------------------------
# Real client
# ---------------------------------------------------------------------------


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = anthropic.Anthropic(
            api_key=settings.require_api_key(),  # raises ConfigError if missing
            timeout=settings.request_timeout,
            max_retries=settings.max_retries,  # SDK retries 408/409/429/5xx with backoff
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "max_tokens": self.settings.max_tokens,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        if system:
            kwargs["system"] = system
        if self.settings.refusal_fallback:
            # If the model declines on safety grounds, let the API retry on a fallback model.
            kwargs["extra_headers"] = {"anthropic-beta": "server-side-fallback-2026-07-01"}
            kwargs["extra_body"] = {"fallbacks": "default"}

        try:
            message = self._client.messages.create(**kwargs)
        # Most specific first: each category is handled differently by the caller.
        except anthropic.APITimeoutError as exc:
            raise LLMTimeoutError(f"LLM request timed out after {self.settings.request_timeout}s") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMUnavailableError(f"Could not reach the LLM API: {exc}") from exc
        except anthropic.AuthenticationError as exc:
            raise LLMAuthError("The API key was rejected (401). Check ANTHROPIC_API_KEY.") from exc
        except anthropic.PermissionDeniedError as exc:
            raise LLMAuthError(f"Permission denied (403): {exc.message}") from exc
        except anthropic.RateLimitError as exc:
            raise LLMRateLimitError("Rate limited by the LLM API (429) - retries exhausted.") from exc
        except anthropic.NotFoundError as exc:
            raise LLMBadRequestError(f"Not found (404) - is LLM_MODEL={self.settings.model!r} valid? {exc.message}") from exc
        except anthropic.BadRequestError as exc:
            raise LLMBadRequestError(f"Request rejected (400): {exc.message}") from exc
        except anthropic.InternalServerError as exc:
            raise LLMUnavailableError(f"LLM API server error ({exc.status_code}).") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"LLM API error ({exc.status_code}): {exc.message}") from exc

        return self._normalize(message)

    @staticmethod
    def _normalize(message: Any) -> LLMResponse:
        content = getattr(message, "content", None)
        if not isinstance(content, list):
            raise LLMInvalidResponseError("LLM response has no content list.")

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(block.text)
            elif block_type == "tool_use":
                arguments = block.input
                if isinstance(arguments, str):  # defensive: should already be a dict
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        pass  # leave as a string; the executor reports it as malformed
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=arguments))
            # thinking blocks etc. are kept in `content` but not shown to the user

        stop_reason = getattr(message, "stop_reason", None)
        logger.debug("LLM stop_reason=%s tool_calls=%d", stop_reason, len(tool_calls))
        return LLMResponse(
            text="".join(text_parts).strip(),
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            content=list(content),
        )
