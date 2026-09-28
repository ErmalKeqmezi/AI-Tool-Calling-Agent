"""A scripted stand-in for the LLM so the agent loop can be tested without an API key."""

from __future__ import annotations

import copy
from typing import Any

from app.llm.client import LLMError, LLMResponse, ToolCall


def text(t: str, stop_reason: str = "end_turn") -> LLMResponse:
    return LLMResponse(text=t, tool_calls=[], stop_reason=stop_reason, content=[{"type": "text", "text": t}])


def tools(*calls: tuple[str, Any], preamble: str = "") -> LLMResponse:
    """tools(("calculate", {"expression": "1+1"}), ...) -> an LLM response requesting those tools."""
    tool_calls = [ToolCall(id=f"toolu_{i}_{name}", name=name, arguments=args) for i, (name, args) in enumerate(calls)]
    content: list[dict[str, Any]] = [{"type": "text", "text": preamble}] if preamble else []
    content += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments} for c in tool_calls]
    return LLMResponse(text=preamble, tool_calls=tool_calls, stop_reason="tool_use", content=content)


class FakeLLM:
    """Returns the scripted responses in order. An LLMError in the script is raised instead."""

    def __init__(self, *script: LLMResponse | LLMError):
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []  # what the agent sent on each call

    def chat(self, messages, tools=None, system=None) -> LLMResponse:
        self.calls.append({"messages": copy.deepcopy(messages), "tools": tools, "system": system})
        if not self.script:
            raise AssertionError("FakeLLM ran out of scripted responses")
        item = self.script.pop(0)
        if isinstance(item, LLMError):
            raise item
        return item

    def last_tool_results(self, call_index: int = -1) -> list[dict[str, Any]]:
        """The tool_result blocks the agent sent to the LLM on a given call."""
        last_msg = self.calls[call_index]["messages"][-1]
        return [b for b in last_msg["content"] if isinstance(b, dict) and b.get("type") == "tool_result"]
