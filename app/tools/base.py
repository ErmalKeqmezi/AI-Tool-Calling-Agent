"""Core tool abstractions: the Tool definition, permissions, results, and the registry."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable


class Permission(str, Enum):
    READ = "read"  # no side effects - runs automatically
    WRITE = "write"  # side effects (files, emails, ...) - needs user confirmation


class ToolError(Exception):
    """Raised by a tool implementation for an *expected* failure (bad input, city not found...).

    The executor turns it into {"success": false, "error": {...}} for the LLM.
    """

    def __init__(self, message: str, error_type: str = "tool_error"):
        super().__init__(message)
        self.error_type = error_type


def ok(result: Any) -> dict[str, Any]:
    return {"success": True, "result": result}


def fail(error_type: str, message: str) -> dict[str, Any]:
    return {"success": False, "error": {"type": error_type, "message": message}}


@dataclass(frozen=True)
class Tool:
    """A formal tool definition: what the LLM sees (name/description/parameters)
    plus what Python runs (function) and how dangerous it is (permission)."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for the arguments object
    function: Callable[..., Any]
    permission: Permission = Permission.READ
    title: str = ""  # human-friendly name for UIs, e.g. "Calculator" (not sent to the LLM)

    @property
    def display_name(self) -> str:
        return self.title or self.name.replace("_", " ").title()

    def to_llm_schema(self) -> dict[str, Any]:
        """Anthropic's wire format calls the parameter schema `input_schema`."""
        return {"name": self.name, "description": self.description, "input_schema": self.parameters}


class ToolRegistry:
    """Central name -> Tool lookup. The agent and executor never hardcode tool names."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} is already registered")
        if tool.parameters.get("type") != "object":
            raise ValueError(f"Tool {tool.name!r}: parameters schema must have type 'object'")
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    def schemas(self) -> list[dict[str, Any]]:
        """Tool definitions to send to the LLM (stable order)."""
        return [tool.to_llm_schema() for tool in self._tools.values()]

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
