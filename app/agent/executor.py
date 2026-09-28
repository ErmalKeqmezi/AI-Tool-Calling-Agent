"""ToolExecutor: the only place where a tool request from the LLM becomes a Python call.

It never trusts the LLM: the tool name is looked up in the registry, the arguments are
parsed and validated against the tool's schema, and any exception is caught and returned
as a structured error so a single failure can never crash the agent.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from app.tools.base import Permission, ToolError, ToolRegistry, fail, ok
from app.tools.validation import validate

logger = logging.getLogger("agent.executor")

MAX_RESULT_CHARS = 20_000  # keep huge tool outputs from flooding the context window


class ToolExecutor:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def execute(self, name: Any, arguments: Any, approved: bool = False) -> dict[str, Any]:
        """Run one tool call. `approved` must be True for WRITE tools - only the agent sets it,
        after the user confirmed. Nothing the LLM outputs can set it."""
        # 1. Tool name must be a known string.
        if not isinstance(name, str) or name not in self.registry:
            logger.warning("Unknown tool requested: %r", name)
            return fail("unknown_tool", f"Tool {name!r} does not exist. Available tools: {self.registry.names()}")
        tool = self.registry.get(name)
        assert tool is not None

        # Defense in depth: even if a caller forgets the confirmation step, WRITE tools refuse.
        if tool.permission is Permission.WRITE and not approved:
            logger.warning("Blocked unapproved WRITE tool: %s", name)
            return fail("permission_required", f"Tool {name!r} requires user confirmation.")

        # 2. Arguments must be a JSON object (accept a JSON string, reject anything else).
        if arguments is None:
            arguments = {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                return fail("malformed_arguments", f"Arguments are not valid JSON: {exc.msg}")
        if not isinstance(arguments, dict):
            return fail("malformed_arguments", f"Arguments must be a JSON object, got {type(arguments).__name__}")

        # 3. Arguments must match the tool's schema (required, types, ranges, no extras).
        problems = validate(arguments, tool.parameters)
        if problems:
            logger.warning("Invalid arguments for %s: %s", name, problems)
            return fail("invalid_arguments", "; ".join(problems))

        # 4. Run it, timing the call and catching everything.
        logger.info("TOOL SELECTED: %s", name)
        logger.info("ARGUMENTS: %s", json.dumps(arguments, ensure_ascii=False))
        started = time.perf_counter()
        try:
            result = ok(tool.function(**arguments))
        except ToolError as exc:
            result = fail(exc.error_type, str(exc))
        except Exception as exc:  # noqa: BLE001 - a buggy tool must not take the agent down
            logger.exception("Tool %s crashed", name)
            result = fail("execution_error", f"{type(exc).__name__}: {exc}")
        elapsed_ms = (time.perf_counter() - started) * 1000

        result = _truncate(result)
        logger.info("TOOL RESULT (%s, %.0f ms): %s", name, elapsed_ms, _preview(result))
        return result


def _truncate(result: dict[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(result, ensure_ascii=False, default=str)
    if len(encoded) <= MAX_RESULT_CHARS:
        return result
    return {
        "success": result.get("success", False),
        "truncated": True,
        "result": encoded[:MAX_RESULT_CHARS] + "...",
    }


def _preview(result: dict[str, Any], limit: int = 300) -> str:
    text = json.dumps(result, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "..."
