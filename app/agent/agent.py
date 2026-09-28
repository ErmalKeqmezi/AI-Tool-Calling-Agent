"""The agent: conversation memory + the LLM <-> tool loop + the permission gate.

    user message
        -> LLM (with tool definitions)
        -> tool calls?  no  -> final answer
                        yes -> executor runs each call -> tool results appended -> LLM again
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from app.agent.executor import ToolExecutor
from app.llm.client import ChatModel, LLMError, LLMResponse, ToolCall
from app.tools.base import Permission, ToolRegistry, fail

logger = logging.getLogger("agent.loop")

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful assistant that can call tools.

- Use a tool whenever it gives a more reliable answer than your own knowledge: arithmetic \
(calculate), current date/time (get_current_time), weather (get_weather), and recent or \
factual lookups (search_web). Answer directly when no tool is needed.
- A request may need several tools, or one tool's output as input to another - call them \
in whatever order the task requires.
- If a tool returns an error, read it: fix the arguments and retry if that makes sense, \
otherwise explain the problem to the user.
- Some tools change things (files). The application asks the user for confirmation before \
running those, so just call the tool - do not ask for permission yourself. If a tool result \
says the user declined, respect that and do not retry it.
- Use the earlier conversation to resolve follow-up questions like "what about tomorrow?".
- Keep answers concise."""

_YES = {"y", "yes", "yeah", "yep", "sure", "ok", "okay", "confirm", "confirmed", "approve",
        "approved", "go ahead", "do it", "proceed", "yes please", "allow"}
_NO = {"n", "no", "nope", "cancel", "stop", "deny", "denied", "don't", "do not", "no thanks", "abort"}


@dataclass
class PendingConfirmation:
    """Tool calls waiting for the user's yes/no before they may run."""

    calls: list[ToolCall]  # all calls from the assistant message, in order
    results: dict[str, dict[str, Any]]  # tool_use_id -> result, for calls already executed
    awaiting: list[ToolCall]  # the WRITE calls that need approval

    def describe(self) -> list[dict[str, Any]]:
        return [{"name": c.name, "arguments": c.arguments} for c in self.awaiting]


@dataclass
class AgentReply:
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # what ran during this turn
    pending_confirmation: list[dict[str, Any]] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "response": self.text,
            "tool_calls": self.tool_calls,
            "pending_confirmation": self.pending_confirmation,
            "error": self.error,
        }


@dataclass
class AgentEvent:
    """Progress notifications for UIs. Emitted only for things that actually happen.

    type is one of:
      "thinking"              data: {"iteration": int}              - an LLM call is starting
      "tool_start"            data: {"name", "arguments"}           - a tool is about to run
      "tool_end"              data: {"name", "arguments", "result"} - a tool finished (or was declined)
      "confirmation_required" data: {"calls": [{"name", "arguments"}]}
    """

    type: str
    data: dict[str, Any] = field(default_factory=dict)


EventHandler = Callable[[AgentEvent], None]


class Agent:
    def __init__(
        self,
        llm: ChatModel,
        registry: ToolRegistry,
        executor: ToolExecutor | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_iterations: int = 10,
        max_history_turns: int = 20,
        auto_approve_writes: bool = False,
    ):
        self.llm = llm
        self.registry = registry
        self.executor = executor or ToolExecutor(registry)
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.max_history_turns = max_history_turns
        self.auto_approve_writes = auto_approve_writes
        self.messages: list[dict[str, Any]] = []  # short-term memory (Anthropic message format)
        self.pending: PendingConfirmation | None = None
        self._on_event: EventHandler | None = None

    # ------------------------------------------------------------------ public

    def chat(self, user_message: str, on_event: EventHandler | None = None) -> AgentReply:
        """Handle one user message. `on_event` (optional) receives live progress events."""
        logger.info("USER: %s", user_message)
        trace: list[dict[str, Any]] = []
        self._on_event = on_event

        try:
            if self.pending is not None:
                reply = self._resume_after_confirmation(user_message, trace)
            else:
                self._trim_history()
                self.messages.append({"role": "user", "content": user_message})
                reply = self._run_loop(trace)
        finally:
            self._on_event = None

        logger.info("FINAL: %s", reply.text)
        return reply

    def reset(self) -> None:
        self.messages.clear()
        self.pending = None

    # ------------------------------------------------------------------ the loop

    def _run_loop(self, trace: list[dict[str, Any]]) -> AgentReply:
        for iteration in range(1, self.max_iterations + 1):
            self._emit("thinking", iteration=iteration)
            try:
                response = self.llm.chat(self.messages, tools=self.registry.schemas(), system=self.system_prompt)
            except LLMError as exc:
                logger.error("LLM ERROR (%s): %s", type(exc).__name__, exc)
                # Keep history valid (it must alternate user/assistant) for the next turn.
                self._append_assistant_text(f"[The previous request failed: {type(exc).__name__}]")
                return AgentReply(
                    text=f"Sorry, I couldn't reach the language model ({exc}). Please try again.",
                    tool_calls=trace,
                    error=type(exc).__name__,
                )

            self._log_response(iteration, response)

            # --- No tool calls: this is the final answer. ---------------------
            if not response.tool_calls:
                if response.stop_reason == "pause_turn":
                    self.messages.append({"role": "assistant", "content": response.content})
                    continue
                text = self._final_text(response)
                self.messages.append({"role": "assistant", "content": response.content or [{"type": "text", "text": text}]})
                return AgentReply(text=text, tool_calls=trace)

            # A tool_use block cut off by max_tokens can't be trusted - don't run it.
            if response.stop_reason == "max_tokens":
                text = (response.text + "\n\n" if response.text else "") + "[Response was cut off before the tool call completed.]"
                self._append_assistant_text(text)
                return AgentReply(text=text, tool_calls=trace, error="max_tokens")

            # --- Tool calls: record the request, run the tools, send results back.
            self.messages.append({"role": "assistant", "content": response.content})

            results: dict[str, dict[str, Any]] = {}
            awaiting: list[ToolCall] = []
            for call in response.tool_calls:
                tool = self.registry.get(call.name) if isinstance(call.name, str) else None
                if tool is not None and tool.permission is Permission.WRITE and not self.auto_approve_writes:
                    awaiting.append(call)  # held back until the user answers
                    continue
                results[call.id] = self._execute(call, trace, approved=self.auto_approve_writes)

            if awaiting:
                self.pending = PendingConfirmation(calls=response.tool_calls, results=results, awaiting=awaiting)
                logger.info("AWAITING CONFIRMATION: %s", json.dumps(self.pending.describe(), ensure_ascii=False))
                self._emit("confirmation_required", calls=self.pending.describe())
                return AgentReply(
                    text=self._confirmation_prompt(awaiting, response.text),
                    tool_calls=trace,
                    pending_confirmation=self.pending.describe(),
                )

            self.messages.append({"role": "user", "content": self._result_blocks(response.tool_calls, results)})
            # ...and loop: the LLM now sees the results and decides what to do next.

        logger.warning("Stopped after %d iterations without a final answer", self.max_iterations)
        text = "I had to stop because this request needed too many steps. Please try a simpler request."
        self._close_open_tool_turn()
        self._append_assistant_text(text)
        return AgentReply(text=text, tool_calls=trace, error="max_iterations")

    # ------------------------------------------------------------------ permissions

    def _resume_after_confirmation(self, user_message: str, trace: list[dict[str, Any]]) -> AgentReply:
        pending = self.pending
        assert pending is not None
        self.pending = None

        answer = user_message.strip().lower().rstrip(".!")
        approved = answer in _YES
        extra_text = None if (answer in _YES or answer in _NO) else user_message
        logger.info("CONFIRMATION: %s", "approved" if approved else "declined")

        for call in pending.awaiting:
            if approved:
                pending.results[call.id] = self._execute(call, trace, approved=True)
            else:
                result = fail("permission_denied", "The user declined this action. Do not retry it.")
                pending.results[call.id] = result
                trace.append({"name": call.name, "arguments": call.arguments, "result": result})
                self._emit("tool_end", name=call.name, arguments=call.arguments, result=result)

        content: list[dict[str, Any]] = self._result_blocks(pending.calls, pending.results)
        if extra_text:  # the user replied with something else - pass it along
            content.append({"type": "text", "text": extra_text})
        self.messages.append({"role": "user", "content": content})
        return self._run_loop(trace)

    @staticmethod
    def _confirmation_prompt(calls: list[ToolCall], model_text: str) -> str:
        lines = [model_text, ""] if model_text else []
        lines.append("This action requires your confirmation:")
        for call in calls:
            lines.append(f"  - {call.name}({json.dumps(call.arguments, ensure_ascii=False)})")
        lines.append("Reply 'yes' to proceed or 'no' to cancel.")
        return "\n".join(lines)

    # ------------------------------------------------------------------ helpers

    def _execute(self, call: ToolCall, trace: list[dict[str, Any]], approved: bool) -> dict[str, Any]:
        self._emit("tool_start", name=call.name, arguments=call.arguments)
        result = self.executor.execute(call.name, call.arguments, approved=approved)
        trace.append({"name": call.name, "arguments": call.arguments, "result": result})
        self._emit("tool_end", name=call.name, arguments=call.arguments, result=result)
        return result

    def _emit(self, event_type: str, **data: Any) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(AgentEvent(event_type, data))
        except Exception:  # noqa: BLE001 - a broken UI callback must never break the agent
            logger.exception("Event handler failed for %s", event_type)

    @staticmethod
    def _result_blocks(calls: list[ToolCall], results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        """One tool_result per tool_use, all in a single user message, same order as the calls."""
        return [
            {
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": json.dumps(results[call.id], ensure_ascii=False, default=str),
                "is_error": not results[call.id].get("success", False),
            }
            for call in calls
        ]

    @staticmethod
    def _final_text(response: LLMResponse) -> str:
        if response.stop_reason == "refusal":
            return response.text or "I can't help with that request."
        if response.stop_reason == "max_tokens":
            return (response.text + "\n\n[Response truncated: reached the token limit.]").strip()
        return response.text or "(no response)"

    def _append_assistant_text(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})

    def _close_open_tool_turn(self) -> None:
        """If history ends with an assistant tool_use turn, answer it so history stays valid."""
        if self.messages and self.messages[-1]["role"] == "user":
            return
        last = self.messages[-1] if self.messages else None
        calls = [b for b in (last or {}).get("content", []) if _block_type(b) == "tool_use"]
        if calls:
            self.messages.append({
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": _block_attr(b, "id"),
                     "content": json.dumps(fail("aborted", "Agent stopped.")), "is_error": True}
                    for b in calls
                ],
            })

    def _trim_history(self) -> None:
        """Keep the last `max_history_turns` user turns.

        We only cut at the start of a real user turn (plain text), never between an assistant
        tool_use and its tool_result - the API rejects a history that splits those pairs.
        """
        if self.max_history_turns <= 1:
            self.messages.clear()
            return
        starts = [i for i, m in enumerate(self.messages) if m["role"] == "user" and isinstance(m["content"], str)]
        if len(starts) >= self.max_history_turns:
            cut = starts[len(starts) - self.max_history_turns + 1]
            del self.messages[:cut]

    @staticmethod
    def _log_response(iteration: int, response: LLMResponse) -> None:
        logger.info(
            "LLM RESPONSE #%d: stop_reason=%s tools=%s text=%r",
            iteration,
            response.stop_reason,
            [c.name for c in response.tool_calls],
            response.text[:200],
        )


def _block_type(block: Any) -> str | None:
    return block.get("type") if isinstance(block, dict) else getattr(block, "type", None)


def _block_attr(block: Any, name: str) -> Any:
    return block.get(name) if isinstance(block, dict) else getattr(block, name, None)
