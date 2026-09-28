"""Streamlit UI for the tool-calling agent.

    streamlit run app.py

This file is only the UI: it keeps the chat in st.session_state, sends each message to the
existing Agent, and draws what the agent reports back (final text + real tool calls).
All LLM and tool logic stays in app/agent and app/tools.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import streamlit as st

from app.agent import Agent, AgentEvent, create_agent
from app.config.settings import ConfigError, Settings, get_settings
from app.logging_config import setup_logging
from app.tools import Permission, ToolRegistry, build_registry
from app.ui import presenters as ui

logger = logging.getLogger("agent.ui")

USER_AVATAR = "👤"
AGENT_AVATAR = "🤖"
EXAMPLE_PROMPTS = [
    "What is 25 × 17?",
    "What time is it in Tokyo?",
    "What's the weather in Berlin?",
    "Search the web for the latest Python release.",
]

CSS = """
<style>
.block-container { padding-top: 2.2rem; padding-bottom: 6rem; max-width: 860px; }
.role-label { font-size: 0.78rem; font-weight: 600; letter-spacing: 0.04em;
              text-transform: uppercase; opacity: 0.6; margin-bottom: 0.2rem; }
.welcome { text-align: center; padding: 3.5rem 0 1.5rem 0; }
.welcome h2 { margin-bottom: 0.3rem; }
.welcome p { opacity: 0.7; }
section[data-testid="stSidebar"] .stMarkdown p { margin-bottom: 0.25rem; }
</style>
"""


# ---------------------------------------------------------------------------- setup


@st.cache_resource
def load_settings() -> Settings:
    settings = get_settings()
    setup_logging(settings.log_level)
    return settings


def init_session(settings: Settings) -> None:
    """One Agent per browser session, created once and kept across reruns."""
    state = st.session_state
    state.setdefault("messages", [])  # the UI's copy of the conversation (see README)
    if "agent" not in state:
        try:
            state.agent = create_agent(settings)
            state.agent_error = None
        except ConfigError as exc:
            state.agent = None
            state.agent_error = str(exc)


def current_registry(settings: Settings) -> ToolRegistry:
    agent: Agent | None = st.session_state.agent
    return agent.registry if agent else build_registry(settings)


# ---------------------------------------------------------------------------- rendering


def role_label(text: str) -> None:
    st.markdown(f"<div class='role-label'>{text}</div>", unsafe_allow_html=True)


def render_tool_call(call: dict[str, Any], registry: ToolRegistry) -> None:
    """One collapsible block per tool call that the backend actually executed."""
    tool = registry.get(call["name"]) if isinstance(call.get("name"), str) else None
    display_name = tool.display_name if tool else str(call.get("name"))
    icon, status = ui.tool_status(call)
    summary = ui.summarize_arguments(call.get("arguments"))
    result = call.get("result") or {}

    label = f"{icon} 🔧 **{call['name']}** · `{summary}` · {status}" if summary else f"{icon} 🔧 **{call['name']}** · {status}"
    with st.expander(label, expanded=False):
        st.markdown("**Arguments**")
        st.json(call.get("arguments") or {}, expanded=True)
        if result.get("success"):
            st.markdown("**Result**")
            value = result.get("result")
            if isinstance(value, (dict, list)):
                st.json(value, expanded=True)
            else:
                st.code(str(value), language=None)
        elif status == "Declined":
            st.info("You declined this action, so it was not executed.")
        else:
            st.warning(ui.tool_failure_message(display_name, result))


def render_message(message: dict[str, Any], index: int, registry: ToolRegistry) -> None:
    if message["role"] == "user":
        with st.chat_message("user", avatar=USER_AVATAR):
            role_label("You")
            st.markdown(ui.escape_markdown(message.get("display") or message["content"]))
        return

    with st.chat_message("assistant", avatar=AGENT_AVATAR):
        role_label("Agent")
        for call in message.get("tool_calls", []):
            render_tool_call(call, registry)

        if message.get("pending_confirmation"):
            render_confirmation_request(message["pending_confirmation"], index)
            return

        friendly = ui.friendly_error(message.get("error"))
        if friendly:
            st.warning(friendly)
        elif message.get("content"):
            st.markdown(ui.escape_markdown(message["content"]))


def render_confirmation_request(calls: list[dict[str, Any]], index: int) -> None:
    """A WRITE tool is waiting. The agent (not the UI) holds it until the user answers."""
    st.info("🔐 This action requires your confirmation.")
    for call in calls:
        st.code(f"{call['name']}({json.dumps(call['arguments'], ensure_ascii=False, indent=2)})", language="python")
    if is_awaiting_confirmation(index):
        render_confirmation_buttons(index)


def render_confirmation_buttons(index: int) -> None:
    approve, deny, _ = st.columns([1, 1, 3])
    if approve.button("✅ Approve", key=f"approve_{index}", type="primary", width="stretch"):
        queue_prompt("yes", display="✅ Approved")
    if deny.button("❌ Deny", key=f"deny_{index}", width="stretch"):
        queue_prompt("no", display="❌ Denied")


def is_awaiting_confirmation(index: int) -> bool:
    agent: Agent | None = st.session_state.agent
    return index == len(st.session_state.messages) - 1 and agent is not None and agent.pending is not None


def render_welcome() -> None:
    st.markdown(
        "<div class='welcome'><h2>🤖 AI Tool Agent</h2>"
        "<p>Ask me something and I'll decide which tools I need.</p></div>",
        unsafe_allow_html=True,
    )
    columns = st.columns(2)
    for i, example in enumerate(EXAMPLE_PROMPTS):
        if columns[i % 2].button(example, key=f"example_{i}", width="stretch", disabled=st.session_state.agent is None):
            queue_prompt(example)


def render_sidebar(settings: Settings, registry: ToolRegistry) -> None:
    with st.sidebar:
        st.markdown("### AI TOOL AGENT")
        st.caption(f"Model: `{settings.model}`")

        st.markdown("**Tools**")
        for tool in registry.tools():
            note = " · *asks first*" if tool.permission is Permission.WRITE else ""
            st.markdown(f"✓ {tool.display_name}{note}", help=tool.description)

        st.divider()
        st.markdown("**Agent Status**")
        if st.session_state.agent is not None:
            st.markdown(":green[●] Online")
        else:
            st.markdown(":red[●] Offline")
            st.caption(st.session_state.agent_error or "")

        st.divider()
        st.markdown("**Conversation**")
        if st.button("Clear conversation", width="stretch", disabled=not st.session_state.messages):
            st.session_state.messages = []
            st.session_state.pop("queued_prompt", None)
            if st.session_state.agent is not None:
                st.session_state.agent.reset()  # the agent's own memory + pending confirmation
            st.rerun()


# ---------------------------------------------------------------------------- actions


def queue_prompt(text: str, display: str | None = None) -> None:
    """Buttons (examples, approve/deny) send a message on the next run, like typing it."""
    st.session_state.queued_prompt = {"text": text, "display": display}
    st.rerun()


def run_agent(prompt: str, display: str | None, registry: ToolRegistry) -> None:
    """Send one message to the agent, showing its real progress while it works."""
    state = st.session_state
    user_message = {"role": "user", "content": prompt, "display": display}
    state.messages.append(user_message)
    render_message(user_message, len(state.messages) - 1, registry)

    with st.chat_message("assistant", avatar=AGENT_AVATAR):
        role_label("Agent")
        tools_area = st.container()  # finished tool calls appear here as they complete
        status = st.status("🤔 Agent is thinking...", expanded=False)

        def on_event(event: AgentEvent) -> None:
            if event.type == "thinking":
                status.update(label="🤔 Agent is thinking...", state="running")
            elif event.type == "tool_start":
                tool = registry.get(event.data["name"])
                name = tool.display_name if tool else event.data["name"]
                status.update(label=f"🔧 Using {name}...", state="running")
            elif event.type == "tool_end":
                with tools_area:
                    render_tool_call(event.data, registry)

        try:
            reply = state.agent.chat(prompt, on_event=on_event)
            assistant_message = {"role": "assistant", "content": reply.text, **_reply_fields(reply)}
        except Exception:  # noqa: BLE001 - the agent already handles expected errors; this is a last resort
            logger.exception("Unexpected error while running the agent")
            assistant_message = {"role": "assistant", "content": "", "tool_calls": [], "error": "unexpected",
                                 "pending_confirmation": None}
        status.update(label="Done", state="complete")

    state.messages.append(assistant_message)
    st.rerun()  # redraw everything from session_state so live and history views are identical


def _reply_fields(reply) -> dict[str, Any]:
    data = reply.to_dict()
    return {"tool_calls": data["tool_calls"], "error": data["error"], "pending_confirmation": data["pending_confirmation"]}


# ---------------------------------------------------------------------------- page


def main() -> None:
    st.set_page_config(page_title="AI Tool Agent", page_icon="🤖", layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)

    settings = load_settings()
    init_session(settings)
    registry = current_registry(settings)
    state = st.session_state

    render_sidebar(settings, registry)

    st.markdown("## 🤖 AI Tool Agent")
    st.caption("An AI assistant that can reason and use tools.")

    typed = st.chat_input("Ask the agent something...", disabled=state.agent is None)
    queued = state.pop("queued_prompt", None)
    prompt, display = (typed, None) if typed else ((queued["text"], queued["display"]) if queued else (None, None))

    if state.agent is None:
        st.error("⚠️ The agent is offline: ANTHROPIC_API_KEY is not set. Add it to `.env` and restart the app.")

    if not state.messages and not prompt:
        render_welcome()
    for index, message in enumerate(state.messages):
        render_message(message, index, registry)

    if prompt and state.agent is not None:
        run_agent(prompt, display, registry)


main()
