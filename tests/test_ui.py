"""Streamlit UI tests (headless, via streamlit.testing.AppTest) with a scripted agent."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.agent import Agent
from app.llm.client import LLMTimeoutError
from app.ui import presenters as ui
from tests.fakes import FakeLLM, text, tools

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _app(registry, *script):
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["agent"] = Agent(llm=FakeLLM(*script), registry=registry)
    at.session_state["agent_error"] = None
    return at.run()


def test_welcome_screen_and_sidebar_tools(registry):
    at = _app(registry)
    assert not at.exception
    assert {b.label for b in at.button} >= {"What is 25 × 17?", "What time is it in Tokyo?",
                                                 "What's the weather in Berlin?",
                                                 "Search the web for the latest Python release."}
    sidebar = " ".join(m.value for m in at.sidebar.markdown)
    for name in ("Calculator", "Weather", "Time", "Web Search", "Online"):
        assert name in sidebar


def test_multiple_tool_calls_rendered(registry):
    at = _app(
        registry,
        tools(("calculate", {"expression": "25 * 17"}), ("get_current_time", {"timezone": "Asia/Tokyo"})),
        text("25 × 17 = 425. It's evening in Tokyo."),
    )
    at.chat_input[0].set_value("What is 25 × 17 and what time is it in Tokyo?").run()

    assert not at.exception
    labels = [e.label for e in at.expander]
    assert labels[0].startswith("✓ 🔧 **calculate** · `expression: 25 * 17` · Completed")
    assert "get_current_time" in labels[1] and "Asia/Tokyo" in labels[1]
    assert any("425. It's evening" in m.value for m in at.markdown)
    stored = at.session_state["messages"]
    assert [m["role"] for m in stored] == ["user", "assistant"]
    assert len(stored[1]["tool_calls"]) == 2


def test_example_button_sends_prompt(registry):
    at = _app(registry, tools(("calculate", {"expression": "25 * 17"})), text("425"))
    next(b for b in at.button if b.label == "What is 25 × 17?").click().run()
    assert at.session_state["messages"][0]["content"] == "What is 25 × 17?"
    assert at.session_state["messages"][1]["content"] == "425"


def test_llm_error_shows_friendly_message(registry):
    at = _app(registry, LLMTimeoutError("socket timeout at 10.0.0.1"))
    at.chat_input[0].set_value("Hello").run()
    warnings = [w.value for w in at.warning]
    assert len(warnings) == 1 and "Unable to contact the AI model. Please try again." in warnings[0]
    assert not any("10.0.0.1" in m.value for m in at.markdown)  # raw exception text is not shown


def test_tool_failure_is_marked(registry):
    at = _app(registry, tools(("calculate", {"expression": "1/0"})), text("Can't divide by zero."))
    at.chat_input[0].set_value("1/0").run()
    assert at.expander[0].label.startswith("✗")
    assert "calculator tool failed" in at.warning[0].value


def test_write_tool_approve_flow(registry, settings):
    at = _app(registry, tools(("create_file", {"filename": "hi.txt", "content": "hi"})), text("Created hi.txt."))
    at.chat_input[0].set_value("create hi.txt").run()
    approve = next(b for b in at.button if b.label == "✅ Approve")
    approve.click().run()
    assert (settings.file_sandbox_dir / "hi.txt").exists()
    assert at.session_state["messages"][2]["display"] == "✅ Approved"
    assert at.session_state["messages"][-1]["content"] == "Created hi.txt."


def test_clear_conversation(registry):
    at = _app(registry, text("Hi!"))
    at.chat_input[0].set_value("Hello").run()
    next(b for b in at.sidebar.button if b.label == "Clear conversation").click().run()
    assert at.session_state["messages"] == []
    assert at.session_state["agent"].messages == []


@pytest.mark.parametrize("error, expected", [("LLMRateLimitError", "busy"), ("max_iterations", None), (None, None)])
def test_friendly_error(error, expected):
    message = ui.friendly_error(error)
    assert (message is None) if expected is None else (expected in message)


def test_presenters():
    assert ui.summarize_arguments({"expression": "25 * 17"}) == "expression: 25 * 17"
    assert ui.escape_markdown("$5") == r"\$5"
    assert "internal error" in ui.tool_failure_message("Calculator", {"error": {"type": "execution_error", "message": "Traceback..."}})
