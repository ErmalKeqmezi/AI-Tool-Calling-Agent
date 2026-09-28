"""End-to-end tests against the real LLM. Skipped unless ANTHROPIC_API_KEY is set.

    pytest tests/test_live.py -v
"""

import pytest

from app.agent import create_agent
from app.config.settings import get_settings

settings = get_settings()
pytestmark = pytest.mark.skipif(not settings.anthropic_api_key, reason="ANTHROPIC_API_KEY not set")


@pytest.fixture
def agent(tmp_path):
    from dataclasses import replace

    return create_agent(replace(settings, file_sandbox_dir=tmp_path))


def _tools_used(reply):
    return [c["name"] for c in reply.tool_calls]


def test_live_calculation(agent):
    reply = agent.chat("What is 15 * 30?")
    assert "calculate" in _tools_used(reply)
    assert "450" in reply.text


def test_live_no_tool(agent):
    reply = agent.chat("Hello")
    assert _tools_used(reply) == []


def test_live_time(agent):
    reply = agent.chat("What time is it in Tokyo?")
    assert "get_current_time" in _tools_used(reply)


def test_live_multi_tool(agent):
    reply = agent.chat("What is 25 * 17 and what time is it in Tokyo?")
    assert {"calculate", "get_current_time"} <= set(_tools_used(reply))
    assert "425" in reply.text


def test_live_write_needs_confirmation(agent):
    reply = agent.chat("Create a file called hello.txt containing 'hi'.")
    assert reply.pending_confirmation and reply.pending_confirmation[0]["name"] == "create_file"
