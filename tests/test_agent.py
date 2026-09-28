"""Agent loop tests - the 10 required scenarios plus permissions and memory."""

import json

from app.llm.client import LLMRateLimitError, LLMTimeoutError
from tests.fakes import FakeLLM, text, tools


def _result(block):
    return json.loads(block["content"])


# 1. Normal conversation without tools / 10. request requiring no tool
def test_no_tool_call(make_agent):
    llm = FakeLLM(text("Hello! How can I help?"))
    agent = make_agent(llm)

    reply = agent.chat("Hello")

    assert reply.text == "Hello! How can I help?"
    assert reply.tool_calls == []
    assert len(llm.calls) == 1
    assert {t["name"] for t in llm.calls[0]["tools"]} >= {"calculate", "get_current_time", "get_weather", "search_web"}


# 2. One tool call
def test_single_tool_call(make_agent):
    llm = FakeLLM(tools(("calculate", {"expression": "25 * 17"})), text("25 × 17 = 425"))
    agent = make_agent(llm)

    reply = agent.chat("What is 25 * 17?")

    assert reply.text == "25 × 17 = 425"
    assert [c["name"] for c in reply.tool_calls] == ["calculate"]
    [block] = llm.last_tool_results()
    assert _result(block) == {"success": True, "result": {"expression": "25 * 17", "value": 425}}
    assert block["is_error"] is False
    # history: user, assistant(tool_use), user(tool_result), assistant(final)
    assert [m["role"] for m in agent.messages] == ["user", "assistant", "user", "assistant"]


# 3. Multiple tool calls in one LLM response (parallel)
def test_multiple_tool_calls_in_one_response(make_agent):
    llm = FakeLLM(
        tools(("calculate", {"expression": "25 * 17"}), ("get_current_time", {"timezone": "Asia/Tokyo"})),
        text("425, and it's evening in Tokyo."),
    )
    reply = make_agent(llm).chat("What is 25 * 17 and what time is it in Tokyo?")

    assert [c["name"] for c in reply.tool_calls] == ["calculate", "get_current_time"]
    results = llm.last_tool_results()
    assert len(results) == 2  # both results go back in ONE user message
    assert _result(results[0])["result"]["value"] == 425
    assert _result(results[1])["result"]["timezone"] == "Asia/Tokyo"


# 9. Multiple sequential tool calls, the second depending on the first
def test_sequential_dependent_tool_calls(make_agent):
    llm = FakeLLM(
        tools(("calculate", {"expression": "25 * 17"})),
        tools(("calculate", {"expression": "425 / 5"})),  # uses the first result
        text("25 × 17 = 425, divided by 5 is 85."),
    )
    reply = make_agent(llm).chat("Multiply 25 by 17, then divide the result by 5.")

    assert [c["result"]["result"]["value"] for c in reply.tool_calls] == [425, 85]
    assert len(llm.calls) == 3


# 4. Invalid tool name
def test_unknown_tool_is_reported_to_llm(make_agent):
    llm = FakeLLM(tools(("launch_rockets", {})), text("I don't have that tool."))
    reply = make_agent(llm).chat("Launch the rockets")

    [block] = llm.last_tool_results()
    assert block["is_error"] is True
    assert _result(block)["error"]["type"] == "unknown_tool"
    assert reply.text == "I don't have that tool."


# 5. Missing argument
def test_missing_argument(make_agent):
    llm = FakeLLM(tools(("calculate", {})), text("Which expression?"))
    make_agent(llm).chat("calculate")

    error = _result(llm.last_tool_results()[0])["error"]
    assert error["type"] == "invalid_arguments"
    assert "missing required argument 'expression'" in error["message"]


# 6. Invalid argument (wrong type) - and the LLM can self-correct on the next iteration
def test_invalid_argument_then_self_correction(make_agent):
    llm = FakeLLM(
        tools(("calculate", {"expression": 425})),
        tools(("calculate", {"expression": "425"})),
        text("425"),
    )
    reply = make_agent(llm).chat("What is 425?")

    first, second = reply.tool_calls
    assert first["result"]["error"]["type"] == "invalid_arguments"
    assert second["result"]["success"] is True


def test_malformed_json_arguments(make_agent):
    llm = FakeLLM(tools(("calculate", '{"expression": "1+')), text("Oops."))
    make_agent(llm).chat("1+")
    assert _result(llm.last_tool_results()[0])["error"]["type"] == "malformed_arguments"


# 7. Tool failure (expected error and unexpected crash)
def test_tool_error_does_not_crash_agent(make_agent):
    llm = FakeLLM(tools(("calculate", {"expression": "1 / 0"})), text("You can't divide by zero."))
    reply = make_agent(llm).chat("1/0?")
    assert reply.tool_calls[0]["result"]["error"]["type"] == "math_error"
    assert reply.error is None


def test_tool_crash_is_caught(make_agent, registry):
    from app.tools.base import Tool

    def broken(**_):
        raise RuntimeError("disk on fire")

    registry.register(Tool("broken", "Always crashes.", {"type": "object", "properties": {}}, broken))
    llm = FakeLLM(tools(("broken", {})), text("The tool failed."))
    reply = make_agent(llm).chat("run broken")

    error = reply.tool_calls[0]["result"]["error"]
    assert error["type"] == "execution_error" and "disk on fire" in error["message"]
    assert reply.text == "The tool failed."


# 8. LLM API failure
def test_llm_failure_returns_friendly_error_and_keeps_history_valid(make_agent):
    llm = FakeLLM(LLMTimeoutError("timed out"), text("Hi again!"))
    agent = make_agent(llm)

    reply = agent.chat("Hello")
    assert reply.error == "LLMTimeoutError"
    assert "couldn't reach" in reply.text

    # History still alternates user/assistant, so the next turn works.
    reply = agent.chat("Hello?")
    assert reply.text == "Hi again!"
    roles = [m["role"] for m in agent.messages]
    assert roles == ["user", "assistant", "user", "assistant"]


def test_llm_failure_mid_loop(make_agent):
    llm = FakeLLM(tools(("calculate", {"expression": "2+2"})), LLMRateLimitError("429"))
    reply = make_agent(llm).chat("2+2 then explain")
    assert reply.error == "LLMRateLimitError"
    assert reply.tool_calls[0]["result"]["success"] is True


def test_max_iterations_guard(make_agent):
    llm = FakeLLM(*[tools(("calculate", {"expression": "1+1"})) for _ in range(3)])
    agent = make_agent(llm, max_iterations=3)
    reply = agent.chat("loop forever")
    assert reply.error == "max_iterations"
    assert agent.messages[-1]["role"] == "assistant"


# ---------------------------------------------------------------- permissions


def test_write_tool_requires_confirmation_then_runs_on_yes(make_agent, settings):
    llm = FakeLLM(
        tools(("create_file", {"filename": "hello.txt", "content": "hi"})),
        text("Created hello.txt."),
    )
    agent = make_agent(llm)

    reply = agent.chat("Create hello.txt saying hi")
    assert reply.pending_confirmation == [{"name": "create_file", "arguments": {"filename": "hello.txt", "content": "hi"}}]
    assert "requires your confirmation" in reply.text
    assert not (settings.file_sandbox_dir / "hello.txt").exists()  # nothing ran yet
    assert len(llm.calls) == 1

    reply = agent.chat("yes")
    assert reply.text == "Created hello.txt."
    assert (settings.file_sandbox_dir / "hello.txt").read_text() == "hi"


def test_write_tool_declined(make_agent, settings):
    llm = FakeLLM(tools(("delete_file", {"filename": "x.txt"})), text("OK, I won't delete it."))
    agent = make_agent(llm)
    agent.chat("delete x.txt")

    reply = agent.chat("no")
    assert reply.text == "OK, I won't delete it."
    assert _result(llm.last_tool_results()[0])["error"]["type"] == "permission_denied"


def test_other_reply_while_pending_counts_as_decline_and_is_forwarded(make_agent):
    llm = FakeLLM(tools(("delete_file", {"filename": "x.txt"})), text("Sure - it's sunny."))
    agent = make_agent(llm)
    agent.chat("delete x.txt")
    agent.chat("actually, what's the weather?")

    last_user = llm.calls[-1]["messages"][-1]["content"]
    assert last_user[0]["type"] == "tool_result" and last_user[0]["is_error"]
    assert last_user[-1] == {"type": "text", "text": "actually, what's the weather?"}


def test_mixed_read_and_write_calls(make_agent):
    llm = FakeLLM(
        tools(("calculate", {"expression": "6*7"}), ("create_file", {"filename": "a.txt", "content": "42"})),
        text("Saved 42."),
    )
    agent = make_agent(llm)
    reply = agent.chat("compute 6*7 and save it")
    assert [c["name"] for c in reply.tool_calls] == ["calculate"]  # READ ran immediately
    assert reply.pending_confirmation[0]["name"] == "create_file"

    agent.chat("yes")
    results = llm.last_tool_results()
    assert [r["tool_use_id"] for r in results] == ["toolu_0_calculate", "toolu_1_create_file"]
    assert all(not r["is_error"] for r in results)


def test_executor_refuses_write_without_approval(executor):
    result = executor.execute("create_file", {"filename": "a.txt", "content": "x"})
    assert result["error"]["type"] == "permission_required"


# ---------------------------------------------------------------- memory


def test_follow_up_question_sees_previous_turns(make_agent):
    llm = FakeLLM(
        tools(("get_current_time", {"timezone": "Europe/Berlin"})),
        text("It's 14:00 in Berlin."),
        text("Berlin is in Germany."),
    )
    agent = make_agent(llm)
    agent.chat("What time is it in Berlin?")
    agent.chat("Which country is that in?")

    sent = llm.calls[-1]["messages"]
    assert sent[0]["content"] == "What time is it in Berlin?"
    assert sent[-1]["content"] == "Which country is that in?"


def test_history_trimming_keeps_tool_pairs_intact(make_agent):
    script = []
    for _ in range(5):
        script += [tools(("calculate", {"expression": "1+1"})), text("2")]
    agent = make_agent(FakeLLM(*script), max_history_turns=2)
    for _ in range(5):
        agent.chat("1+1?")

    first = agent.messages[0]
    assert first["role"] == "user" and isinstance(first["content"], str)
    assert sum(1 for m in agent.messages if isinstance(m["content"], str)) == 2
