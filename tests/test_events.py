"""The optional on_event hook the Streamlit UI uses for live progress."""

from tests.fakes import FakeLLM, text, tools


def test_events_follow_real_execution(make_agent):
    llm = FakeLLM(
        tools(("calculate", {"expression": "25 * 17"}), ("get_current_time", {"timezone": "Asia/Tokyo"})),
        text("Done."),
    )
    events = []
    reply = make_agent(llm).chat("both please", on_event=events.append)

    assert [e.type for e in events] == ["thinking", "tool_start", "tool_end", "tool_start", "tool_end", "thinking"]
    assert events[2].data["result"]["result"]["value"] == 425
    assert reply.to_dict()["response"] == "Done."
    assert [c["name"] for c in reply.to_dict()["tool_calls"]] == ["calculate", "get_current_time"]


def test_no_tools_to_dict(make_agent):
    reply = make_agent(FakeLLM(text("Hello! How can I help?"))).chat("Hello")
    assert reply.to_dict() == {"response": "Hello! How can I help?", "tool_calls": [],
                               "pending_confirmation": None, "error": None}


def test_confirmation_event_and_declined_tool_end(make_agent):
    llm = FakeLLM(tools(("delete_file", {"filename": "a.txt"})), text("Cancelled."))
    agent = make_agent(llm)
    events = []
    agent.chat("delete a.txt", on_event=events.append)
    assert events[-1].type == "confirmation_required"

    events.clear()
    agent.chat("no", on_event=events.append)
    assert events[0].type == "tool_end" and events[0].data["result"]["error"]["type"] == "permission_denied"


def test_broken_event_handler_does_not_break_agent(make_agent):
    def boom(_):
        raise RuntimeError("ui crashed")

    reply = make_agent(FakeLLM(tools(("calculate", {"expression": "1+1"})), text("2"))).chat("1+1", on_event=boom)
    assert reply.text == "2"
