from fastapi.testclient import TestClient

from app.agent import Agent
from app.api.server import create_app
from tests.fakes import FakeLLM, text, tools


def _client(registry, *script):
    llm = FakeLLM(*script)
    return TestClient(create_app(lambda: Agent(llm=llm, registry=registry))), llm


def test_chat_endpoint(registry):
    client, _ = _client(registry, tools(("calculate", {"expression": "25 * 17"})), text("25 × 17 = 425"))
    body = client.post("/chat", json={"message": "What is 25 * 17?"}).json()

    assert body["response"] == "25 × 17 = 425"
    assert body["tool_calls"][0]["name"] == "calculate"
    assert body["session_id"]


def test_session_memory_and_confirmation(registry):
    client, llm = _client(
        registry,
        tools(("delete_file", {"filename": "a.txt"})),
        text("Cancelled."),
    )
    first = client.post("/chat", json={"message": "delete a.txt"}).json()
    assert first["pending_confirmation"][0]["name"] == "delete_file"

    second = client.post("/chat", json={"message": "no", "session_id": first["session_id"]}).json()
    assert second["response"] == "Cancelled."
    assert second["pending_confirmation"] is None


def test_validation_error(registry):
    client, _ = _client(registry)
    assert client.post("/chat", json={"message": ""}).status_code == 422
    assert client.get("/health").json() == {"status": "ok"}
