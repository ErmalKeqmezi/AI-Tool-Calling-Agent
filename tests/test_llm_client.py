"""LLMClient: missing key, SDK error mapping, and response normalization."""

from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from app.config.settings import ConfigError, Settings
from app.llm import client as llm_module
from app.llm.client import (
    LLMAuthError, LLMBadRequestError, LLMClient, LLMInvalidResponseError, LLMRateLimitError,
    LLMTimeoutError, LLMUnavailableError,
)

REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def _status_error(cls, code):
    return cls("boom", response=httpx2.Response(code, request=REQUEST), body=None)


def test_missing_api_key():
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        LLMClient(Settings(anthropic_api_key=None))


@pytest.fixture
def client():
    return LLMClient(Settings(anthropic_api_key="sk-ant-test", refusal_fallback=False))


@pytest.mark.parametrize(
    "exc, expected",
    [
        (anthropic.APITimeoutError(request=REQUEST), LLMTimeoutError),
        (anthropic.APIConnectionError(request=REQUEST), LLMUnavailableError),
        (_status_error(anthropic.AuthenticationError, 401), LLMAuthError),
        (_status_error(anthropic.RateLimitError, 429), LLMRateLimitError),
        (_status_error(anthropic.BadRequestError, 400), LLMBadRequestError),
        (_status_error(anthropic.NotFoundError, 404), LLMBadRequestError),
        (_status_error(anthropic.InternalServerError, 500), LLMUnavailableError),
    ],
)
def test_sdk_errors_are_mapped(client, monkeypatch, exc, expected):
    def raise_(**_):
        raise exc

    monkeypatch.setattr(client._client.messages, "create", raise_)
    with pytest.raises(expected):
        client.chat([{"role": "user", "content": "hi"}])


def test_normalizes_text_and_tool_use(client, monkeypatch):
    message = SimpleNamespace(
        stop_reason="tool_use",
        content=[
            SimpleNamespace(type="thinking", thinking="", signature="sig"),
            SimpleNamespace(type="text", text="Let me calculate."),
            SimpleNamespace(type="tool_use", id="toolu_1", name="calculate", input={"expression": "2+2"}),
        ],
    )
    monkeypatch.setattr(client._client.messages, "create", lambda **_: message)
    response = client.chat([{"role": "user", "content": "2+2"}], tools=[{"name": "calculate"}])

    assert response.text == "Let me calculate."
    assert response.tool_calls[0].name == "calculate"
    assert response.tool_calls[0].arguments == {"expression": "2+2"}
    assert len(response.content) == 3  # thinking block preserved for the history


def test_invalid_response(client, monkeypatch):
    monkeypatch.setattr(client._client.messages, "create", lambda **_: SimpleNamespace(content=None))
    with pytest.raises(LLMInvalidResponseError):
        client.chat([{"role": "user", "content": "hi"}])


def test_fallback_params_sent_when_enabled(monkeypatch):
    c = LLMClient(Settings(anthropic_api_key="sk-ant-test", refusal_fallback=True))
    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="hi")])

    monkeypatch.setattr(c._client.messages, "create", create)
    c.chat([{"role": "user", "content": "hi"}])
    assert seen["extra_body"] == {"fallbacks": "default"}


def test_secrets_are_redacted_from_logs(caplog):
    import logging
    from app.logging_config import RedactSecretsFilter

    record = logging.LogRecord("agent", logging.INFO, "", 0, "key=%s", ("sk-ant-api03-SECRET",), None)
    RedactSecretsFilter().filter(record)
    assert "SECRET" not in record.getMessage()
    assert llm_module  # module imported without side effects
