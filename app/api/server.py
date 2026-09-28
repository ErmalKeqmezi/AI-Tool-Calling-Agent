"""HTTP API. Only translates HTTP <-> Agent; all agent logic lives in app.agent.

    uvicorn app.api.server:create_app --factory --reload
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections import OrderedDict
from typing import Any, Callable

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.agent import Agent, create_agent
from app.config.settings import ConfigError, get_settings
from app.logging_config import setup_logging

logger = logging.getLogger("agent.api")

MAX_SESSIONS = 1000


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)
    session_id: str | None = Field(default=None, description="Omit to start a new conversation.")


class ChatResponse(BaseModel):
    response: str
    session_id: str
    tool_calls: list[dict[str, Any]] = []
    pending_confirmation: list[dict[str, Any]] | None = None
    error: str | None = None


class SessionStore:
    """In-memory conversations (one Agent per session), oldest evicted first."""

    def __init__(self, factory: Callable[[], Agent], max_sessions: int = MAX_SESSIONS):
        self._factory = factory
        self._max = max_sessions
        self._agents: OrderedDict[str, Agent] = OrderedDict()
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def get_or_create(self, session_id: str | None) -> tuple[str, Agent, threading.Lock]:
        with self._guard:
            if session_id is None or session_id not in self._agents:
                session_id = session_id or uuid.uuid4().hex
                self._agents[session_id] = self._factory()
                self._locks[session_id] = threading.Lock()
                while len(self._agents) > self._max:
                    old, _ = self._agents.popitem(last=False)
                    self._locks.pop(old, None)
            self._agents.move_to_end(session_id)
            return session_id, self._agents[session_id], self._locks[session_id]

    def delete(self, session_id: str) -> bool:
        with self._guard:
            self._locks.pop(session_id, None)
            return self._agents.pop(session_id, None) is not None


def create_app(agent_factory: Callable[[], Agent] | None = None) -> FastAPI:
    if agent_factory is None:
        settings = get_settings()
        setup_logging(settings.log_level)
        settings.require_api_key()  # fail fast at startup, not on the first request
        agent_factory = lambda: create_agent(settings)  # noqa: E731

    store = SessionStore(agent_factory)
    app = FastAPI(title="AI Tool Agent", version="1.0.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        # Sync endpoint: FastAPI runs it in a thread pool, so a slow LLM call doesn't block others.
        try:
            session_id, agent, lock = store.get_or_create(request.session_id)
        except ConfigError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        with lock:  # one request at a time per conversation
            reply = agent.chat(request.message)
        return ChatResponse(
            response=reply.text,
            session_id=session_id,
            tool_calls=reply.tool_calls,
            pending_confirmation=reply.pending_confirmation,
            error=reply.error,
        )

    @app.delete("/sessions/{session_id}")
    def delete_session(session_id: str) -> dict[str, bool]:
        if not store.delete(session_id):
            raise HTTPException(status_code=404, detail="Unknown session")
        return {"deleted": True}

    return app

