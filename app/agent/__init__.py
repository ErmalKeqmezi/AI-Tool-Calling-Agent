"""Agent package: `create_agent()` wires settings -> LLM client -> registry -> executor -> Agent."""

from __future__ import annotations

from app.agent.agent import Agent, AgentEvent, AgentReply
from app.agent.executor import ToolExecutor
from app.config.settings import Settings
from app.llm.client import ChatModel, LLMClient
from app.tools import build_registry


def create_agent(settings: Settings, llm: ChatModel | None = None) -> Agent:
    registry = build_registry(settings)
    return Agent(
        llm=llm or LLMClient(settings),
        registry=registry,
        executor=ToolExecutor(registry),
        max_iterations=settings.max_agent_iterations,
        max_history_turns=settings.max_history_turns,
    )


__all__ = ["Agent", "AgentEvent", "AgentReply", "ToolExecutor", "create_agent"]
