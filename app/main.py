"""Command-line chat.

    python -m app.main              # full agent with tools
    python -m app.main --no-tools   # Phase 1: plain LLM chat, no tools
    python -m app.main -v           # DEBUG logging
"""

from __future__ import annotations

import argparse
import sys

from app.agent import create_agent
from app.config.settings import ConfigError, get_settings
from app.llm.client import LLMClient, LLMError
from app.logging_config import setup_logging


def run_plain_chat(client: LLMClient) -> None:
    """Phase 1: user -> LLM -> response, with history but no tools."""
    messages: list[dict] = []
    while True:
        user = _read_input()
        if user is None:
            return
        messages.append({"role": "user", "content": user})
        try:
            response = client.chat(messages)
        except LLMError as exc:
            print(f"[error] {exc}")
            messages.pop()  # drop the unanswered message so history stays valid
            continue
        messages.append({"role": "assistant", "content": response.content})
        print(f"assistant> {response.text}\n")


def run_agent_chat(settings) -> None:
    agent = create_agent(settings)
    print(f"Tools: {', '.join(agent.registry.names())}")
    while True:
        user = _read_input()
        if user is None:
            return
        if user == "/reset":
            agent.reset()
            print("(conversation cleared)\n")
            continue
        reply = agent.chat(user)
        print(f"assistant> {reply.text}\n")


def _read_input() -> str | None:
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if text in {"/quit", "/exit"}:
            return None
        if text:
            return text


def main() -> int:
    parser = argparse.ArgumentParser(description="AI tool-calling agent")
    parser.add_argument("--no-tools", action="store_true", help="plain LLM chat (Phase 1)")
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging")
    args = parser.parse_args()

    try:
        settings = get_settings()
        setup_logging("DEBUG" if args.verbose else settings.log_level)
        print(f"Model: {settings.model}   (/reset clears memory, /quit exits)")
        if args.no_tools:
            run_plain_chat(LLMClient(settings))
        else:
            run_agent_chat(settings)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
