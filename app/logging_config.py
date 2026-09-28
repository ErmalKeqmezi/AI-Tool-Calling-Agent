"""Logging setup, with a filter that scrubs anything that looks like a secret."""

from __future__ import annotations

import logging
import re
import sys

_SECRET_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]+"),  # Anthropic keys
    re.compile(r"tvly-[A-Za-z0-9_\-]+"),  # Tavily keys
    re.compile(r"(?i)(api[_-]?key|authorization|x-api-key)(\"?\s*[:=]\s*\"?)(Bearer\s+)?[^\s\",}]+"),
]


class RedactSecretsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        redacted = message
        for pattern in _SECRET_PATTERNS:
            if pattern.groups >= 2:
                redacted = pattern.sub(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]", redacted)
            else:
                redacted = pattern.sub("[REDACTED]", redacted)
        if redacted != message:
            record.msg, record.args = redacted, ()
        return True


def setup_logging(level: str = "INFO") -> None:
    # Windows consoles default to a legacy code page; make sure non-ASCII symbols print correctly.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s", "%H:%M:%S"))
    handler.addFilter(RedactSecretsFilter())

    root = logging.getLogger("agent")
    root.handlers[:] = [handler]
    root.setLevel(level)
    root.propagate = False

    # The SDK / HTTP libraries are noisy at DEBUG and may echo headers - keep them quiet.
    for noisy in ("anthropic", "httpx", "httpx2", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
