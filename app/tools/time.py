"""get_current_time(timezone) - current date and time in an IANA timezone."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones

from app.tools.base import Permission, Tool, ToolError


def get_current_time(timezone: str = "UTC") -> dict[str, Any]:
    timezone = timezone.strip()
    try:
        zone = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        # Help the LLM self-correct: suggest zones whose name contains the input.
        needle = timezone.replace(" ", "_").lower().split("/")[-1]
        suggestions = sorted(z for z in available_timezones() if needle and needle in z.lower())[:5]
        hint = f" Did you mean one of {suggestions}?" if suggestions else ""
        raise ToolError(
            f"Unknown timezone {timezone!r}. Use an IANA name like 'Asia/Tokyo' or 'Europe/Berlin'.{hint}",
            "invalid_timezone",
        ) from exc

    now = datetime.now(zone)
    return {
        "timezone": timezone,
        "iso": now.isoformat(timespec="seconds"),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "day_of_week": now.strftime("%A"),
        "utc_offset": now.strftime("%z")[:3] + ":" + now.strftime("%z")[3:],
    }


TIME_TOOL = Tool(
    name="get_current_time",
    description=(
        "Get the current date and time in a timezone. Use this whenever the user asks what time or "
        "date it is somewhere, or when an answer depends on today's date. Convert city names to an "
        "IANA timezone yourself (Tokyo -> 'Asia/Tokyo', New York -> 'America/New_York')."
    ),
    parameters={
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "IANA timezone name, e.g. 'Asia/Tokyo', 'Europe/Berlin', 'UTC'.",
                "minLength": 1,
            }
        },
        "required": ["timezone"],
        "additionalProperties": False,
    },
    function=get_current_time,
    title="Time",
    permission=Permission.READ,
)
