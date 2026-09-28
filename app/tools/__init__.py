"""Tool registration. To add a tool: write its module, then add one line below."""

from __future__ import annotations

from app.config.settings import Settings
from app.tools.base import Permission, Tool, ToolRegistry
from app.tools.calculator import CALCULATOR_TOOL
from app.tools.files import make_file_tools
from app.tools.search import make_search_tool
from app.tools.time import TIME_TOOL
from app.tools.weather import make_weather_tool


def build_registry(settings: Settings) -> ToolRegistry:
    registry = ToolRegistry()

    # READ tools - run automatically
    registry.register(CALCULATOR_TOOL)
    registry.register(TIME_TOOL)
    registry.register(make_weather_tool(settings.tool_http_timeout))
    registry.register(make_search_tool(settings.tavily_api_key, settings.tool_http_timeout))

    # WRITE tools - require user confirmation
    for tool in make_file_tools(settings.file_sandbox_dir):
        registry.register(tool)

    return registry


__all__ = ["Permission", "Tool", "ToolRegistry", "build_registry"]
