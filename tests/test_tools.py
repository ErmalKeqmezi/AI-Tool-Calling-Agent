"""Tool, validation, and executor unit tests (no network - HTTP is mocked)."""

import httpx
import pytest

from app.tools import calculator, search, weather
from app.tools.base import Tool, ToolError, ToolRegistry
from app.tools.time import get_current_time
from app.tools.validation import validate


# ---------------------------------------------------------------- calculator

@pytest.mark.parametrize(
    "expression, value",
    [("25 * 17", 425), ("15 * 30", 450), ("2 ** 10", 1024), ("(1 + 2) * 3", 9), ("10 / 4", 2.5),
     ("sqrt(16)", 4), ("-3 + 5", 2), ("7 // 2", 3), ("25 × 4", 100), ("2^3", 8)],
)
def test_calculator(expression, value):
    assert calculator.calculate(expression)["value"] == value


@pytest.mark.parametrize(
    "expression",
    ["__import__('os').system('echo hi')", "open('x')", "a + 1", "(1).real", "9 ** 9 ** 9", "1 +", "", "[1,2]"],
)
def test_calculator_rejects_unsafe_or_invalid(expression):
    with pytest.raises(ToolError):
        calculator.calculate(expression)


def test_calculator_division_by_zero():
    with pytest.raises(ToolError) as info:
        calculator.calculate("1/0")
    assert info.value.error_type == "math_error"


# ---------------------------------------------------------------- time

def test_time_valid_zone():
    result = get_current_time("Asia/Tokyo")
    assert result["timezone"] == "Asia/Tokyo" and result["utc_offset"] == "+09:00"


def test_time_invalid_zone_suggests():
    with pytest.raises(ToolError) as info:
        get_current_time("Tokyo")
    assert "Asia/Tokyo" in str(info.value)


# ---------------------------------------------------------------- validation

SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string", "minLength": 1}, "days": {"type": "integer", "minimum": 0, "maximum": 6}},
    "required": ["city"],
    "additionalProperties": False,
}


@pytest.mark.parametrize(
    "args, fragment",
    [({}, "missing required argument 'city'"), ({"city": 5}, "expected string"),
     ({"city": "X", "days": 9}, "must be <= 6"), ({"city": "X", "days": True}, "expected integer"),
     ({"city": "X", "extra": 1}, "unexpected argument 'extra'"), ({"city": ""}, "at least 1")],
)
def test_validation_errors(args, fragment):
    assert any(fragment in e for e in validate(args, SCHEMA))


def test_validation_ok():
    assert validate({"city": "Berlin", "days": 1}, SCHEMA) == []


# ---------------------------------------------------------------- registry / executor

def test_registry_rejects_duplicates_and_exports_schema():
    reg = ToolRegistry()
    reg.register(calculator.CALCULATOR_TOOL)
    with pytest.raises(ValueError):
        reg.register(calculator.CALCULATOR_TOOL)
    [schema] = reg.schemas()
    assert set(schema) == {"name", "description", "input_schema"}


def test_executor_success(executor):
    assert executor.execute("calculate", {"expression": "25 * 17"}) == {
        "success": True, "result": {"expression": "25 * 17", "value": 425}
    }


@pytest.mark.parametrize(
    "name, args, error_type",
    [("nope", {}, "unknown_tool"), (None, {}, "unknown_tool"), ("calculate", {}, "invalid_arguments"),
     ("calculate", {"expression": 5}, "invalid_arguments"), ("calculate", "{bad json", "malformed_arguments"),
     ("calculate", ["1+1"], "malformed_arguments")],
)
def test_executor_rejects(executor, name, args, error_type):
    result = executor.execute(name, args)
    assert result["success"] is False and result["error"]["type"] == error_type


def test_executor_accepts_json_string_arguments(executor):
    assert executor.execute("calculate", '{"expression": "2+2"}')["result"]["value"] == 4


def test_new_tool_needs_no_agent_changes(registry, executor):
    registry.register(Tool(
        name="reverse", description="Reverse text.",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        function=lambda text: text[::-1],
    ))
    assert executor.execute("reverse", {"text": "abc"}) == {"success": True, "result": "cba"}


# ---------------------------------------------------------------- file tools

def test_file_tools_sandbox(executor, settings):
    ok = executor.execute("create_file", {"filename": "notes/a.txt", "content": "hi"}, approved=True)
    assert ok["success"] and (settings.file_sandbox_dir / "notes" / "a.txt").exists()
    escape = executor.execute("create_file", {"filename": "../evil.txt", "content": "x"}, approved=True)
    assert escape["error"]["type"] == "forbidden_path"
    assert executor.execute("delete_file", {"filename": "notes/a.txt"}, approved=True)["success"]


# ---------------------------------------------------------------- HTTP tools (mocked)

def _mock_httpx(monkeypatch, module, handler):
    real_client = httpx.Client
    monkeypatch.setattr(module.httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))


def test_weather_tomorrow(monkeypatch):
    def handler(request):
        if "geocoding" in request.url.host:
            return httpx.Response(200, json={"results": [{"name": "Berlin", "country": "Germany", "latitude": 52.5, "longitude": 13.4}]})
        return httpx.Response(200, json={
            "timezone": "Europe/Berlin",
            "current": {"time": "2026-09-28T12:00", "temperature_2m": 18.0, "apparent_temperature": 17.0,
                        "relative_humidity_2m": 60, "wind_speed_10m": 10, "weather_code": 1},
            "daily": {"time": [f"2026-09-{28 + i}" for i in range(3)], "weather_code": [1, 61, 3],
                      "temperature_2m_max": [19, 15, 16], "temperature_2m_min": [9, 8, 7],
                      "precipitation_probability_max": [5, 80, 20]},
        })

    _mock_httpx(monkeypatch, weather, handler)
    result = weather.make_weather_tool().function(city="Berlin", days_ahead=1)
    assert result["forecast"] == {"date": "2026-09-29", "condition": "slight rain", "temp_max_c": 15,
                                  "temp_min_c": 8, "precipitation_probability_pct": 80}
    assert "current" not in result


def test_weather_city_not_found(monkeypatch):
    _mock_httpx(monkeypatch, weather, lambda r: httpx.Response(200, json={}))
    with pytest.raises(ToolError) as info:
        weather.make_weather_tool().function(city="Atlantis")
    assert info.value.error_type == "city_not_found"


def test_weather_timeout(monkeypatch):
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    _mock_httpx(monkeypatch, weather, handler)
    with pytest.raises(ToolError) as info:
        weather.make_weather_tool().function(city="Berlin")
    assert info.value.error_type == "timeout"


def test_search_rate_limited(monkeypatch):
    _mock_httpx(monkeypatch, search, lambda r: httpx.Response(429))
    with pytest.raises(ToolError) as info:
        search.make_search_tool(tavily_api_key="tvly-test").function(query="news")
    assert info.value.error_type == "rate_limited"


def test_search_duckduckgo(monkeypatch):
    _mock_httpx(monkeypatch, search, lambda r: httpx.Response(200, json={
        "Heading": "Python", "AbstractText": "A programming language.", "AbstractURL": "https://python.org",
        "RelatedTopics": [],
    }))
    result = search.make_search_tool().function(query="python language")
    assert result["results"][0]["url"] == "https://python.org"
