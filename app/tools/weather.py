"""get_weather(city, days_ahead) - current weather and daily forecast from Open-Meteo.

Open-Meteo is free and needs no API key: https://open-meteo.com/
"""

from __future__ import annotations

from typing import Any

import httpx

from app.tools.base import Permission, Tool, ToolError

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
MAX_DAYS_AHEAD = 6

# WMO weather interpretation codes -> text
_WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog",
    48: "depositing rime fog", 51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    56: "light freezing drizzle", 57: "dense freezing drizzle", 61: "slight rain",
    63: "moderate rain", 65: "heavy rain", 66: "light freezing rain", 67: "heavy freezing rain",
    71: "slight snow", 73: "moderate snow", 75: "heavy snow", 77: "snow grains",
    80: "slight rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    85: "slight snow showers", 86: "heavy snow showers", 95: "thunderstorm",
    96: "thunderstorm with slight hail", 99: "thunderstorm with heavy hail",
}


def _get_json(client: httpx.Client, url: str, params: dict[str, Any]) -> dict[str, Any]:
    try:
        response = client.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except httpx.TimeoutException as exc:
        raise ToolError("Weather service timed out.", "timeout") from exc
    except httpx.HTTPStatusError as exc:
        raise ToolError(f"Weather service returned HTTP {exc.response.status_code}.", "upstream_error") from exc
    except (httpx.HTTPError, ValueError) as exc:
        raise ToolError(f"Weather service unavailable: {exc}", "upstream_error") from exc


def make_weather_tool(http_timeout: float = 10.0) -> Tool:
    def get_weather(city: str, days_ahead: int = 0) -> dict[str, Any]:
        city = city.strip()
        with httpx.Client(timeout=http_timeout) as client:
            geo = _get_json(client, GEOCODING_URL, {"name": city, "count": 1, "format": "json"})
            matches = geo.get("results") or []
            if not matches:
                raise ToolError(f"Could not find a city named {city!r}.", "city_not_found")
            place = matches[0]

            forecast = _get_json(
                client,
                FORECAST_URL,
                {
                    "latitude": place["latitude"],
                    "longitude": place["longitude"],
                    "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m",
                    "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                    "forecast_days": MAX_DAYS_AHEAD + 1,
                    "timezone": "auto",
                },
            )

        try:
            daily = forecast["daily"]
            day = {
                "date": daily["time"][days_ahead],
                "condition": _WEATHER_CODES.get(daily["weather_code"][days_ahead], "unknown"),
                "temp_max_c": daily["temperature_2m_max"][days_ahead],
                "temp_min_c": daily["temperature_2m_min"][days_ahead],
                "precipitation_probability_pct": daily["precipitation_probability_max"][days_ahead],
            }
            result: dict[str, Any] = {
                "location": ", ".join(p for p in (place.get("name"), place.get("admin1"), place.get("country")) if p),
                "timezone": forecast.get("timezone"),
                "days_ahead": days_ahead,
                "forecast": day,
            }
            if days_ahead == 0:
                current = forecast["current"]
                result["current"] = {
                    "time": current["time"],
                    "temperature_c": current["temperature_2m"],
                    "feels_like_c": current["apparent_temperature"],
                    "humidity_pct": current["relative_humidity_2m"],
                    "wind_speed_kmh": current["wind_speed_10m"],
                    "condition": _WEATHER_CODES.get(current["weather_code"], "unknown"),
                }
        except (KeyError, IndexError, TypeError) as exc:
            raise ToolError("Weather service returned an unexpected response.", "upstream_error") from exc
        return result

    return Tool(
        name="get_weather",
        description=(
            "Get the weather for a city: current conditions (days_ahead=0) or the daily forecast "
            f"up to {MAX_DAYS_AHEAD} days ahead (1 = tomorrow). Temperatures are in Celsius."
        ),
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. 'Berlin' or 'Paris'.", "minLength": 1},
                "days_ahead": {
                    "type": "integer",
                    "description": "0 = today/now (default), 1 = tomorrow, ... up to 6.",
                    "minimum": 0,
                    "maximum": MAX_DAYS_AHEAD,
                },
            },
            "required": ["city"],
            "additionalProperties": False,
        },
        function=get_weather,
        title="Weather",
        permission=Permission.READ,
    )
