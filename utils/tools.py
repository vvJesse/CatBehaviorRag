from __future__ import annotations

import logging
from typing import Annotated

import requests
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

_OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"


# ---------------------------------------------------------------------------
# Weather tool
# ---------------------------------------------------------------------------

class WeatherInput(BaseModel):
    latitude: Annotated[str, Field(description="纬度，例如 23.1291")]
    longitude: Annotated[str, Field(description="经度，例如 113.2644")]
    date: Annotated[str, Field(description="查询日期，格式 YYYY-MM-DD")]


@retry(
    retry=retry_if_exception_type(requests.RequestException),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def _fetch_weather(latitude: str, longitude: str, date: str) -> str:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": date,
        "end_date": date,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "Asia/Shanghai",
    }
    resp = requests.get(_OPEN_METEO_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    daily = data.get("daily", {})
    if not daily.get("time"):
        return f"未找到 {date} 的天气数据"

    temp_max = daily["temperature_2m_max"][0]
    temp_min = daily["temperature_2m_min"][0]
    precip = daily["precipitation_sum"][0]

    result = (
        f"日期：{date}，"
        f"最高气温：{temp_max}°C，最低气温：{temp_min}°C，"
        f"降水量：{precip}mm"
    )
    logger.info("weather tool: lat=%s lon=%s date=%s -> %s", latitude, longitude, date, result)
    return result


def _call_weather(latitude: str, longitude: str, date: str) -> str:
    try:
        return _fetch_weather(latitude, longitude, date)
    except requests.RequestException as e:
        logger.warning("天气 API 请求失败（重试耗尽）: %s", e)
        return f"天气数据获取失败：{e}"


weather_tool = StructuredTool.from_function(
    func=_call_weather,
    name="weather",
    description="查询指定经纬度坐标在某日期的历史天气（最高温、最低温、降水量）",
    args_schema=WeatherInput,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TOOLS: list[StructuredTool] = [weather_tool]
TOOL_MAP: dict[str, StructuredTool] = {t.name: t for t in TOOLS}
