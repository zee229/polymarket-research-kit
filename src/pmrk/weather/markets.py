"""Daily highest-temperature markets: discovery and rule parsing.

Events are enumerated per recurring Gamma series `<city>-daily-weather`; the `daily-temperature` tag only exists
from late December 2025, so tag-based discovery would miss 2025 markets.

Why regex is acceptable here: bucket labels (`44-45°F`, `26°C or below`) and descriptions are machine-generated from
a few fixed English Polymarket templates, not free text, and every failure is surfaced (0 of 117,690 labels failed
in the study). The settlement config (station, source, unit) is parsed **per event**: stations, sources and even
units change over time for the same city.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from pmrk.config import data_dir
from pmrk.polymarket import gamma
from pmrk.polymarket.markets import markets_frame
from pmrk.polymarket.store import save_markets

log = logging.getLogger(__name__)

SERIES_SUFFIX = "-daily-weather"

_NUM = r"(-?\d+)"
_LABEL_PATTERNS = [
    (re.compile(rf"^(?:<\s*){_NUM}\s*°?\s*[FC]?$"), "lt"),
    (re.compile(rf"^{_NUM}\s*°?\s*[FC]?\s+or\s+(?:below|lower|less)$", re.I), "le"),
    (re.compile(rf"^{_NUM}\s*°?\s*[FC]?\s+or\s+(?:higher|above|more)$", re.I), "ge"),
    (re.compile(rf"^{_NUM}\s*[-–]\s*{_NUM}\s*°?\s*[FC]?$"), "range"),
    (re.compile(rf"^{_NUM}\s*°?\s*[FC]?$"), "eq"),
]
_ICAO_PATTERNS = [
    re.compile(r"timeseries\?site=([A-Za-z0-9]{4})\b"),
    re.compile(r"wunderground\.com/history/daily/[^\s]*?/([A-Za-z0-9]{4})(?:[/.\s]|$)"),
]
_DATE_PATTERNS = [
    (re.compile(r"\bon (\d{1,2} [A-Z][a-z]{2} '\d{2})"), "%d %b '%y"),
    (re.compile(r"\bon (\d{1,2} [A-Z][a-z]{2} \d{4})"), "%d %b %Y"),
    (re.compile(r"\bon ([A-Z][a-z]+ \d{1,2}, \d{4})"), "%B %d, %Y"),
]


def weather_dir() -> Path:
    return data_dir() / "weather"


def weather_markets_path() -> Path:
    return weather_dir() / "markets.parquet"


def parse_label(label: str) -> tuple[int | None, int | None]:
    """Inclusive integer bounds [low, high] of a bucket label in the market unit; None = open end."""
    text = label.strip()
    for pattern, kind in _LABEL_PATTERNS:
        m = pattern.match(text)
        if not m:
            continue
        a = int(m.group(1))
        if kind == "lt":
            return None, a - 1
        if kind == "le":
            return None, a
        if kind == "ge":
            return a, None
        if kind == "range":
            return a, int(m.group(2))
        return a, a
    raise ValueError(f"unparsed bucket label: {label!r}")


def parse_unit(labels: list[str], description: str) -> str:
    joined = " ".join(labels)
    if "°C" in joined or "° C" in joined:
        return "C"
    if "°F" in joined:
        return "F"
    return "C" if "degrees Celsius" in description else "F"


def parse_icao(description: str, resolution_source: str | None) -> str | None:
    for text in (description, resolution_source or ""):
        for pattern in _ICAO_PATTERNS:
            if m := pattern.search(text):
                return m.group(1).upper()
    return None


def parse_source(description: str) -> str:
    """`wu` (Weather Underground), `noaa` (weather.gov timeseries), `hko` (Hong Kong Observatory) or `unknown`."""
    if "Hong Kong Observatory" in description:
        return "hko"
    first = description.split("resolution source for this market will be", 1)[-1][:200]
    if "NOAA" in first:
        return "noaa"
    if "Wunderground" in first or "Weather Underground" in first:
        return "wu"
    return "unknown"


def parse_local_date(description: str, event_date: str | None) -> date | None:
    """The observation day comes from the description: slugs are wrong at least once."""
    for pattern, fmt in _DATE_PATTERNS:
        if m := pattern.search(description):
            return datetime.strptime(m.group(1), fmt).date()
    return date.fromisoformat(event_date) if event_date else None


def event_rules(event: dict[str, Any], city: str) -> list[dict[str, Any]]:
    """Weather columns per market of one event."""
    desc = event.get("description") or ""
    labels = [m.get("groupItemTitle") or "" for m in event["markets"]]
    common = {
        "event_id": str(event["id"]),
        "city": city,
        "station_icao": parse_icao(desc, event.get("resolutionSource")),
        "source": parse_source(desc),
        "unit": parse_unit(labels, desc),
        "local_date": parse_local_date(desc, event.get("eventDate")),
        "has_wu_fallback": "Weather Underground Daily Observations table will be used" in desc,
    }
    rows = []
    for m, label in zip(event["markets"], labels, strict=True):
        low, high = parse_label(label)
        rows.append(
            {**common, "market_id": str(m["id"]), "bucket_label": label, "bucket_low": low, "bucket_high": high}
        )
    return rows


WEATHER_SCHEMA = {
    "event_id": pl.Utf8,
    "city": pl.Utf8,
    "station_icao": pl.Utf8,
    "source": pl.Utf8,
    "unit": pl.Utf8,
    "local_date": pl.Date,
    "has_wu_fallback": pl.Boolean,
    "market_id": pl.Utf8,
    "bucket_label": pl.Utf8,
    "bucket_low": pl.Int32,
    "bucket_high": pl.Int32,
}


def weather_frame(events_by_city: dict[str, list[dict[str, Any]]]) -> pl.DataFrame:
    rows, failures = [], []
    for city, events in events_by_city.items():
        for ev in events:
            if not ev.get("markets"):
                continue
            try:
                rows.extend(event_rules(ev, city))
            except ValueError as exc:
                failures.append((ev.get("slug"), str(exc)))
    for slug, err in failures:
        log.warning("parse failure %s: %s", slug, err)
    return pl.DataFrame(rows, schema=WEATHER_SCHEMA)


def fetch_all() -> pl.DataFrame:
    """Crawl every `*-daily-weather` series; store core rows and the weather rule table."""
    series = [s for s in gamma.list_series() if (s.get("slug") or "").endswith(SERIES_SUFFIX)]
    by_city = {s["slug"].removesuffix(SERIES_SUFFIX): gamma.series_events(s["id"]) for s in series}
    core = save_markets(markets_frame([e for evs in by_city.values() for e in evs]))
    wx = weather_frame(by_city)
    weather_dir().mkdir(parents=True, exist_ok=True)
    wx.write_parquet(weather_markets_path())
    log.info("%d series, %d weather markets, %d core markets stored", len(series), wx.height, core.height)
    return wx


def load(markets: pl.DataFrame | None = None) -> pl.DataFrame:
    """Core market rows joined with weather rules, plus `opened_after_day_start`."""
    from pmrk.polymarket.store import load_markets

    core = load_markets() if markets is None else markets
    wx = pl.read_parquet(weather_markets_path())
    df = core.join(wx.drop("event_id"), on="market_id", how="inner")
    open_ts = pl.coalesce("start_ts", "created_ts")
    # UTC+14 is the earliest any local day starts; opening after that means trading began with the day under way.
    day0 = pl.col("local_date").cast(pl.Datetime("us", "UTC")) - pl.duration(hours=14)
    return df.with_columns(opened_after_day_start=open_ts > day0)


def settleable() -> pl.Expr:
    """METAR-settled events that traded before the observation day began (the backtest universe)."""
    return (
        pl.col("station_icao").is_not_null()
        & pl.col("source").is_in(["wu", "noaa"])
        & ~pl.col("archived_copy")
        & ~pl.col("opened_after_day_start").fill_null(True)
    )
