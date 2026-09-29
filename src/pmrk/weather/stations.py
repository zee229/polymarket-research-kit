"""Station metadata: coordinates and elevation (aviationweather.gov) and IANA timezone (Open-Meteo)."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from pmrk.polymarket.http import Client
from pmrk.weather.markets import weather_dir

_awc = Client("awc", min_interval_s=1.0)
_om = Client("open_meteo_meta", min_interval_s=0.5)


def stations_path() -> Path:
    return weather_dir() / "stations.parquet"


def build_stations(icaos: list[str]) -> pl.DataFrame:
    """Fetch and store metadata for the given ICAO codes; raises if any is unknown."""
    info = _awc.get_json(
        "https://aviationweather.gov/api/data/stationinfo",
        {"ids": ",".join(sorted(icaos)), "format": "json"},
        cache=True,
    )
    rows = []
    for s in info:
        tz = _om.get_json(
            "https://api.open-meteo.com/v1/forecast",
            {
                "latitude": s["lat"],
                "longitude": s["lon"],
                "timezone": "auto",
                "forecast_days": 1,
                "daily": "temperature_2m_max",
            },
            cache=True,
        )["timezone"]
        rows.append(
            {
                "station_icao": s["icaoId"],
                "name": s["site"],
                "country": s["country"],
                "lat": float(s["lat"]),
                "lon": float(s["lon"]),
                "elev_m": float(s["elev"]),
                "tz": tz,
            }
        )
    missing = set(icaos) - {r["station_icao"] for r in rows}
    if missing:
        raise ValueError(f"no station metadata for {sorted(missing)}")
    df = pl.DataFrame(rows).sort("station_icao")
    stations_path().parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(stations_path())
    return df


def load_stations() -> pl.DataFrame:
    return pl.read_parquet(stations_path())


def with_local_date(df: pl.DataFrame, ts_col: str, stations: pl.DataFrame, local_time: bool = False) -> pl.DataFrame:
    """Add `local_date` (and optionally naive `local_time`) of `ts_col` in each station's timezone (DST-aware)."""
    parts = []
    for (tz,), sub in df.join(stations.select("station_icao", "tz"), on="station_icao").group_by("tz"):
        local = pl.col(ts_col).dt.convert_time_zone(tz)
        cols = {"local_date": local.dt.date()}
        if local_time:
            cols["local_time"] = local.dt.replace_time_zone(None)
        parts.append(sub.with_columns(**cols).drop("tz"))
    return pl.concat(parts, how="diagonal_relaxed") if parts else df.with_columns(local_date=pl.lit(None, pl.Date))


def local_day_bounds(keys: pl.DataFrame, stations: pl.DataFrame) -> pl.DataFrame:
    """Add `day_start_utc` / `day_end_utc` of (station_icao, local_date) rows."""
    parts = []
    for (tz,), sub in keys.join(stations.select("station_icao", "tz"), on="station_icao").group_by("tz"):
        start = pl.col("local_date").cast(pl.Datetime("us"))
        to_utc = lambda e: e.dt.replace_time_zone(tz, ambiguous="earliest").dt.convert_time_zone("UTC")  # noqa: B023
        parts.append(
            sub.with_columns(day_start_utc=to_utc(start), day_end_utc=to_utc(start + pl.duration(days=1))).drop("tz")
        )
    return pl.concat(parts, how="diagonal_relaxed")
