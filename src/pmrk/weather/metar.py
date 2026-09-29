"""METAR/SPECI archive (Iowa Environmental Mesonet) and decoding.

METAR is a fixed WMO/ICAO code format, not natural language, so positional regexes over its code groups are the
standard way to decode it:
- main group `TT/DD` (integer degC, `M` = minus);
- remark `T snnn snnn` (tenths degC, mostly US ASOS);
- remark `1snnn` (6-hour max, tenths degC);
- cloud layers (FEW/SCT/BKN/OVC/VV + CB/TCU) and wind (KT or MPS).

IEM with `report_type=3,4` returns routine reports and SPECIs without labelling them, so a station's routine
minute(s) are inferred as those holding >= 20% of its reports. IEM `valid` is the observation time; there are no
receipt timestamps.
"""

from __future__ import annotations

import csv
import io
import logging
import re
import time
from datetime import date
from pathlib import Path

import httpx
import polars as pl

from pmrk.polymarket.http import Client
from pmrk.weather.markets import weather_dir

log = logging.getLogger(__name__)

ASOS = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
FIRST_YEAR = 2023

_client = Client("iem", min_interval_s=6.0, timeout_s=300)

_MAIN = re.compile(r"\s(M?\d{2})/(M?\d{2})?(?=\s|$)")
_TGROUP = re.compile(r"\sT([01])(\d{3})([01])(\d{3})(?=\s|$)")
_MAX6 = re.compile(r"\s1([01])(\d{3})(?=\s|$)")
_CLOUD = re.compile(r"\b(FEW|SCT|BKN|OVC|VV)(\d{3}|///)(CB|TCU)?")
_CLEAR = re.compile(r"\b(CLR|SKC|NSC|NCD|CAVOK)\b")
_WIND = re.compile(r"\b(\d{3}|VRB)(\d{2,3})(?:G\d{2,3})?(KT|MPS)\b")
_OKTAS = {"FEW": 1.5, "SCT": 3.5, "BKN": 6.0, "OVC": 8.0, "VV": 8.0}
KT_PER_MPS = 1.943844


def raw_dir() -> Path:
    return weather_dir() / "metar"


def obs_path() -> Path:
    return weather_dir() / "obs.parquet"


def _signed_int(tok: str) -> int:
    return -int(tok[1:]) if tok.startswith("M") else int(tok)


def parse_metar(text: str) -> dict[str, float | None]:
    """Temperature groups; None for groups not present."""
    body, _, rmk = text.partition(" RMK ")
    mains = _MAIN.findall(" " + body)
    temp_c = float(_signed_int(mains[-1][0])) if mains else None
    t = _TGROUP.search(" " + rmk) if rmk else None
    precise = (-1 if t.group(1) == "1" else 1) * int(t.group(2)) / 10 if t else None
    m6 = _MAX6.search(" " + rmk) if rmk else None
    max6 = (-1 if m6.group(1) == "1" else 1) * int(m6.group(2)) / 10 if m6 else None
    return {"temp_c": temp_c, "temp_c_precise": precise, "max6h_c": max6}


def parse_sky_wind(metar: str) -> tuple[float | None, bool, float | None]:
    """Cloud fraction of the thickest layer (0-1), convective-cloud flag, wind speed in knots."""
    body = metar.split(" RMK ")[0]
    layers = _CLOUD.findall(body)
    if layers:
        cover = max(_OKTAS[k] for k, _, _ in layers) / 8.0
    elif _CLEAR.search(body):
        cover = 0.0
    else:
        cover = None
    w = _WIND.search(body)
    wind = float(w.group(2)) * (KT_PER_MPS if w.group(3) == "MPS" else 1.0) if w else None
    return cover, any(c for _, _, c in layers), wind


def _params(icao: str, year: int) -> dict:
    end = min(date(year + 1, 1, 1), date.today())
    return {
        "station": icao,
        "data": ["tmpf", "metar"],
        "year1": year,
        "month1": 1,
        "day1": 1,
        "year2": end.year,
        "month2": end.month,
        "day2": end.day,
        "tz": "Etc/UTC",
        "format": "onlycomma",
        "latlon": "yes",
        "elev": "yes",
        "missing": "empty",
        "trace": "empty",
        "report_type": [3, 4],
    }


def _parse_csv(text: str) -> pl.DataFrame:
    """Tolerant CSV parse: raw METAR fields may contain commas; malformed rows are dropped and counted."""
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return pl.DataFrame()
    header, body = rows[0], rows[1:]
    good = [r for r in body if len(r) == len(header)]
    if len(good) < len(body):
        log.warning("dropped %d malformed rows", len(body) - len(good))
    return pl.DataFrame(good, schema=header, orient="row")


def fetch_station_year(icao: str, year: int) -> pl.DataFrame:
    """Raw rows of one station-year; past years are cached as parquet, the current year is re-fetched."""
    path = raw_dir() / icao / f"{year}.parquet"
    if path.exists() and year < date.today().year:
        return pl.read_parquet(path)
    for attempt in range(6):
        text = _client.get_text(ASOS, _params(icao, year))
        if not text.startswith("Too many requests"):
            break
        time.sleep(30 * (attempt + 1))
    else:
        raise httpx.HTTPError(f"IEM throttled {icao} {year}")
    df = _parse_csv(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return df


def download(icaos: list[str], first_year: int = FIRST_YEAR) -> None:
    for icao in icaos:
        for year in range(first_year, date.today().year + 1):
            try:
                df = fetch_station_year(icao, year)
                log.info("%s %d: %d rows", icao, year, df.height)
            except (httpx.HTTPError, pl.exceptions.PolarsError) as exc:
                log.warning("%s %d failed (rerun resumes): %r", icao, year, exc)


def flag_routine(df: pl.DataFrame) -> pl.DataFrame:
    """Routine reports are at a station's scheduled minute(s): any minute holding >= 20% of its reports."""
    share = (
        df.group_by("station_icao", "minute")
        .len()
        .with_columns(share=pl.col("len") / pl.col("len").sum().over("station_icao"))
    )
    sched = share.filter(pl.col("share") >= 0.2).select("station_icao", "minute").with_columns(is_routine=pl.lit(True))
    return df.join(sched, on=["station_icao", "minute"], how="left").with_columns(
        is_routine=pl.col("is_routine").fill_null(False)
    )


def decode(raw: pl.DataFrame) -> pl.DataFrame:
    """raw: station_icao, valid ('YYYY-MM-DD HH:MM' UTC), metar -> decoded observation table."""
    df = raw.filter(pl.col("metar").is_not_null()).unique(["station_icao", "valid", "metar"])
    parsed = pl.DataFrame(
        [parse_metar(m) for m in df["metar"].to_list()],
        schema={"temp_c": pl.Float64, "temp_c_precise": pl.Float64, "max6h_c": pl.Float64},
    )
    df = df.hstack(parsed).with_columns(
        obs_time=pl.col("valid").str.to_datetime("%Y-%m-%d %H:%M", time_zone="UTC", time_unit="us"),
        minute=pl.col("valid").str.slice(14, 2).cast(pl.Int32),
    )
    return flag_routine(df).drop("valid").sort("station_icao", "obs_time")


def build_obs() -> pl.DataFrame:
    frames = []
    for d in sorted(p for p in raw_dir().iterdir() if p.is_dir()):
        files = [f for f in sorted(d.glob("*.parquet")) if pl.read_parquet_schema(f)]
        if files:
            raw = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
            frames.append(raw.select(station_icao=pl.lit(d.name), valid="valid", metar="metar"))
    df = decode(pl.concat(frames))
    df.write_parquet(obs_path())
    return df
