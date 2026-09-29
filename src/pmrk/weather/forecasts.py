"""Exact single-run 2 m temperature at station points from the public AWS archives (no API quota).

- ECMWF IFS 0.25 deg open data: `s3://ecmwf-forecasts/{date}/{HH}z/ifs/0p25/{stream}/`. 00/12Z are stream `oper`;
  06/18Z were `scda` until 2026-05-11 and `oper` from 2026-05-12, so both are tried.
- NOAA GFS 0.5 deg: `s3://noaa-gfs-bdp-pds/gfs.{date}/{HH}/atmos/gfs.t{HH}z.pgrb2.0p50.fFFF` (0.25 deg was rejected:
  its 2 m message is ~3 MB vs 0.16 MB).

Only the 2 m temperature message is range-requested via the `.index`/`.idx` files, decoded with ecCodes (the
`[weather]` extra) and interpolated to station points; the global field is discarded. Every row keeps the exact
`init_time`, so `available_time = init + delay` is exact up to the (conservative) delay assumption.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import numpy as np
import polars as pl
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from pmrk.config import USER_AGENT
from pmrk.weather.markets import weather_dir

log = logging.getLogger(__name__)

FIRST_DAY = date(2024, 3, 14)
CYCLES = (0, 6, 12, 18)
STEPS = tuple(range(0, 91, 3))
# Conservative delay from init to the full run being downloadable.
DELAYS = {"ecmwf_ifs025": timedelta(hours=8), "gfs_050": timedelta(hours=5)}

ECMWF = "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"
GFS = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"

_http = httpx.Client(timeout=120, headers={"User-Agent": USER_AGENT})


def raw_dir() -> Path:
    return weather_dir() / "runs"


def runs_path() -> Path:
    return weather_dir() / "run_forecasts.parquet"


class Missing(Exception):
    """File not in the archive (run or step not published)."""


@retry(
    retry=retry_if_exception_type(httpx.TransportError),
    wait=wait_exponential(multiplier=2, max=180),
    stop=stop_after_attempt(10),
    reraise=True,
)
def _get(url: str, rng: tuple[int, int] | None = None) -> bytes:
    """GET with S3 503 SlowDown / 5xx retried; 403/404 mean 'not published'."""
    r = _http.get(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {})
    if r.status_code in (403, 404):
        raise Missing(url)
    if r.status_code >= 500 or r.status_code == 429:
        raise httpx.TransportError(f"{r.status_code} {url}")
    r.raise_for_status()
    return r.content


def _ecmwf_range(init: datetime, step: int) -> tuple[str, tuple[int, int]]:
    streams = ("oper",) if init.hour in (0, 12) else ("scda", "oper")
    for stream in streams:
        base = f"{ECMWF}/{init:%Y%m%d}/{init:%H}z/ifs/0p25/{stream}/{init:%Y%m%d%H}0000-{step}h-{stream}-fc"
        try:
            index = _get(base + ".index").decode()
        except Missing:
            continue
        for line in index.splitlines():
            rec = json.loads(line)
            if rec.get("param") == "2t":
                off, ln = int(rec["_offset"]), int(rec["_length"])
                return base + ".grib2", (off, off + ln - 1)
    raise Missing(f"no 2t for {init} +{step}h")


def _gfs_range(init: datetime, step: int) -> tuple[str, tuple[int, int]]:
    base = f"{GFS}/gfs.{init:%Y%m%d}/{init:%H}/atmos/gfs.t{init:%H}z.pgrb2.0p50.f{step:03d}"
    lines = _get(base + ".idx").decode().splitlines()
    for i, line in enumerate(lines):
        if ":TMP:2 m above ground:" in line:
            start = int(line.split(":")[1])
            end = int(lines[i + 1].split(":")[1]) - 1 if i + 1 < len(lines) else start + 5_000_000
            return base, (start, end)
    raise Missing(f"no TMP:2 m in {base}.idx")


RANGES = {"ecmwf_ifs025": _ecmwf_range, "gfs_050": _gfs_range}


def _decode(msg: bytes) -> tuple[np.ndarray, dict]:
    import eccodes  # [weather] extra

    h = eccodes.codes_new_from_message(msg)
    try:
        keys = (
            "Ni",
            "Nj",
            "latitudeOfFirstGridPointInDegrees",
            "longitudeOfFirstGridPointInDegrees",
            "jDirectionIncrementInDegrees",
            "iDirectionIncrementInDegrees",
            "jScansPositively",
        )
        meta = {k: eccodes.codes_get(h, k) for k in keys}
        vals = eccodes.codes_get_values(h).reshape(meta["Nj"], meta["Ni"])
    finally:
        eccodes.codes_release(h)
    return vals, meta


class PointInterp:
    """Bilinear weights to station points, restricted to land cells when any neighbour is land.

    Plain bilinear mixes sea temperatures into coastal airports (seen: KSFO 3.7 degC off Open-Meteo).
    """

    def __init__(self, meta: dict, land: np.ndarray, lats: np.ndarray, lons: np.ndarray):
        ni, nj = meta["Ni"], meta["Nj"]
        lat0, lon0 = meta["latitudeOfFirstGridPointInDegrees"], meta["longitudeOfFirstGridPointInDegrees"]
        dlat, dlon = meta["jDirectionIncrementInDegrees"], meta["iDirectionIncrementInDegrees"]
        y = ((lat0 - lats) if meta["jScansPositively"] == 0 else (lats - lat0)) / dlat
        x = ((lons - lon0) % 360.0) / dlon
        y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
        fy, fx = y - y0, x - x0
        y1 = np.clip(y0 + 1, 0, nj - 1)
        ys = np.stack([y0, y0, y1, y1], axis=1)
        xs = np.stack([x0 % ni, (x0 + 1) % ni, x0 % ni, (x0 + 1) % ni], axis=1)
        w = np.stack([(1 - fy) * (1 - fx), (1 - fy) * fx, fy * (1 - fx), fy * fx], axis=1)
        w_land = np.where(land[ys, xs] > 0.5, w, 0.0)
        w = np.where((w_land.sum(axis=1) > 1e-6)[:, None], w_land, w)
        self.ys, self.xs, self.w = ys, xs, w / w.sum(axis=1, keepdims=True)

    def __call__(self, vals: np.ndarray) -> np.ndarray:
        return (vals[self.ys, self.xs] * self.w).sum(axis=1)


def _land_mask(model: str) -> tuple[np.ndarray, dict]:
    init = datetime(2026, 6, 1, 0, tzinfo=UTC)
    if model == "ecmwf_ifs025":
        base = f"{ECMWF}/{init:%Y%m%d}/00z/ifs/0p25/oper/{init:%Y%m%d}000000-0h-oper-fc"
        recs = [json.loads(x) for x in _get(base + ".index").decode().splitlines()]
        rec = next(r for r in recs if r.get("param") == "lsm")
        msg = _get(base + ".grib2", (int(rec["_offset"]), int(rec["_offset"]) + int(rec["_length"]) - 1))
    else:
        base = f"{GFS}/gfs.{init:%Y%m%d}/00/atmos/gfs.t00z.pgrb2.0p50.f000"
        lines = _get(base + ".idx").decode().splitlines()
        i = next(i for i, x in enumerate(lines) if ":LAND:surface:" in x)
        msg = _get(base, (int(lines[i].split(":")[1]), int(lines[i + 1].split(":")[1]) - 1))
    return _decode(msg)


def fetch_run(model: str, init: datetime, st: pl.DataFrame, interp: PointInterp) -> pl.DataFrame:
    frames = []
    for step in STEPS:
        try:
            url, rng = RANGES[model](init, step)
            vals, _ = _decode(_get(url, rng))
        except Missing:
            continue
        frames.append(
            pl.DataFrame({"station_icao": st["station_icao"], "temp_c": interp(vals) - 273.15}).with_columns(
                valid_time=pl.lit(init + timedelta(hours=step)), step=pl.lit(step, pl.Int16)
            )
        )
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames).with_columns(model=pl.lit(model), init_time=pl.lit(init))


def fetch_day(model: str, day: date, st: pl.DataFrame, interp: PointInterp) -> str:
    path = raw_dir() / model / f"{day}.parquet"
    if path.exists():
        return "cached"
    parts = [fetch_run(model, datetime(day.year, day.month, day.day, h, tzinfo=UTC), st, interp) for h in CYCLES]
    parts = [p for p in parts if p.height]
    if not parts:
        return "missing"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pl.concat(parts).write_parquet(tmp)
    tmp.replace(path)
    return f"{sum(p['init_time'].n_unique() for p in parts)} runs"


def download(
    stations: pl.DataFrame, models: list[str], first: date = FIRST_DAY, last: date | None = None, workers: int = 16
) -> None:
    """One parquet per (model, init day); existing files are skipped. S3 may answer 503 SlowDown under load:
    rerun with fewer workers, downloads resume per day."""
    st = stations.select("station_icao", "lat", "lon")
    last = last or date.today() - timedelta(days=1)
    days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    for m in models:
        land, meta = _land_mask(m)
        interp = PointInterp(meta, land, st["lat"].to_numpy(), st["lon"].to_numpy())
        jobs = [d for d in days if not (raw_dir() / m / f"{d}.parquet").exists()]
        log.info("%s: %d days to fetch", m, len(jobs))
        with ThreadPoolExecutor(workers) as pool:
            futs = {pool.submit(fetch_day, m, d, st, interp): d for d in jobs}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    res = fut.result()
                except (httpx.HTTPError, Missing, ValueError) as exc:
                    res = f"FAILED {exc!r}"
                if i % 50 == 0 or res.startswith(("FAILED", "missing")):
                    log.info("[%d/%d] %s %s: %s", i, len(jobs), m, futs[fut], res)


def build_runs() -> pl.DataFrame:
    """All downloaded steps with `available_time = init_time + DELAYS[model]`."""
    df = pl.scan_parquet(str(raw_dir() / "*" / "*.parquet")).collect()
    delay = pl.col("model").replace_strict({m: d for m, d in DELAYS.items()}, return_dtype=pl.Duration("us"))
    df = df.with_columns(available_time=pl.col("init_time") + delay)
    df.write_parquet(runs_path())
    return df
