"""Snapshot targets and decision grids with lookahead guards.

Guard (applied per market): a time t is usable only if `created_ts <= t <= known_ts - buffer`, where `known_ts` is
the earliest of the scheduled end, the actual close and the oracle resolution. This drops everything at or after
the moment the outcome could have been known, and the last `buffer` (default 1 h) before an actual early close.
"""

from __future__ import annotations

import re
from datetime import timedelta

import polars as pl

DEFAULT_HORIZONS: dict[str, timedelta] = {
    "30d": timedelta(days=30),
    "7d": timedelta(days=7),
    "1d": timedelta(days=1),
    "6h": timedelta(hours=6),
    "1h": timedelta(hours=1),
}
LIFE_FRACTIONS: dict[str, float] = {"life10": 0.1, "life50": 0.5, "life90": 0.9}
LOOKAHEAD_BUFFER = timedelta(hours=1)

_UNITS = {"m": "minutes", "h": "hours", "d": "days"}


def parse_duration(text: str) -> timedelta:
    """'90m', '6h', '7d' -> timedelta."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([mhd])\s*", text)
    if not m:
        raise ValueError(f"bad duration {text!r}: use e.g. 90m, 6h, 7d")
    return timedelta(**{_UNITS[m.group(2)]: float(m.group(1))})


def usable(time_col: str, buffer: timedelta = LOOKAHEAD_BUFFER) -> pl.Expr:
    """True where `time_col` passes the per-market guard (needs created_ts and known_ts columns)."""
    t = pl.col(time_col)
    after_creation = pl.col("created_ts").is_null() | (t >= pl.col("created_ts"))
    return after_creation & (t <= pl.col("known_ts") - buffer)


def snapshot_targets(
    markets: pl.DataFrame,
    horizons: dict[str, timedelta] | None = None,
    life_fractions: dict[str, float] | None = None,
    buffer: timedelta = LOOKAHEAD_BUFFER,
) -> pl.DataFrame:
    """(market_id, snapshot, target_ts): `end_ts - h` per horizon and `created + f * life` per fraction, guarded."""
    horizons = DEFAULT_HORIZONS if horizons is None else horizons
    life_fractions = LIFE_FRACTIONS if life_fractions is None else life_fractions
    base = markets.select("market_id", "created_ts", "end_ts", "known_ts").filter(pl.col("end_ts").is_not_null())
    parts = [base.with_columns(snapshot=pl.lit(k), target_ts=pl.col("end_ts") - v) for k, v in horizons.items()]
    life = pl.col("end_ts") - pl.col("created_ts")
    parts += [
        base.filter(pl.col("created_ts") < pl.col("end_ts")).with_columns(
            snapshot=pl.lit(k), target_ts=pl.col("created_ts") + life * f
        )
        for k, f in life_fractions.items()
    ]
    if not parts:
        return pl.DataFrame(schema={"market_id": pl.Utf8, "snapshot": pl.Utf8, "target_ts": pl.Datetime("us", "UTC")})
    out = pl.concat(parts).with_columns(pl.col("target_ts").dt.cast_time_unit("us"))
    return out.filter(usable("target_ts", buffer)).select("market_id", "snapshot", "target_ts")


def decision_grid(
    markets: pl.DataFrame, every: timedelta = timedelta(hours=1), buffer: timedelta = LOOKAHEAD_BUFFER
) -> pl.DataFrame:
    """(event_id, decision_time) every `every` from the first full step after opening until the last guard."""
    ev = (
        markets.group_by("event_id")
        .agg(
            t0=pl.min_horizontal(pl.col("start_ts").min(), pl.col("created_ts").min()),
            t1=pl.col("known_ts").max() - buffer,
        )
        .drop_nulls()
        .with_columns(t0=(pl.col("t0") + every).dt.truncate(every))
        .filter(pl.col("t1") >= pl.col("t0"))
    )
    grid = ev.with_columns(decision_time=pl.datetime_ranges("t0", "t1", interval=every)).explode(
        "decision_time", empty_as_null=False
    )
    return grid.select("event_id", pl.col("decision_time").dt.cast_time_unit("us"))


def market_rows(
    events: pl.DataFrame, markets: pl.DataFrame, time_col: str = "decision_time", buffer: timedelta = LOOKAHEAD_BUFFER
) -> pl.DataFrame:
    """Expand (event_id, time) to every market of the event that passes the guard at that time."""
    cols = ["event_id", "market_id", "created_ts", "known_ts"]
    return events.join(markets.select(cols), on="event_id").filter(usable(time_col, buffer))
