"""Reproduce the settlement value (local-day max temperature) from METARs; implements `SettlementSource`.

Empirical rules (checked against 10,462 clean resolved events):
- degF markets: max over all reports (routine + SPECI) of the T-group tenths-degC value (fallback: integer group),
  converted to degF and rounded half-up;
- degC markets: max over all reports of the integer main-group degC;
- per-station exception: Moscow UUWW uses routine reports only (its SPECIs never reach the NOAA page).
Candidate rules are all computed so `validate` can show which one each (unit, source) regime follows.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from pmrk.weather.markets import weather_dir, weather_markets_path
from pmrk.weather.stations import with_local_date

ROUTINE_ONLY_STATIONS = frozenset({"UUWW"})
CHOSEN = {"F": "max_f_from_precise", "C": "max_c_int"}
RULES_C = ["max_c_int", "max_c_int_routine", "max_c_from_precise"]
RULES_F = ["max_f_from_precise", "max_f_from_precise_routine", "max_f_from_int", "max_f_from_precise_floor"]
MIN_OBS_PER_DAY = 12


def daily_path() -> Path:
    return weather_dir() / "daily_max.parquet"


def round_half_up(expr: pl.Expr) -> pl.Expr:
    return (expr + 0.5).floor()


def to_f(expr: pl.Expr) -> pl.Expr:
    return expr * 9.0 / 5.0 + 32.0


def build_daily(obs: pl.DataFrame, stations: pl.DataFrame) -> pl.DataFrame:
    """Per station and local day: report counts and every candidate settlement value."""
    o = with_local_date(obs.filter(pl.col("temp_c").is_not_null()), "obs_time", stations, local_time=True)
    best_c = pl.coalesce("temp_c_precise", "temp_c")
    routine_only = pl.col("station_icao").is_in(list(ROUTINE_ONLY_STATIONS))
    daily = (
        o.group_by("station_icao", "local_date")
        .agg(
            n_obs=pl.len(),
            n_routine=pl.col("is_routine").sum(),
            max_c_int=pl.col("temp_c").max(),
            max_c_int_routine=pl.col("temp_c").filter(pl.col("is_routine")).max(),
            max_c_precise=best_c.max(),
            max_c_precise_routine=best_c.filter(pl.col("is_routine")).max(),
            max6h_c=pl.col("max6h_c").max(),
        )
        .with_columns(
            **{
                c: pl.when(routine_only).then(pl.col(f"{c}_routine")).otherwise(pl.col(c))
                for c in ("max_c_int", "max_c_precise")
            },
        )
        .with_columns(
            max_f_from_precise=round_half_up(to_f(pl.col("max_c_precise"))),
            max_f_from_precise_routine=round_half_up(to_f(pl.col("max_c_precise_routine"))),
            max_f_from_int=round_half_up(to_f(pl.col("max_c_int"))),
            max_f_from_precise_floor=to_f(pl.col("max_c_precise")).floor(),
            max_c_from_precise=round_half_up(pl.col("max_c_precise")),
        )
    )
    return daily.sort("station_icao", "local_date")


def validate(wx: pl.DataFrame, daily: pl.DataFrame) -> pl.DataFrame:
    """Winning bucket per event vs every candidate rule, plus `ok` / `our_value` for the chosen rule."""
    winners = wx.filter(
        (pl.col("winner") == 0) & pl.col("station_icao").is_not_null() & pl.col("source").is_in(["wu", "noaa"])
    )
    j = winners.join(daily, on=["station_icao", "local_date"], how="left")
    inside = lambda rule: (
        (pl.col(rule) >= pl.col("bucket_low").fill_null(-10_000))
        & (  # noqa: E731
            pl.col(rule) <= pl.col("bucket_high").fill_null(10_000)
        )
    )
    j = j.with_columns([inside(r).alias(f"ok_{r}") for r in RULES_C + RULES_F])
    f = pl.col("unit") == "F"
    return j.with_columns(
        has_obs=pl.col("n_obs").fill_null(0) >= MIN_OBS_PER_DAY,
        ok=pl.when(f).then(pl.col(f"ok_{CHOSEN['F']}")).otherwise(pl.col(f"ok_{CHOSEN['C']}")),
        our_value=pl.when(f).then(pl.col(CHOSEN["F"])).otherwise(pl.col(CHOSEN["C"])),
        incident=pl.col("archived_copy") | pl.col("opened_after_day_start").fill_null(False),
    )


def reliable_stations(validation: pl.DataFrame, before, min_rate: float = 0.95, min_events: int = 20) -> list[str]:
    """Stations reproduced >= min_rate on clean events before `before` (compute on the tuning half only)."""
    v = validation.filter(pl.col("has_obs") & ~pl.col("incident") & (pl.col("local_date") < before))
    s = v.group_by("station_icao").agg(rate=pl.col("ok").mean(), n=pl.len())
    return s.filter((pl.col("rate") >= min_rate) & (pl.col("n") >= min_events))["station_icao"].to_list()


class MetarSettlement:
    """`SettlementSource`: bucket containing the reconstructed local-day max wins (outcome 0 = "Yes")."""

    name = "metar"

    def __init__(self, daily: pl.DataFrame | None = None, weather_rules: pl.DataFrame | None = None):
        self.daily = daily if daily is not None else pl.read_parquet(daily_path())
        self.rules = weather_rules if weather_rules is not None else pl.read_parquet(weather_markets_path())

    def reproduce(self, markets: pl.DataFrame) -> pl.DataFrame:
        wx = (
            markets.select("market_id")
            .join(self.rules.drop("event_id", strict=False), on="market_id")
            .filter(pl.col("station_icao").is_not_null() & pl.col("source").is_in(["wu", "noaa"]))
        )
        j = wx.join(self.daily, on=["station_icao", "local_date"], how="left").filter(
            pl.col("n_obs").fill_null(0) >= MIN_OBS_PER_DAY
        )
        value = pl.when(pl.col("unit") == "F").then(pl.col(CHOSEN["F"])).otherwise(pl.col(CHOSEN["C"]))
        inside = (value >= pl.col("bucket_low").fill_null(-10_000)) & (value <= pl.col("bucket_high").fill_null(10_000))
        return j.select("market_id", winner_reproduced=pl.when(inside).then(0).otherwise(1).cast(pl.Int8))
