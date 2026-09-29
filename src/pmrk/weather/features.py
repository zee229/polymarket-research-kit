"""Lookahead-safe features for (station, local day D, decision time T) queries.

Invariants (tested in tests/test_weather_lookahead.py):
- the forecast run used at T is the latest with `init + delay <= T` whose steps reach the end of day D;
- the remaining-hours forecast uses only that run's steps with valid_time > T;
- an observation is used only if `obs_time + obs_lag <= T` (dissemination lag, default 10 min);
- station bias uses days in (D-34, D-4]; station climatologies use only days before the fit cutoff.
Temperatures stay in degC; conversion to each market's unit happens per event when bucketizing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import polars as pl

from pmrk.weather.forecasts import DELAYS
from pmrk.weather.metar import parse_sky_wind
from pmrk.weather.settlement import MIN_OBS_PER_DAY, ROUTINE_ONLY_STATIONS
from pmrk.weather.stations import local_day_bounds, with_local_date

OBS_LAG = timedelta(minutes=10)
BIAS_WINDOW_DAYS = 30
BIAS_GAP_DAYS = 4
STEP_H = 3
MODELS = list(DELAYS)


# ---------------------------------------------------------------------------------------------- truth / obs


def truth_table(daily: pl.DataFrame, obs: pl.DataFrame) -> pl.DataFrame:
    """Daily max in degC (`y_c`, tenths where the station reports the T-group) and its rounding half-width."""
    tgroup = obs.group_by("station_icao").agg(has_tgroup=pl.col("temp_c_precise").is_not_null().mean() > 0.5)
    return (
        daily.join(tgroup, on="station_icao", how="left")
        .select(
            "station_icao",
            "local_date",
            "n_obs",
            y_c=pl.col("max_c_precise"),
            y_halfwidth=pl.when(pl.col("has_tgroup")).then(0.05).otherwise(0.5),
        )
        .filter(pl.col("y_c").is_not_null() & (pl.col("n_obs") >= MIN_OBS_PER_DAY))
    )


def obs_local(obs: pl.DataFrame, stations: pl.DataFrame) -> pl.DataFrame:
    return with_local_date(obs, "obs_time", stations, local_time=True)


def _obs_values(obs: pl.DataFrame, lag: timedelta) -> pl.DataFrame:
    keep = ~pl.col("station_icao").is_in(list(ROUTINE_ONLY_STATIONS)) | pl.col("is_routine")
    return (
        obs.filter(keep)
        .with_columns(v=pl.coalesce("temp_c_precise", "temp_c"), usable_from=pl.col("obs_time") + lag)
        .filter(pl.col("v").is_not_null())
    )


def obs_features(q: pl.DataFrame, obs: pl.DataFrame, lag: timedelta = OBS_LAG) -> pl.DataFrame:
    """Max so far on day D, last value, count and last usable time, as of T (settlement-consistent degC)."""
    o = _obs_values(obs, lag).select("station_icao", "local_date", "usable_from", "v")
    j = q.select("qid", "station_icao", "local_date", "decision_time").join(o, on=["station_icao", "local_date"])
    agg = (
        j.filter(pl.col("usable_from") <= pl.col("decision_time"))
        .sort("usable_from")
        .group_by("qid")
        .agg(
            obs_max=pl.col("v").max(),
            obs_last=pl.col("v").last(),
            obs_n=pl.len(),
            obs_last_time=pl.col("usable_from").last(),
        )
    )
    return q.join(agg, on="qid", how="left").with_columns(pl.col("obs_n").fill_null(0))


def remaining_max(q: pl.DataFrame, obs: pl.DataFrame, lag: timedelta = OBS_LAG) -> pl.DataFrame:
    """Training target for intraday fits: max of day-D reports usable after T (never a feature)."""
    o = _obs_values(obs, lag).select("station_icao", "local_date", "usable_from", "v")
    j = q.select("qid", "station_icao", "local_date", "decision_time").join(o, on=["station_icao", "local_date"])
    rem = j.filter(pl.col("usable_from") > pl.col("decision_time")).group_by("qid").agg(rem_max=pl.col("v").max())
    return q.join(rem, on="qid", how="left")


def obs_extra(q: pl.DataFrame, obs: pl.DataFrame, lag: timedelta = OBS_LAG) -> pl.DataFrame:
    """3 h temperature trend, and cloud / convective / wind from the latest usable METAR on day D."""
    o = _obs_values(obs, lag)
    sky = pl.DataFrame(
        [parse_sky_wind(m) for m in o["metar"].to_list()],
        orient="row",
        schema={"cloud": pl.Float64, "convective": pl.Boolean, "wind_kt": pl.Float64},
    )
    meta = o.select("station_icao", "local_date", "usable_from").hstack(sky)
    qq = q.select("qid", "station_icao", "local_date", "decision_time").sort("decision_time")
    by = ["station_icao", "local_date"]
    last = qq.join_asof(
        meta.sort("usable_from"),
        left_on="decision_time",
        right_on="usable_from",
        by=by,
        strategy="backward",
        check_sortedness=False,
    ).select("qid", "cloud", "convective", "wind_kt")
    lag3 = (
        qq.with_columns(t3=pl.col("decision_time") - pl.duration(hours=3))
        .sort("t3")
        .join_asof(
            o.select(*by, "usable_from", "v").sort("usable_from"),
            left_on="t3",
            right_on="usable_from",
            by=by,
            strategy="backward",
            check_sortedness=False,
        )
        .select("qid", v_3h=pl.col("v"))
    )
    return (
        q.join(last, on="qid", how="left")
        .join(lag3, on="qid", how="left")
        .with_columns(trend_3h=pl.col("obs_last") - pl.col("v_3h"))
        .drop("v_3h")
    )


def local_context(q: pl.DataFrame, stations: pl.DataFrame) -> pl.DataFrame:
    """Hours from T to the local end of day D and the local (fractional) hour of T."""
    parts = []
    for (tz,), sub in q.join(stations.select("station_icao", "tz"), on="station_icao").group_by("tz"):
        eod = (
            (pl.col("local_date") + pl.duration(days=1))
            .cast(pl.Datetime("us"))
            .dt.replace_time_zone(tz, ambiguous="earliest", non_existent="null")
            .dt.convert_time_zone("UTC")
        )
        local = pl.col("decision_time").dt.convert_time_zone(tz)
        parts.append(
            sub.with_columns(
                hours_to_eod=(eod - pl.col("decision_time")).dt.total_minutes() / 60.0,
                local_hour=local.dt.hour().cast(pl.Float64) + local.dt.minute() / 60.0,
            ).drop("tz")
        )
    return pl.concat(parts, how="diagonal_relaxed")


def training_queries(truth: pl.DataFrame, stations: pl.DataFrame, min_date: date | None = None) -> pl.DataFrame:
    """Decision grid per station-day for fitting: every 6 h before day D, hourly on day D (local)."""
    if min_date is not None:
        truth = truth.filter(pl.col("local_date") >= min_date)
    offsets = [-60, -54, -48, -42, -36, -30, -24, -18, -12, -6, *range(24)]
    parts = []
    for (tz,), sub in (
        truth.select("station_icao", "local_date")
        .join(stations.select("station_icao", "tz"), on="station_icao")
        .group_by("tz")
    ):
        base = sub.with_columns(off=pl.lit(offsets)).explode("off")
        parts.append(
            base.with_columns(
                decision_time=(pl.col("local_date").cast(pl.Datetime("us")) + pl.duration(hours=pl.col("off")))
                .dt.replace_time_zone(tz, ambiguous="earliest", non_existent="null")
                .dt.convert_time_zone("UTC"),
            ).drop("tz", "off")
        )
    q = pl.concat(parts).drop_nulls("decision_time")
    return q.with_row_index("qid").with_columns(pl.col("qid").cast(pl.Int64))


# ---------------------------------------------------------------------------------------------- forecasts


def build_run_tables(runs: pl.DataFrame, stations: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(steps with local date, per (station, model, init, D) full-day max). Runs not reaching the end of D drop."""
    steps = with_local_date(runs, "valid_time", stations)
    eod = local_day_bounds(steps.select("station_icao", "local_date").unique(), stations)
    daily = (
        steps.group_by("station_icao", "model", "init_time", "local_date")
        .agg(
            fc_max=pl.col("temp_c").max(),
            last_valid=pl.col("valid_time").max(),
            available_time=pl.col("available_time").first(),
        )
        .join(eod, on=["station_icao", "local_date"])
        .filter(pl.col("last_valid") >= pl.col("day_end_utc") - pl.duration(hours=STEP_H))
    )
    return steps, daily


def run_bias(daily: pl.DataFrame, truth: pl.DataFrame) -> pl.DataFrame:
    """Trailing (D-34, D-4] mean error and MSE of each model's 00Z run of D-1, forward-filled up to 7 days."""
    ref = daily.filter(
        (pl.col("init_time").dt.hour() == 0)
        & (pl.col("init_time").dt.date() == pl.col("local_date") - pl.duration(days=1))
    )
    err = (
        ref.select("station_icao", "model", "local_date", "fc_max")
        .join(truth.select("station_icao", "local_date", "y_c"), on=["station_icao", "local_date"])
        .with_columns(e=pl.col("y_c") - pl.col("fc_max"), dt=pl.col("local_date").cast(pl.Datetime("ms")))
        .sort("dt")
    )
    rolled = err.rolling(
        index_column="dt",
        period=f"{BIAS_WINDOW_DAYS}d",
        offset=f"-{BIAS_WINDOW_DAYS + BIAS_GAP_DAYS}d",
        group_by=["station_icao", "model"],
        closed="right",
    ).agg(bias=pl.col("e").mean(), mse=(pl.col("e") ** 2).mean())
    grid = daily.select("station_icao", "model", "local_date").unique()
    out = grid.join(
        rolled.with_columns(local_date=pl.col("dt").dt.date()).drop("dt"),
        on=["station_icao", "model", "local_date"],
        how="left",
    ).sort("local_date")
    return out.with_columns(pl.col("bias", "mse").forward_fill(limit=7).over("station_icao", "model"))


def forecast_features(q: pl.DataFrame, daily: pl.DataFrame, steps: pl.DataFrame, bias: pl.DataFrame) -> pl.DataFrame:
    """Per model, the latest run usable at T: full-day max, remaining-hours max, age; bias-corrected."""
    cand = (
        q.select("qid", "station_icao", "local_date", "decision_time")
        .join(
            daily.select("station_icao", "model", "init_time", "local_date", "fc_max", "available_time"),
            on=["station_icao", "local_date"],
        )
        .filter(pl.col("available_time") <= pl.col("decision_time"))
    )
    best = cand.sort("init_time").group_by("qid", "model").last()
    rem = (
        best.select("qid", "model", "station_icao", "local_date", "init_time", "decision_time")
        .join(
            steps.select("station_icao", "model", "init_time", "local_date", "valid_time", "temp_c"),
            on=["station_icao", "model", "init_time", "local_date"],
        )
        .filter(pl.col("valid_time") > pl.col("decision_time"))
    )
    rem = rem.group_by("qid", "model").agg(fc_rem=pl.col("temp_c").max())
    best = (
        best.join(rem, on=["qid", "model"], how="left")
        .join(bias, on=["station_icao", "model", "local_date"], how="left")
        .with_columns(
            fc_corr=pl.col("fc_max") + pl.col("bias").fill_null(0.0),
            rem_corr=pl.col("fc_rem") + pl.col("bias").fill_null(0.0),
            age_h=(pl.col("decision_time") - pl.col("init_time")).dt.total_minutes() / 60.0,
        )
    )
    if best.is_empty():
        return q
    wide = best.pivot(on="model", index="qid", values=["fc_max", "fc_corr", "rem_corr", "age_h", "mse", "init_time"])
    return q.join(wide, on="qid", how="left")


# ---------------------------------------------------------------------------------------------- climatology


def build_day_profile(obs: pl.DataFrame) -> pl.DataFrame:
    """Per station-day: local hour of the max and the remaining rise after the end of each local hour 0..23."""
    o = obs.with_columns(v=pl.coalesce("temp_c_precise", "temp_c"), h=pl.col("local_time").dt.hour()).filter(
        pl.col("v").is_not_null()
    )
    day = (
        o.group_by("station_icao", "local_date")
        .agg(y=pl.col("v").max(), peak_hour=pl.col("h").sort_by("v", descending=True).first(), n=pl.len())
        .filter(pl.col("n") >= MIN_OBS_PER_DAY)
    )
    hourly = o.group_by("station_icao", "local_date", "h").agg(hmax=pl.col("v").max())
    hours = pl.DataFrame({"h": list(range(24))}, schema={"h": pl.Int8})
    full = day.select("station_icao", "local_date").join(hours, how="cross")
    prof = (
        full.join(hourly.with_columns(pl.col("h").cast(pl.Int8)), on=["station_icao", "local_date", "h"], how="left")
        .sort("h")
        .with_columns(
            so_far=pl.col("hmax")
            .cum_max()
            .over("station_icao", "local_date")
            .forward_fill()
            .over("station_icao", "local_date")
        )
    )
    prof = prof.join(day, on=["station_icao", "local_date"]).with_columns(rise=pl.col("y") - pl.col("so_far"))
    return prof.select(
        "station_icao", "local_date", "h", "rise", "peak_hour", month=pl.col("local_date").dt.month().cast(pl.Int8)
    )


def climatology(profile: pl.DataFrame, cutoff: date) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(station, month) mean peak hour and (station, month, hour) mean remaining rise, from days < cutoff."""
    p = profile.filter(pl.col("local_date") < cutoff)
    peak = (
        p.filter(pl.col("h") == 0)
        .group_by("station_icao", "month")
        .agg(clim_peak_hour=pl.col("peak_hour").cast(pl.Float64).mean())
    )
    rise = p.group_by("station_icao", "month", "h").agg(clim_rise=pl.col("rise").mean())
    return peak, rise


def attach_climatology(f: pl.DataFrame, peak: pl.DataFrame, rise: pl.DataFrame) -> pl.DataFrame:
    f = f.with_columns(
        month=pl.col("local_date").dt.month().cast(pl.Int8), h=pl.col("local_hour").floor().cast(pl.Int8)
    )
    return (
        f.join(peak, on=["station_icao", "month"], how="left")
        .join(rise, on=["station_icao", "month", "h"], how="left")
        .drop("month", "h")
    )


# ---------------------------------------------------------------------------------------------- assembly


@dataclass(frozen=True)
class Context:
    """Everything the feature builder reads, loaded once."""

    stations: pl.DataFrame
    obs: pl.DataFrame  # with local_date / local_time
    truth: pl.DataFrame
    run_daily: pl.DataFrame
    run_steps: pl.DataFrame
    bias: pl.DataFrame
    profile: pl.DataFrame


def build_features(
    q: pl.DataFrame, ctx: Context, with_target: bool = True, obs_lag: timedelta = OBS_LAG
) -> pl.DataFrame:
    """q: qid, station_icao, local_date, decision_time (UTC)."""
    f = forecast_features(q, ctx.run_daily, ctx.run_steps, ctx.bias)
    f = obs_features(f, ctx.obs, obs_lag)
    f = obs_extra(f, ctx.obs, obs_lag)
    f = local_context(f, ctx.stations)
    if with_target:
        f = remaining_max(f, ctx.obs, obs_lag).join(
            ctx.truth.select("station_icao", "local_date", "y_c", "y_halfwidth"),
            on=["station_icao", "local_date"],
            how="left",
        )
    corr = [f"fc_corr_{m}" for m in MODELS if f"fc_corr_{m}" in f.columns]
    rem = [f"rem_corr_{m}" for m in MODELS if f"rem_corr_{m}" in f.columns]
    if not corr:
        return f.with_columns(ens_mean=pl.lit(None, pl.Float64))
    f = f.with_columns(
        ens_mean=pl.mean_horizontal(corr), ens_rem=pl.mean_horizontal(rem) if rem else pl.lit(None, pl.Float64)
    )
    if len(corr) == 2:
        f = f.with_columns(
            ec_minus_gfs=pl.col(corr[0]) - pl.col(corr[1]),
            ens_sd=(pl.col(corr[0]) - pl.col(corr[1])).abs() / np.sqrt(2),
        )
    return f
