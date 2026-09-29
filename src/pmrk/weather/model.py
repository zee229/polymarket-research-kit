"""Walk-forward EMOS fits and the `WeatherModel` ProbabilityModel.

Walk-forward: one refit per calendar month; each fit uses only station-days before `month_start - 4 days`, so
every training target was fully observed before any decision in the test month. Intraday climatologies are
recomputed per fit from days before the same cutoff.
"""

from __future__ import annotations

import logging
import pickle
from datetime import date, timedelta
from functools import cached_property
from pathlib import Path

import numpy as np
import polars as pl

from pmrk.weather import emos
from pmrk.weather.features import (
    BIAS_GAP_DAYS,
    OBS_LAG,
    Context,
    attach_climatology,
    build_day_profile,
    build_features,
    build_run_tables,
    climatology,
    obs_local,
    run_bias,
    training_queries,
    truth_table,
)
from pmrk.weather.forecasts import runs_path
from pmrk.weather.markets import load as load_weather_markets
from pmrk.weather.markets import settleable, weather_dir
from pmrk.weather.metar import obs_path
from pmrk.weather.settlement import daily_path
from pmrk.weather.stations import load_stations, local_day_bounds

log = logging.getLogger(__name__)

FIRST_TEST_MONTH = date(2024, 9, 1)
MIN_DATE = date(2024, 4, 20)  # first AWS runs 2024-03-14 + the 34-day bias window
CLIM_HALF_WINDOW = 15


def fits_path() -> Path:
    return weather_dir() / "emos_fits.pkl"


def grid_path() -> Path:
    return weather_dir() / "grid_features.parquet"


def load_context() -> Context:
    stations = load_stations()
    obs = obs_local(pl.read_parquet(obs_path()), stations)
    truth = truth_table(pl.read_parquet(daily_path()), obs)
    steps, daily = build_run_tables(pl.read_parquet(runs_path()), stations)
    return Context(stations, obs, truth, daily, steps, run_bias(daily, truth), build_day_profile(obs))


def features_for(q: pl.DataFrame, ctx: Context, with_target: bool, obs_lag: timedelta = OBS_LAG) -> pl.DataFrame:
    """Per-station feature build (bounded memory) with the horizon bin attached."""
    parts = []
    for (st,), sub in q.group_by("station_icao"):
        sel = pl.col("station_icao") == st
        sctx = Context(
            ctx.stations,
            ctx.obs.filter(sel),
            ctx.truth.filter(sel),
            ctx.run_daily.filter(sel),
            ctx.run_steps.filter(sel),
            ctx.bias.filter(sel),
            ctx.profile,
        )
        parts.append(build_features(sub, sctx, with_target, obs_lag))
    if not parts:
        return q.clear()
    return pl.concat(parts, how="diagonal_relaxed").with_columns(
        hbin=emos.horizon_bin(pl.col("hours_to_eod"), pl.col("local_hour"), pl.col("obs_n"))
    )


def month_starts(first: date, last: date) -> list[date]:
    out, d = [], first
    while d <= last:
        out.append(d)
        d = date(d.year + (d.month == 12), d.month % 12 + 1, 1)
    return out


def next_month(d: date) -> date:
    return date(d.year + (d.month == 12), d.month % 12 + 1, 1)


def clim_baseline(keys: pl.DataFrame, truth: pl.DataFrame) -> pl.DataFrame:
    """Climatology per station and day-of-year +-15 d, from days at least BIAS_GAP_DAYS before D (any year)."""
    t = truth.select("station_icao", "local_date", "y_c").with_columns(doy=pl.col("local_date").dt.ordinal_day())
    k = keys.select("station_icao", "local_date").unique().with_columns(doy=pl.col("local_date").dt.ordinal_day())
    dd = (pl.col("doy") - pl.col("doy_h")).abs()
    j = k.join(t, on="station_icao", suffix="_h").filter(
        (pl.col("local_date_h") <= pl.col("local_date") - pl.duration(days=BIAS_GAP_DAYS))
        & ((dd <= CLIM_HALF_WINDOW) | (dd >= 366 - CLIM_HALF_WINDOW))
    )
    clim = j.group_by("station_icao", "local_date").agg(
        clim_mu=pl.col("y_c").mean(), clim_sigma=pl.col("y_c").std(), clim_n=pl.len()
    )
    return clim.filter(pl.col("clim_n") >= 20).drop("clim_n")


def raw_ecmwf_sigma(grid: pl.DataFrame) -> pl.DataFrame:
    """Fixed empirical error spread of raw ECMWF per horizon bin, from pre-test data only (a naive baseline)."""
    train = grid.filter(pl.col("local_date") < FIRST_TEST_MONTH - timedelta(days=BIAS_GAP_DAYS))
    return (
        train.drop_nulls(["fc_max_ecmwf_ifs025", "y_c"])
        .group_by("hbin")
        .agg(raw_sigma=((pl.col("y_c") - pl.col("fc_max_ecmwf_ifs025")) ** 2).mean().sqrt())
    )


def fit_walk_forward(ctx: Context) -> dict:
    """Build the training grid, fit one EMOS set per month, store fits + climatologies + baseline sigma."""
    grid = features_for(training_queries(ctx.truth, ctx.stations, MIN_DATE), ctx, with_target=True)
    grid.write_parquet(grid_path())
    fitted: dict = {"raw_sigma": raw_ecmwf_sigma(grid), "months": {}}
    for m0 in month_starts(FIRST_TEST_MONTH, grid["local_date"].max()):
        cutoff = m0 - timedelta(days=BIAS_GAP_DAYS)
        peak, rise = climatology(ctx.profile, cutoff)
        train = attach_climatology(grid.filter(pl.col("local_date") < cutoff), peak, rise)
        models = emos.fit_all(train)
        fitted["months"][m0] = {"models": models, "peak": peak, "rise": rise}
        log.info("%s: train=%d bins=%d", m0, train.height, len(models))
    fits_path().write_bytes(pickle.dumps(fitted))
    return fitted


def bucket_edges_c(unit: pl.Expr, low: pl.Expr, high: pl.Expr) -> tuple[pl.Expr, pl.Expr]:
    """Continuous degC interval [a, b) whose rounded settlement value falls in the integer bucket [low, high]."""
    lo, hi = low.cast(pl.Float64).fill_null(-1000.0) - 0.5, high.cast(pl.Float64).fill_null(1000.0) + 0.5
    f = unit == "F"
    return (pl.when(f).then((lo - 32.0) / 1.8).otherwise(lo), pl.when(f).then((hi - 32.0) / 1.8).otherwise(hi))


def bucket_probs(rows: pl.DataFrame, mu: str, sigma: str, lower_bound: str | None) -> np.ndarray:
    m = rows[lower_bound].fill_null(-np.inf).to_numpy() if lower_bound else np.full(rows.height, -np.inf)
    return emos.interval_prob(
        rows["a"].to_numpy(),
        rows["b"].to_numpy(),
        rows[mu].to_numpy().astype(float),
        rows[sigma].to_numpy().astype(float),
        m,
    )


def horizon_group(df: pl.DataFrame) -> pl.DataFrame:
    """The six reporting horizons of the case study."""
    h, lh = pl.col("hours_to_eod"), pl.col("local_hour")
    return df.with_columns(
        hgroup=pl.when(h >= 48)
        .then(pl.lit("a: >=48h"))
        .when(h >= 36)
        .then(pl.lit("b: 36-48h"))
        .when(h >= 24)
        .then(pl.lit("c: 24-36h"))
        .when(lh < 10)
        .then(pl.lit("d: day D 00-10"))
        .when(lh < 16)
        .then(pl.lit("e: day D 10-16"))
        .otherwise(pl.lit("f: day D 16-24"))
    )


class WeatherModel:
    """`ProbabilityModel` for daily-max buckets: walk-forward EMOS on exact NWP runs + METARs.

    Also returns `p_clim` / `p_raw` baselines, `hgroup`, `city`, `station_icao` and `day` (the local date, used for
    daily caps and date-block bootstrap). `decision_times` adds triggers one minute after each forecast run becomes
    available and `obs_lag` after each METAR on day D.
    """

    name = "weather_emos"

    def __init__(
        self,
        obs_lag: timedelta = OBS_LAG,
        stations: list[str] | None = None,
        triggers: bool = True,
        ctx: Context | None = None,
        fits: dict | None = None,
        rules: pl.DataFrame | None = None,
    ):
        """`ctx`, `fits` and `rules` default to the stored case-study data; pass them to run on other data."""
        self.obs_lag = obs_lag
        self.triggers = triggers
        self.allowed = set(stations) if stations else None
        self._ctx, self._fits, self._rules = ctx, fits, rules

    @cached_property
    def ctx(self) -> Context:
        return self._ctx if self._ctx is not None else load_context()

    @cached_property
    def fits(self) -> dict:
        return self._fits if self._fits is not None else pickle.loads(fits_path().read_bytes())

    @cached_property
    def rules(self) -> pl.DataFrame:
        wx = self._rules if self._rules is not None else load_weather_markets().filter(settleable())
        if self.allowed is not None:
            wx = wx.filter(pl.col("station_icao").is_in(list(self.allowed)))
        return wx.select("market_id", "city", "station_icao", "unit", "local_date", "bucket_low", "bucket_high")

    def decision_times(self, markets: pl.DataFrame) -> pl.DataFrame:
        if not self.triggers:
            return pl.DataFrame(
                schema={"event_id": pl.Utf8, "decision_time": pl.Datetime("us", "UTC"), "trigger": pl.Utf8}
            )
        ev = (
            markets.select("event_id", "market_id", open_ts=pl.coalesce("start_ts", "created_ts"))
            .join(self.rules, on="market_id")
            .unique("event_id")
            .drop("market_id")
        )
        ev = local_day_bounds(ev, self.ctx.stations)
        runs = ev.join(
            self.ctx.run_daily.select("station_icao", "local_date", "available_time").unique(),
            on=["station_icao", "local_date"],
        ).filter((pl.col("available_time") > pl.col("open_ts")) & (pl.col("available_time") < pl.col("day_end_utc")))
        runs = runs.select(
            "event_id", decision_time=pl.col("available_time") + pl.duration(minutes=1), trigger=pl.lit("model_run")
        )
        metar = ev.join(
            self.ctx.obs.select("station_icao", "local_date", "obs_time"), on=["station_icao", "local_date"]
        ).select("event_id", decision_time=pl.col("obs_time") + self.obs_lag, trigger=pl.lit("metar"))
        return pl.concat([runs, metar]).with_columns(pl.col("decision_time").dt.cast_time_unit("us"))

    def predict(self, queries: pl.DataFrame) -> pl.DataFrame:
        rows = queries.select("event_id", "market_id", "decision_time").join(self.rules, on="market_id")
        if rows.is_empty():
            return rows.with_columns(p_model=pl.lit(None, pl.Float64))
        q = (
            rows.select("event_id", "station_icao", "local_date", "decision_time")
            .unique()
            .with_row_index("qid")
            .with_columns(pl.col("qid").cast(pl.Int64))
        )
        feats = features_for(q, self.ctx, with_target=False, obs_lag=self.obs_lag).with_columns(
            fit_month=pl.col("local_date").dt.truncate("1mo")
        )
        # A resolution-day decision without any METAR after 02:00 local is an archive gap, not information.
        feats = feats.filter(~((pl.col("hours_to_eod") < 24) & (pl.col("local_hour") >= 2) & (pl.col("obs_n") == 0)))
        preds = []
        for (month,), sub in feats.group_by("fit_month"):
            f = self.fits["months"].get(month)
            if f is not None:
                preds.append(emos.predict_all(f["models"], attach_climatology(sub, f["peak"], f["rise"])))
        if not preds:
            return rows.clear().with_columns(p_model=pl.lit(None, pl.Float64))
        p = pl.concat(preds, how="diagonal_relaxed")
        p = p.join(clim_baseline(p, self.ctx.truth), on=["station_icao", "local_date"], how="left").join(
            self.fits["raw_sigma"], on="hbin", how="left"
        )
        keep = [
            "event_id",
            "decision_time",
            "mu",
            "sigma",
            "lower_bound",
            "clim_mu",
            "clim_sigma",
            "fc_max_ecmwf_ifs025",
            "raw_sigma",
            "hours_to_eod",
            "local_hour",
            "hbin",
            "obs_n",
        ]
        out = rows.join(p.select([c for c in keep if c in p.columns]), on=["event_id", "decision_time"])
        a, b = bucket_edges_c(pl.col("unit"), pl.col("bucket_low"), pl.col("bucket_high"))
        out = out.with_columns(a=a, b=b)
        out = out.with_columns(
            p_model=pl.Series(bucket_probs(out, "mu", "sigma", "lower_bound")),
            p_clim=pl.Series(bucket_probs(out, "clim_mu", "clim_sigma", "lower_bound")),
            p_raw=pl.Series(bucket_probs(out, "fc_max_ecmwf_ifs025", "raw_sigma", None)),
        )
        out = horizon_group(out).with_columns(day=pl.col("local_date"))
        return out.select(
            "event_id",
            "market_id",
            "decision_time",
            "p_model",
            "p_clim",
            "p_raw",
            "hgroup",
            "hbin",
            "hours_to_eod",
            "local_hour",
            "city",
            "station_icao",
            "day",
        )
