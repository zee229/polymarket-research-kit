"""Lookahead invariants of the weather features, and the WeatherModel run through the core backtest."""

from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from synth import event, market, write_prices, write_trades

from pmrk.backtest.run import build_panel
from pmrk.polymarket.markets import markets_frame
from pmrk.polymarket.store import load_markets, save_markets
from pmrk.weather.emos import FittedBin
from pmrk.weather.features import (
    OBS_LAG,
    Context,
    build_day_profile,
    build_run_tables,
    climatology,
    forecast_features,
    obs_extra,
    obs_features,
    obs_local,
    run_bias,
    truth_table,
)
from pmrk.weather.model import WeatherModel

D = date(2026, 6, 10)
T = datetime(2026, 6, 10, 15, tzinfo=UTC)
DT = pl.Datetime("us", "UTC")
STATIONS = pl.DataFrame({"station_icao": ["KXXX"], "lat": [40.0], "lon": [-73.0], "tz": ["UTC"]})


def _q(t=T):
    return pl.DataFrame(
        {"qid": [1], "station_icao": ["KXXX"], "local_date": [D], "decision_time": [t]},
        schema_overrides={"qid": pl.Int64, "decision_time": DT},
    )


def _obs(times, temps, routine=None, station="KXXX", metar=None):
    n = len(times)
    return pl.DataFrame(
        {
            "station_icao": station,
            "local_date": D,
            "obs_time": times,
            "temp_c": temps,
            "temp_c_precise": [None] * n,
            "is_routine": routine or [True] * n,
            "metar": metar or [f"{station} 101200Z 18005KT FEW040 20/10"] * n,
        },
        schema_overrides={"obs_time": DT, "temp_c_precise": pl.Float64, "temp_c": pl.Float64},
    )


def _runs():
    inits = [datetime(2026, 6, 10, h, tzinfo=UTC) for h in (0, 6, 12)]
    avail = [i + timedelta(hours=8) for i in inits]  # the 12Z run is available at 20Z > T: must not be used
    daily = pl.DataFrame(
        {
            "station_icao": "KXXX",
            "model": "ecmwf_ifs025",
            "init_time": inits,
            "local_date": D,
            "fc_max": [25.0, 27.0, 40.0],
            "available_time": avail,
        }
    )
    steps = pl.DataFrame(
        {
            "station_icao": "KXXX",
            "model": "ecmwf_ifs025",
            "local_date": D,
            "init_time": [inits[1]] * 3 + [inits[2]] * 2,
            "valid_time": [
                T - timedelta(hours=3),
                T + timedelta(hours=3),
                T + timedelta(hours=6),
                T + timedelta(hours=3),
                T + timedelta(hours=6),
            ],
            "temp_c": [30.0, 26.0, 24.0, 40.0, 39.0],
        }
    )
    bias = pl.DataFrame(
        {"station_icao": ["KXXX"], "model": ["ecmwf_ifs025"], "local_date": [D], "bias": [0.0], "mse": [1.0]}
    )
    return daily, steps, bias


def test_latest_available_run_only():
    out = forecast_features(_q(), *_runs())
    assert out["fc_max_ecmwf_ifs025"][0] == 27.0
    assert out["init_time_ecmwf_ifs025"][0] == datetime(2026, 6, 10, 6, tzinfo=UTC)
    assert out["age_h_ecmwf_ifs025"][0] == 9.0


def test_remaining_forecast_uses_only_steps_after_t():
    assert forecast_features(_q(), *_runs())["rem_corr_ecmwf_ifs025"][0] == 26.0


def test_obs_respects_dissemination_lag():
    obs = _obs(
        [T - OBS_LAG - timedelta(minutes=30), T - OBS_LAG, T - OBS_LAG + timedelta(minutes=1)], [20.0, 22.0, 35.0]
    )
    out = obs_features(_q(), obs)
    assert out["obs_max"][0] == 22.0 and out["obs_n"][0] == 2
    assert obs_features(_q(), obs, lag=timedelta(minutes=30))["obs_n"][0] == 1


def test_obs_from_other_local_day_ignored():
    obs = _obs([T - timedelta(hours=20)], [40.0]).with_columns(local_date=pl.lit(D - timedelta(days=1)))
    assert obs_features(_q(), obs)["obs_n"][0] == 0


def test_routine_only_station_ignores_speci():
    obs = _obs([T - timedelta(hours=2), T - timedelta(hours=1)], [21.0, 23.0], routine=[True, False], station="UUWW")
    assert obs_features(_q().with_columns(station_icao=pl.lit("UUWW")), obs)["obs_max"][0] == 21.0


def test_obs_extras_respect_lag():
    obs = _obs(
        [T - timedelta(hours=3, minutes=30), T - timedelta(minutes=30), T - timedelta(minutes=5)],
        [20.0, 23.0, 30.0],
        metar=[
            "KXXX 101130Z 18005KT 10SM FEW040 20/10 A3000",
            "KXXX 101430Z 18010KT 10SM BKN030 23/10 A3000",
            "KXXX 101455Z 18030KT 10SM OVC010CB 30/10 A3000",
        ],
    )
    out = obs_extra(_q().with_columns(obs_last=pl.lit(23.0)), obs)
    assert out["cloud"][0] == 0.75 and out["wind_kt"][0] == 10.0  # the 14:55 report is not usable yet
    assert out["trend_3h"][0] == 3.0


def test_climatology_cutoff():
    prof = pl.DataFrame(
        {
            "station_icao": "KXXX",
            "local_date": [D - timedelta(days=10), D],
            "h": [0, 0],
            "rise": [2.0, 50.0],
            "peak_hour": [14, 3],
            "month": [6, 6],
        },
        schema_overrides={"h": pl.Int8, "month": pl.Int8},
    )
    peak, rise = climatology(prof, D)
    assert rise["clim_rise"][0] == 2.0 and peak["clim_peak_hour"][0] == 14.0


def test_bias_uses_only_days_at_least_gap_before():
    days = [D - timedelta(days=k) for k in range(40)]
    inits = [datetime(d.year, d.month, d.day, tzinfo=UTC) - timedelta(days=1) for d in days]
    daily = pl.DataFrame(
        {"station_icao": "KXXX", "model": "gfs_050", "init_time": inits, "local_date": days, "fc_max": 20.0}
    )
    truth = pl.DataFrame(
        {"station_icao": "KXXX", "local_date": days, "y_c": [30.0 if k < 4 else 20.0 for k in range(40)]}
    )  # +10 error on the last 4 days
    out = run_bias(daily, truth).filter(pl.col("local_date") == D)
    assert out["bias"][0] == 0.0


# ------------------------------------------------------------------------------------ WeatherModel end to end


class AllBins(dict):
    """Identity EMOS for every horizon bin: mean = ensemble (pre-day) or remaining forecast (intraday), sigma = 1."""

    def get(self, name, default=None):
        if name is None:
            return default
        if name.startswith("intra"):
            return FittedBin(name, np.array([1.0] + [0.0] * 11 + [0.0] * 5), (), 0)
        return FittedBin(name, np.array([0.0, 1.0, 0.0] + [0.0] * 4), (), 0)


def _context() -> Context:
    t0 = datetime(2026, 6, 10, tzinfo=UTC)
    obs_times = [t0 + timedelta(hours=h) for h in range(0, 16)]
    temps = [18.0 + 0.4 * h for h in range(16)]  # reaches 24.0 at 15Z
    raw = _obs(obs_times, temps).drop("local_date")
    obs = obs_local(raw, STATIONS)
    daily_max = pl.DataFrame({"station_icao": ["KXXX"], "local_date": [D], "n_obs": [16], "max_c_precise": [24.0]})
    init = datetime(2026, 6, 9, 0, tzinfo=UTC)
    runs = pl.DataFrame(
        {
            "station_icao": "KXXX",
            "model": "ecmwf_ifs025",
            "init_time": init,
            "valid_time": [init + timedelta(hours=s) for s in range(0, 91, 3)],
            "temp_c": 21.0,
            "available_time": init + timedelta(hours=8),
        },
        schema_overrides={"valid_time": DT},
    )
    truth = truth_table(daily_max, raw)
    steps, rdaily = build_run_tables(runs, STATIONS)
    return Context(STATIONS, obs, truth, rdaily, steps, run_bias(rdaily, truth), build_day_profile(obs))


def test_weather_model_through_core_backtest(data_dir):
    open_ts = datetime(2026, 6, 8, 12, tzinfo=UTC)
    end = datetime(2026, 6, 11, 12, tzinfo=UTC)
    labels = ["19°C or below", "20-21°C", "22°C or higher"]
    ms = [
        market(f"b{i}", title=lab, created=open_ts, end=end, prices=("1", "0") if i == 2 else ("0", "1"))
        for i, lab in enumerate(labels)
    ]
    save_markets(markets_frame([event("w1", ms, neg_risk=True, tags=("weather",))]))
    grid = [open_ts + timedelta(minutes=10 * k) for k in range(6 * 72)]
    write_prices(data_dir, "w1", [(f"b{i}", t, p) for t in grid for i, p in enumerate((0.2, 0.5, 0.35))])
    write_trades(data_dir, "w1", [(f"b{i}", t, "BUY", 0, 0.3, 50.0) for t in grid[::3] for i in range(3)])
    rules = pl.DataFrame(
        {
            "market_id": ["b0", "b1", "b2"],
            "city": "x",
            "station_icao": "KXXX",
            "unit": "C",
            "local_date": D,
            "bucket_low": [None, 20, 22],
            "bucket_high": [19, 21, None],
        },
        schema_overrides={"bucket_low": pl.Int32, "bucket_high": pl.Int32},
    )
    model = WeatherModel(
        ctx=_context(),
        fits={
            "months": {
                date(2026, 6, 1): {
                    "models": AllBins(),
                    "peak": pl.DataFrame(
                        schema={"station_icao": pl.Utf8, "month": pl.Int8, "clim_peak_hour": pl.Float64}
                    ),
                    "rise": pl.DataFrame(
                        schema={"station_icao": pl.Utf8, "month": pl.Int8, "h": pl.Int8, "clim_rise": pl.Float64}
                    ),
                }
            },
            "raw_sigma": pl.DataFrame({"hbin": ["x"], "raw_sigma": [1.0]}),
        },
        rules=rules,
    )
    panel = build_panel(load_markets(), model)
    assert panel.height > 0 and {"grid", "metar", "model_run"} <= set(panel["trigger"])
    metar_times = panel.filter(pl.col("trigger") == "metar")["decision_time"].unique().sort()
    assert metar_times[0] == datetime(2026, 6, 10, 0, 10, tzinfo=UTC)  # first METAR + 10 min lag
    sums = panel.group_by("decision_time").agg(pl.col("p_model").sum())["p_model"]
    assert sums.to_list() == pytest.approx([1.0] * sums.len())
    # once a 22.0+ observation is usable, the top bucket is certain (hard lower bound)
    late = panel.filter(
        (pl.col("market_id") == "b2") & (pl.col("decision_time") >= datetime(2026, 6, 10, 10, 10, tzinfo=UTC))
    )
    assert late.height and (late["p_model"] > 0.99).all()
