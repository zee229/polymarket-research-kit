"""`pmrk weather ...`: reproduce the daily max-temperature case study end to end.

pmrk weather markets       crawl every <city>-daily-weather series (core + weather rule tables)
pmrk weather stations      station coordinates and timezones
pmrk weather prices        1-minute price history of settleable events (core store)
pmrk weather trades        taker trades of settleable events (core store)
pmrk weather metar         IEM METAR/SPECI archive, decoded
pmrk weather settlement    reproduce resolutions from METARs (SettlementSource)
pmrk weather forecasts     exact ECMWF/GFS runs from AWS (needs ecCodes)
pmrk weather fit           walk-forward EMOS fits
pmrk weather eval          model vs market log loss by horizon
pmrk weather backtest      taker backtest of the WeatherModel through the core engine
"""

from __future__ import annotations

import argparse
import json
from datetime import date

import polars as pl

from pmrk.config import data_dir
from pmrk.snapshots.horizons import parse_duration


def _print(df: pl.DataFrame) -> None:
    with pl.Config(tbl_rows=80, tbl_width_chars=200, tbl_cols=14, float_precision=4):
        print(df)


def _settleable_markets() -> pl.DataFrame:
    from pmrk.weather.markets import load, settleable

    return load().filter(settleable())


def _validation() -> pl.DataFrame:
    from pmrk.weather import settlement
    from pmrk.weather.markets import load

    return settlement.validate(load(), pl.read_parquet(settlement.daily_path()))


def _split(args: argparse.Namespace) -> date:
    if args.split:
        return args.split
    ev = _settleable_markets().filter(pl.col("clean")).select("event_id", "local_date").unique().sort("local_date")
    return ev["local_date"][ev.height // 2]


def _model(args: argparse.Namespace, split: date, triggers: bool):  # noqa: ANN202 - WeatherModel
    from pmrk.weather.model import WeatherModel
    from pmrk.weather.settlement import reliable_stations

    ok = reliable_stations(_validation(), split)
    return WeatherModel(obs_lag=parse_duration(args.obs_lag), stations=ok, triggers=triggers)


def cmd(args: argparse.Namespace) -> None:  # noqa: C901 - flat dispatch
    from pmrk.polymarket import store
    from pmrk.weather import forecasts, markets, metar, settlement, stations

    if args.cmd == "markets":
        wx = markets.fetch_all()
        _print(wx.group_by("source", "unit").agg(markets=pl.len(), events=pl.col("event_id").n_unique()))
    elif args.cmd == "stations":
        icaos = _settleable_markets()["station_icao"].unique().to_list()
        _print(stations.build_stations(icaos))
    elif args.cmd in ("prices", "trades"):
        m = _settleable_markets().filter(pl.col("closed"))
        fn = store.download_prices if args.cmd == "prices" else store.download_trades
        print(fn(m, workers=args.workers))
    elif args.cmd == "metar":
        metar.download(_settleable_markets()["station_icao"].unique().sort().to_list(), args.first_year)
        print(metar.build_obs().group_by("station_icao").agg(pl.len(), pl.col("is_routine").mean()))
    elif args.cmd == "settlement":
        daily = settlement.build_daily(pl.read_parquet(metar.obs_path()), stations.load_stations())
        daily.write_parquet(settlement.daily_path())
        v = _validation().filter(pl.col("has_obs"))
        clean = v.filter(~pl.col("incident"))
        _print(clean.group_by("unit", "source").agg(events=pl.len(), reproduced=pl.col("ok").mean()).sort("unit"))
        print(f"all clean events: {clean.height}, reproduced {clean['ok'].mean():.4f}")
        _print(clean.group_by("station_icao").agg(n=pl.len(), rate=pl.col("ok").mean()).sort("rate").head(10))
    elif args.cmd == "forecasts":
        forecasts.download(stations.load_stations(), args.models, workers=args.workers)
        print(forecasts.build_runs().group_by("model").agg(runs=pl.col("init_time").n_unique()))
    elif args.cmd == "fit":
        from pmrk.weather.model import fit_walk_forward, load_context

        fits = fit_walk_forward(load_context())
        print(f"{len(fits['months'])} monthly fits -> {data_dir() / 'weather'}")
    elif args.cmd == "eval":
        _eval(args)
    elif args.cmd == "backtest":
        _backtest(args)


def _eval(args: argparse.Namespace) -> None:
    from pmrk.backtest.run import build_panel
    from pmrk.stats.reliability import head_to_head, outcome_losses

    split = _split(args)
    model = _model(args, split, triggers=False)
    panel = build_panel(_settleable_markets(), model, args.batch_events)
    out = []
    for col in ("p_model", "p_clim", "p_raw"):
        norm = pl.col(col).clip(1e-4, 1.0) / pl.col(col).clip(1e-4, 1.0).sum().over("event_id", "decision_time")
        losses = outcome_losses(panel.with_columns(norm.alias(col)), col).with_columns(block=pl.col("day"))
        out.append(head_to_head(losses, "hgroup").with_columns(model=pl.lit(col)))
    res = pl.concat(out)
    path = data_dir() / "reports" / "weather" / "logloss_by_horizon.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    res.write_csv(path)
    _print(res.sort("model", "group"))


def _backtest(args: argparse.Namespace) -> None:
    from pmrk.backtest import run as bt
    from pmrk.backtest.engine import SimConfig
    from pmrk.execution.costs import attach_cost

    split = _split(args)
    model = _model(args, split, triggers=True)
    panel = bt.build_panel(_settleable_markets(), model, args.batch_events)
    if args.band_cost:
        table = pl.read_parquet(data_dir() / "costs.parquet")
        by = ["category"] if "category" in table.columns else None
        panel = attach_cost(panel, table, stat=f"{args.band_cost}_cost", by=by)
    res, trades = bt.run(panel, SimConfig(cost=args.cost), split, breakdowns=("trigger", "hgroup", "city", "side"))
    suffix = f"_{args.band_cost}" if args.band_cost else ""
    out = data_dir() / "reports" / "weather" / f"backtest_lag{args.obs_lag}{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=str))
    trades.write_parquet(out / "trades.parquet")
    for k, v in res.items():
        if k.startswith("test_") and isinstance(v, dict):
            print(f"{k:28s} trades={v['n_trades']:>6} roi={v['roi']:+.3f} ci={v.get('ci', {}).get('roi')}")
    print(f"threshold {res['chosen_threshold']} (tuned before {res['split_date']}) -> {out}")


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(
        prog="pmrk weather", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "cmd",
        choices=[
            "markets",
            "stations",
            "prices",
            "trades",
            "metar",
            "settlement",
            "forecasts",
            "fit",
            "eval",
            "backtest",
        ],
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--first-year", type=int, default=2023, help="metar: first archive year")
    ap.add_argument("--models", nargs="+", default=["ecmwf_ifs025", "gfs_050"], help="forecasts: models")
    ap.add_argument("--split", type=date.fromisoformat, help="tune/test split (default: median event date)")
    ap.add_argument("--obs-lag", default="10m", help="METAR usable this long after observation time")
    ap.add_argument("--cost", type=float, default=0.03, help="flat cost above mid per share")
    ap.add_argument("--band-cost", choices=["median", "mean", "p75"], help="use measured cost by band (costs.parquet)")
    ap.add_argument("--batch-events", type=int, default=800)
    cmd(ap.parse_args(argv))
