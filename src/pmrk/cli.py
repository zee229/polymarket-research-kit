"""`pmrk` command line. Every command works on any set of Polymarket markets stored under the data dir."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date
from pathlib import Path

import polars as pl

from pmrk.config import ENV_DATA_DIR, data_dir
from pmrk.snapshots.horizons import DEFAULT_HORIZONS, LIFE_FRACTIONS, parse_duration

log = logging.getLogger("pmrk")


def _date(text: str) -> date:
    return date.fromisoformat(text)


def _reports() -> Path:
    return data_dir() / "reports"


def _print(df: pl.DataFrame, rows: int = 40) -> None:
    with pl.Config(tbl_rows=rows, tbl_width_chars=200, tbl_cols=16, float_precision=4):
        print(df)


def _select(markets: pl.DataFrame, args: argparse.Namespace) -> pl.DataFrame:
    """Apply the common --event / --category / --clean-only filters to the stored market table."""
    if getattr(args, "event", None):
        keys = set(args.event)
        markets = markets.filter(pl.col("event_id").is_in(keys) | pl.col("event_slug").is_in(keys))
    if getattr(args, "category", None):
        markets = markets.filter(pl.col("category").is_in(args.category))
    if getattr(args, "clean_only", False):
        markets = markets.filter(pl.col("clean"))
    return markets


# ------------------------------------------------------------------------------------------------ fetch


def cmd_fetch(args: argparse.Namespace) -> None:
    from pmrk.polymarket import gamma, store
    from pmrk.polymarket.markets import markets_frame

    if args.what == "markets":
        if args.event:
            events = [gamma.get_event(k) for k in args.event]
        elif args.start and args.end:
            closed = None if args.include_open else True
            events = list(gamma.iter_events(args.start, args.end, tag_slug=args.tag, closed=closed))
        else:
            raise SystemExit("fetch markets needs --event, or --start and --end (optionally --tag)")
        df = store.save_markets(markets_frame(events))
        print(f"{len(events)} events fetched; {df.height} markets stored in {store.markets_path()}")
        _print(
            df.group_by("category", "structure")
            .agg(markets=pl.len(), events=pl.col("event_id").n_unique(), clean=pl.col("clean").sum())
            .sort("markets", descending=True)
        )
        return
    markets = _select(store.load_markets(), args)
    if args.what == "prices":
        lookback = parse_duration(args.lookback) if args.lookback else None
        print(store.download_prices(markets, args.fidelity, lookback, args.workers))
    else:
        print(store.download_trades(markets, args.workers))


# ------------------------------------------------------------------------------------------------ snapshot


def cmd_snapshot(args: argparse.Namespace) -> None:
    from pmrk.polymarket import store
    from pmrk.scan.discovery import sample_markets
    from pmrk.snapshots.horizons import snapshot_targets
    from pmrk.snapshots.prices import fetch_snapshots, price_at

    markets = _select(store.load_markets(), args).filter(pl.col("clean"))
    if args.sample:
        markets = sample_markets(markets, args.sample, args.per_event, args.seed)
    horizons = {h: parse_duration(h) for h in args.horizons.split(",")} if args.horizons else DEFAULT_HORIZONS
    targets = snapshot_targets(markets, horizons, LIFE_FRACTIONS if args.life else {}, parse_duration(args.buffer))
    stale = parse_duration(args.max_stale)
    if args.from_stored:
        ids = markets["event_id"].unique().to_list()
        snaps = price_at(targets, store.load_prices(ids), "target_ts", stale).drop_nulls("p")
    else:
        snaps = fetch_snapshots(markets, targets, args.fidelity, stale, args.workers)
    out = data_dir() / "snapshots.parquet"
    snaps.write_parquet(out)
    print(f"{targets.height} guarded targets, {snaps.height} snapshots with a price -> {out}")
    _print(snaps.group_by("snapshot").agg(n=pl.len()).sort("snapshot"))


# ------------------------------------------------------------------------------------------------ costs


def cmd_costs(args: argparse.Namespace) -> None:
    from pmrk.execution.costs import cost_table, trade_costs
    from pmrk.polymarket import store

    markets = _select(store.load_markets(), args)
    ids = sorted(
        store.stored_event_ids("prices") & store.stored_event_ids("trades") & set(markets["event_id"].to_list())
    )
    if not ids:
        raise SystemExit("no events with both stored prices and trades: run `pmrk fetch prices` and `fetch trades`")
    costs = trade_costs(store.load_trades(ids), store.load_prices(ids), parse_duration(args.tolerance))
    by = list(args.by or [])
    if by:
        costs = costs.join(markets.select("market_id", *by), on="market_id")
    table = cost_table(costs, by)
    out = data_dir() / "costs.parquet"
    table.write_parquet(out)
    print(f"{costs.height} trades in {len(ids)} events -> {out}")
    _print(table, 80)


# ------------------------------------------------------------------------------------------------ calibration


def cmd_calibration(args: argparse.Namespace) -> None:
    from pmrk.polymarket import store
    from pmrk.scan import discovery, holdout

    out_dir = _reports() / "calibration"
    snaps_path = data_dir() / "snapshots.parquet"
    if not snaps_path.exists():
        raise SystemExit("no snapshots.parquet: run `pmrk snapshot` first")
    markets = store.load_markets()
    assembled = discovery.assemble(pl.read_parquet(snaps_path), markets, args.split)
    costs_path = data_dir() / "costs.parquet"
    costs = pl.read_parquet(costs_path) if costs_path.exists() else None
    if costs is None:
        log.warning(
            "no costs.parquet: assuming %.2f per share; run `pmrk costs` for measured costs", discovery.DEFAULT_COST
        )
    if args.step == "discover":
        cells, cand = discovery.discover(assembled, costs, args.min_events, args.q)
        out_dir.mkdir(parents=True, exist_ok=True)
        cells.write_csv(out_dir / "discovery_cells.csv")
        discovery.summary_by_band(assembled).write_csv(out_dir / "discovery_by_band.csv")
        digest = discovery.freeze(cand, cells, out_dir, args.split)
        print(
            f"{cells.height} cells tested, {int(cells['bh_pass'].sum())} BH-significant, {cand.height} frozen "
            f"(sha256[:16] {digest}) -> {out_dir}"
        )
        _print(cand.select("category", "band", "snapshot", "side", "events", "diff", "pval", "exp_roi").head(20))
        return
    frozen = discovery.load_frozen(out_dir)
    hold = assembled.filter(pl.col("split") == "holdout")
    if args.fetch_trades:
        need = hold.select("event_id").unique()
        missing = need.filter(~pl.col("event_id").is_in(list(store.stored_event_ids("trades"))))
        store.download_trades(markets.join(missing, on="event_id"), args.workers)
    trades = store.load_trades(hold["event_id"].unique().to_list())
    res = holdout.evaluate(hold, frozen, trades, costs, args.max_events)
    res.write_parquet(out_dir / "holdout_results.parquet")
    flat = res.select([c for c in res.columns if res[c].dtype != pl.List(pl.Float64)])
    flat.write_csv(out_dir / "holdout_results.csv")
    cols = [
        c
        for c in (
            "cand",
            "category",
            "band",
            "snapshot",
            "side",
            "mid_roi",
            "print_roi",
            "print_lim_roi",
            "print_lim_events",
            "fill_rate",
            "fillable_usd_per_day",
            "pass",
        )
        if c in res.columns
    ]
    _print(res.select(cols).sort("print_lim_roi", descending=True, nulls_last=True), 30)
    print(f"passing all criteria: {int(res['pass'].sum())} of {res.height}")


# ------------------------------------------------------------------------------------------------ backtest


def cmd_backtest(args: argparse.Namespace) -> None:
    from pmrk.backtest import run as bt
    from pmrk.backtest.engine import SimConfig
    from pmrk.interfaces import load_model
    from pmrk.polymarket import store

    model = load_model(args.model)
    markets = _select(store.load_markets(), args)
    out_dir = _reports() / "backtest" / model.name
    panel = bt.build_panel(
        markets,
        model,
        args.batch_events,
        out_dir / "panel.parquet",
        every=parse_duration(args.every),
        buffer=parse_duration(args.buffer),
    )
    if panel.is_empty():
        raise SystemExit("empty panel: check that prices/trades are stored for these markets and the model predicts")
    if args.band_cost:
        from pmrk.execution.costs import attach_cost

        table = pl.read_parquet(data_dir() / "costs.parquet")
        by = ["category"] if "category" in table.columns else None
        panel = attach_cost(panel, table, stat=f"{args.band_cost}_cost", by=by)
    base = SimConfig(cost=args.cost, threshold=args.threshold, sizing=args.sizing)
    res, trades = bt.run(panel, base, args.split)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(res, indent=1, default=str))
    trades.write_parquet(out_dir / "trades.parquet")
    main = res["test_main"]
    print(
        f"model {model.name}: split {res['split_date']}, threshold {res['chosen_threshold']} (tuned), "
        f"test events {res['test_events']}"
    )
    for k, v in res.items():
        if k.startswith("test_") and isinstance(v, dict):
            ci = v.get("ci", {}).get("roi")
            ci_txt = f"({ci[0]:+.3f}, {ci[1]:+.3f})" if ci else ""
            print(f"  {k:32s} trades={v['n_trades']:>6} roi={v['roi']:+.3f} {ci_txt}")
    print(f"main: pnl ${main['pnl']:.0f}, turnover ${main['turnover']:.0f} -> {out_dir}")
    if args.plot:
        bt.plot_heatmap(res["test_grid"], out_dir / "heatmap_roi.png")


# ------------------------------------------------------------------------------------------------ parser


def _market_filters(p: argparse.ArgumentParser) -> None:
    p.add_argument("--event", nargs="+", help="event ids or slugs")
    p.add_argument("--category", nargs="+", help="categories (see polymarket/categories.py)")
    p.add_argument("--clean-only", action="store_true", help="only cleanly resolved markets")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="pmrk", description=__doc__)
    ap.add_argument("--data-dir", help=f"data directory (default ${ENV_DATA_DIR} or ./data)")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download markets, prices or trades")
    f.add_argument("what", choices=["markets", "prices", "trades"])
    _market_filters(f)
    f.add_argument("--tag", help="Gamma tag slug (with --start/--end)")
    f.add_argument("--start", type=_date, help="events ending on or after (YYYY-MM-DD)")
    f.add_argument("--end", type=_date, help="events ending before (YYYY-MM-DD)")
    f.add_argument("--include-open", action="store_true", help="also fetch events that are not closed")
    f.add_argument("--fidelity", type=int, default=1, help="price resolution in minutes")
    f.add_argument("--lookback", help="only price history this long before resolution, e.g. 14d")
    f.add_argument("--workers", type=int, default=8)
    f.set_defaults(func=cmd_fetch)

    s = sub.add_parser("snapshot", help="prices at fixed horizons before resolution (lookahead-guarded)")
    _market_filters(s)
    s.add_argument("--horizons", help=f"comma list, default {','.join(DEFAULT_HORIZONS)}")
    s.add_argument("--no-life", dest="life", action="store_false", help="skip the 10/50/90%% lifetime snapshots")
    s.add_argument("--buffer", default="1h", help="no snapshot within this of the earliest end/close/resolution")
    s.add_argument("--max-stale", default="60m")
    s.add_argument("--fidelity", type=int, default=10)
    s.add_argument("--from-stored", action="store_true", help="use stored price history instead of fetching windows")
    s.add_argument("--sample", type=int, help="random events per category")
    s.add_argument("--per-event", type=int, default=3, help="markets per sampled event (0 = all)")
    s.add_argument("--seed", type=int, default=20260928)
    s.add_argument("--workers", type=int, default=16)
    s.set_defaults(func=cmd_snapshot)

    c = sub.add_parser("costs", help="measured taker cost vs mid by price band")
    _market_filters(c)
    c.add_argument("--by", nargs="+", help="extra keys, e.g. category")
    c.add_argument("--tolerance", default="5m", help="max age of the mid a trade is compared to")
    c.set_defaults(func=cmd_costs)

    k = sub.add_parser("calibration", help="discovery screen with frozen candidates, then holdout test")
    k.add_argument("step", choices=["discover", "holdout"])
    k.add_argument("--split", type=_date, required=True, help="events ending before this date are discovery")
    k.add_argument("--min-events", type=int, default=100)
    k.add_argument("--q", type=float, default=0.05, help="Benjamini-Hochberg FDR")
    k.add_argument("--max-events", type=int, default=400, help="holdout events per candidate")
    k.add_argument("--fetch-trades", action="store_true", help="download missing trades for holdout events")
    k.add_argument("--workers", type=int, default=8)
    k.set_defaults(func=cmd_calibration)

    b = sub.add_parser("backtest", help="walk a ProbabilityModel through stored markets")
    _market_filters(b)
    b.add_argument("--model", required=True, help="'package.module:Name' or 'path/file.py:Name'")
    b.add_argument("--every", default="1h", help="decision grid step")
    b.add_argument("--buffer", default="1h", help="no decision within this of the earliest end/close/resolution")
    b.add_argument("--cost", type=float, default=0.03, help="flat cost above mid per share")
    b.add_argument("--threshold", type=float, default=0.05, help="only used if tuning finds nothing")
    b.add_argument("--sizing", choices=["flat", "kelly"], default="flat")
    b.add_argument("--band-cost", choices=["median", "mean", "p75"], help="measured cost by band from `pmrk costs`")
    b.add_argument("--split", type=_date, help="tune/test split date (default: median event date)")
    b.add_argument("--batch-events", type=int, default=500)
    b.add_argument("--plot", action="store_true", help="write a threshold x cost heatmap (needs [plots])")
    b.set_defaults(func=cmd_backtest)

    w = sub.add_parser("weather", help="weather case study (needs the [weather] extra)", add_help=False)
    w.add_argument("rest", nargs=argparse.REMAINDER)
    w.set_defaults(func=cmd_weather)
    return ap


def cmd_weather(args: argparse.Namespace) -> None:
    try:
        from pmrk.weather.cli import main as weather_main
    except ImportError as exc:
        raise SystemExit(f"weather commands need the extra: uv sync --extra weather ({exc})") from exc
    weather_main(args.rest)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.data_dir:
        os.environ[ENV_DATA_DIR] = args.data_dir
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.func(args)


if __name__ == "__main__":
    main()
