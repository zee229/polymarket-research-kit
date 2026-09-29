"""Price at a time (last point at or before it, bounded staleness) and event-level normalization.

negRisk events are groups of mutually exclusive binaries whose mids usually sum to slightly more than 1
(overround). `q_market` divides by that sum so the event's outcome probabilities add up to 1; standalone binaries
keep `q_market = p`.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta

import httpx
import polars as pl

from pmrk.polymarket.clob import price_history

log = logging.getLogger(__name__)

MAX_STALE = timedelta(minutes=60)


def price_at(
    rows: pl.DataFrame, prices: pl.DataFrame, time_col: str = "decision_time", max_stale: timedelta = MAX_STALE
) -> pl.DataFrame:
    """Attach `p` / `price_ts`: last price of the market at or before `time_col`; null if older than max_stale."""
    px = prices.select("market_id", pl.col("ts").alias("price_ts"), "p").sort("price_ts")
    out = rows.sort(time_col).join_asof(
        px, left_on=time_col, right_on="price_ts", by="market_id", strategy="backward", check_sortedness=False
    )
    stale = pl.col(time_col) - pl.col("price_ts") > max_stale
    return out.with_columns(
        p=pl.when(stale).then(None).otherwise(pl.col("p")),
        price_ts=pl.when(stale).then(None).otherwise(pl.col("price_ts")),
    )


def normalize_event(rows: pl.DataFrame, time_col: str = "decision_time") -> pl.DataFrame:
    """Add `p_sum` and `q_market`; negRisk (event, time) groups with any missing price are dropped entirely."""
    neg = pl.col("neg_risk").fill_null(False)
    rows = rows.with_columns(_miss=pl.col("p").is_null().any().over("event_id", time_col))
    rows = rows.filter(pl.col("p").is_not_null() & ~(neg & pl.col("_miss"))).drop("_miss")
    p_sum = pl.col("p").sum().over("event_id", time_col)
    return rows.with_columns(p_sum=pl.when(neg).then(p_sum).otherwise(pl.col("p"))).with_columns(
        q_market=pl.when(neg & (pl.col("p_sum") > 0)).then(pl.col("p") / pl.col("p_sum")).otherwise(pl.col("p"))
    )


def overround(rows: pl.DataFrame, time_col: str = "target_ts") -> pl.DataFrame:
    """Sum of mids per complete negRisk (event, time): rows must cover every market of the event."""
    g = (
        rows.filter(pl.col("neg_risk"))
        .group_by("event_id", time_col)
        .agg(n=pl.len(), m=pl.col("markets_in_event").first(), sum_p=pl.col("p").sum())
    )
    return g.filter(pl.col("n") == pl.col("m")).drop("n", "m")


# ------------------------------------------------------------------ targeted fetch (no full history needed)

MAIN_WINDOW = timedelta(days=15)
SMALL_WINDOW = timedelta(hours=2)


def _market_snapshots(market: dict, targets: pl.DataFrame, fidelity_min: int, max_stale: timedelta) -> pl.DataFrame:
    """One window [end - 15d, min(end, closed)] covers recent targets; each older target gets a 2 h window."""
    main_end = min(t for t in (market["end_ts"], market["closed_ts"]) if t is not None)
    main_start = main_end - MAIN_WINDOW
    tg = targets["target_ts"].to_list()
    windows = [(main_start, main_end)] if any(t >= main_start for t in tg) else []
    windows += [(t - SMALL_WINDOW, t) for t in tg if t < main_start]
    frames = [price_history(market["token0"], a, b, fidelity_min, window=MAIN_WINDOW) for a, b in windows]
    px = pl.concat(frames).unique("ts").with_columns(market_id=pl.lit(market["market_id"]))
    return price_at(targets, px, "target_ts", max_stale).drop_nulls("p")


def fetch_snapshots(
    markets: pl.DataFrame,
    targets: pl.DataFrame,
    fidelity_min: int = 10,
    max_stale: timedelta = MAX_STALE,
    workers: int = 16,
) -> pl.DataFrame:
    """Prices at snapshot targets fetched directly (about 1.4 requests per market instead of full history)."""
    by_market = {k[0]: g for k, g in targets.group_by("market_id")}
    rows = markets.filter(pl.col("market_id").is_in(list(by_market)) & pl.col("token0").is_not_null())
    out, failed = [], 0
    with ThreadPoolExecutor(workers) as pool:
        futs = {
            pool.submit(_market_snapshots, m, by_market[m["market_id"]], fidelity_min, max_stale): m["market_id"]
            for m in rows.iter_rows(named=True)
        }
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                out.append(fut.result())
            except (httpx.HTTPError, ValueError) as exc:
                failed += 1
                log.warning("market %s failed: %r", futs[fut], exc)
            if i % 1000 == 0:
                log.info("snapshots: %d/%d markets, %d failed", i, len(futs), failed)
    if not out:
        return targets.clear().with_columns(price_ts=pl.lit(None, pl.Datetime("us", "UTC")), p=pl.lit(None, pl.Float64))
    return pl.concat(out)
