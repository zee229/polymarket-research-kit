"""On-disk layout under the data dir and resumable bulk downloads.

    <data>/markets.parquet            normalized market table (merged across fetches)
    <data>/prices/<event_id>.parquet  outcome-0 price history of every market in the event
    <data>/trades/<event_id>.parquet  taker trades of every market in the event

An existing per-event file means "done", so an interrupted download resumes where it stopped. Only outcome-0
prices are stored: for a binary market the outcome-1 price is 1 - p.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import polars as pl

from pmrk.config import data_dir
from pmrk.polymarket.clob import price_history
from pmrk.polymarket.data_api import TRADE_SCHEMA, outcome0_view, taker_trades
from pmrk.polymarket.markets import with_derived

log = logging.getLogger(__name__)


def markets_path() -> Path:
    return data_dir() / "markets.parquet"


def prices_dir() -> Path:
    return data_dir() / "prices"


def trades_dir() -> Path:
    return data_dir() / "trades"


def load_markets(path: Path | None = None) -> pl.DataFrame:
    path = path or markets_path()
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `pmrk fetch markets` first")
    return pl.read_parquet(path)


def save_markets(new: pl.DataFrame, path: Path | None = None) -> pl.DataFrame:
    """Merge into the stored table (newer rows win) and recompute per-event columns."""
    path = path or markets_path()
    if path.exists():
        old = pl.read_parquet(path).select(new.columns)
        new = pl.concat([old, new], how="vertical_relaxed").unique("market_id", keep="last", maintain_order=True)
    out = with_derived(new)
    _write(path, out)
    return out


def _write(path: Path, df: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


def _event_prices(event: pl.DataFrame, fidelity_min: int, lookback: timedelta | None) -> pl.DataFrame:
    now = datetime.now(UTC)
    frames = []
    for m in event.iter_rows(named=True):
        first = m["start_ts"] or m["created_ts"]
        if first is None or not m["token0"]:
            continue
        last = max(t for t in (m["end_ts"], m["closed_ts"], m["uma_end_ts"], first) if t is not None)
        start = first - timedelta(hours=1)
        if lookback is not None:
            start = max(start, (m["known_ts"] or last) - lookback)
        end = min(last + timedelta(days=1), now)
        px = price_history(m["token0"], start, end, fidelity_min)
        frames.append(px.with_columns(market_id=pl.lit(m["market_id"])))
    if not frames:
        return pl.DataFrame(schema={"ts": pl.Datetime("us", "UTC"), "p": pl.Float64, "market_id": pl.Utf8})
    return pl.concat(frames)


def _event_trades(event: pl.DataFrame) -> pl.DataFrame:
    frames = []
    for m in event.iter_rows(named=True):
        if not m["condition_id"]:
            continue
        tr, truncated = taker_trades(m["condition_id"])
        frames.append(tr.with_columns(market_id=pl.lit(m["market_id"]), truncated=pl.lit(truncated)))
    if not frames:
        return pl.DataFrame(schema={**TRADE_SCHEMA, "market_id": pl.Utf8, "truncated": pl.Boolean})
    return pl.concat(frames)


def _download(
    markets: pl.DataFrame, out_dir: Path, fetch: Callable[[pl.DataFrame], pl.DataFrame], workers: int
) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = {k[0]: g for k, g in markets.group_by("event_id")}
    todo = [e for e in groups if not (out_dir / f"{e}.parquet").exists()]
    log.info("%s: %d events, %d to fetch", out_dir.name, len(groups), len(todo))
    ok = failed = 0
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(fetch, groups[e]): e for e in todo}
        for fut in as_completed(futures):
            event_id = futures[fut]
            try:
                _write(out_dir / f"{event_id}.parquet", fut.result())
                ok += 1
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                failed += 1
                log.warning("event %s failed (rerun resumes): %r", event_id, exc)
            if (ok + failed) % 100 == 0:
                log.info("%s: %d/%d done, %d failed", out_dir.name, ok + failed, len(todo), failed)
    return {"events": len(groups), "fetched": ok, "failed": failed, "skipped": len(groups) - len(todo)}


def download_prices(
    markets: pl.DataFrame, fidelity_min: int = 1, lookback: timedelta | None = None, workers: int = 8
) -> dict[str, int]:
    """Outcome-0 price history per event; `lookback` limits history to the last N before `known_ts`."""
    return _download(markets, prices_dir(), lambda ev: _event_prices(ev, fidelity_min, lookback), workers)


def download_trades(markets: pl.DataFrame, workers: int = 8) -> dict[str, int]:
    return _download(markets, trades_dir(), _event_trades, workers)


def _load(out_dir: Path, event_ids: Iterable[str]) -> list[pl.DataFrame]:
    files = [out_dir / f"{e}.parquet" for e in event_ids]
    return [pl.read_parquet(f).with_columns(event_id=pl.lit(f.stem)) for f in files if f.exists()]


def load_prices(event_ids: Iterable[str]) -> pl.DataFrame:
    """Stored prices of the given events: event_id, market_id, ts, p (sorted)."""
    frames = [f for f in _load(prices_dir(), event_ids) if f.height]
    if not frames:
        return pl.DataFrame(
            schema={"ts": pl.Datetime("us", "UTC"), "p": pl.Float64, "market_id": pl.Utf8, "event_id": pl.Utf8}
        )
    return pl.concat(frames, how="diagonal_relaxed").sort("market_id", "ts")


def load_trades(event_ids: Iterable[str]) -> pl.DataFrame:
    """Stored taker trades of the given events in outcome-0 terms (see `data_api.outcome0_view`)."""
    frames = [f for f in _load(trades_dir(), event_ids) if f.height]
    if not frames:
        empty = pl.DataFrame(
            schema={**TRADE_SCHEMA, "market_id": pl.Utf8, "truncated": pl.Boolean, "event_id": pl.Utf8}
        )
        return outcome0_view(empty)
    return outcome0_view(pl.concat(frames, how="diagonal_relaxed").sort("market_id", "ts"))


def stored_event_ids(kind: str) -> set[str]:
    """Event ids with a stored `prices` or `trades` file."""
    d = prices_dir() if kind == "prices" else trades_dir()
    return {f.stem for f in d.glob("*.parquet")} if d.exists() else set()
