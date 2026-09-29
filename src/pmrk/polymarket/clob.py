"""CLOB `/prices-history`: the price series of one outcome token.

For closed markets, fine-grained history is only returned with explicit `startTs/endTs`; `interval=max` with a
small fidelity comes back empty. Windows longer than ~15 days are rejected, so requests are chunked (5 days by
default, which is what the weather study used at 1-minute fidelity). The meaning of `p` is not documented; it
behaves like a mid-price proxy, not an executable price.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from pmrk.polymarket.http import Client

CLOB = "https://clob.polymarket.com"
MAX_WINDOW = timedelta(days=5)
PRICE_SCHEMA = {"ts": pl.Datetime("us", "UTC"), "p": pl.Float64}

_client = Client("clob", min_interval_s=0.03)


def price_history(
    token_id: str, start: datetime, end: datetime, fidelity_min: int = 1, window: timedelta = MAX_WINDOW
) -> pl.DataFrame:
    """Points (ts, p) of `token_id` in [start, end], fetched in windows of at most `window`."""
    frames, t0 = [], start
    while t0 < end:
        t1 = min(t0 + window, end)
        params = {
            "market": token_id,
            "startTs": int(t0.timestamp()),
            "endTs": int(t1.timestamp()),
            "fidelity": fidelity_min,
        }
        hist = _client.get_json(f"{CLOB}/prices-history", params).get("history") or []
        if hist:
            frames.append(
                pl.DataFrame(hist).select(
                    ts=pl.from_epoch(pl.col("t").cast(pl.Int64), time_unit="s").dt.replace_time_zone("UTC"),
                    p=pl.col("p").cast(pl.Float64),
                )
            )
        t0 = t1
    if not frames:
        return pl.DataFrame(schema=PRICE_SCHEMA)
    return pl.concat(frames).unique("ts").sort("ts")
