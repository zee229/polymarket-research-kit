"""Data API `/trades`: executed taker trades of a market.

With `takerOnly=true` the summed `size` equals Gamma `volumeNum` (verified on weather markets), so these are the
market's actual prints. The API stops paginating at an offset of ~10k; very active markets are truncated and
flagged.
"""

from __future__ import annotations

import polars as pl

from pmrk.polymarket.http import Client

DATA_API = "https://data-api.polymarket.com"
PAGE = 500
MAX_OFFSET = 10_000
TRADE_SCHEMA = {
    "ts": pl.Datetime("us", "UTC"),
    "side": pl.Utf8,
    "outcome_index": pl.Int8,
    "price": pl.Float64,
    "size": pl.Float64,
    "tx": pl.Utf8,
}

_client = Client("data_api", min_interval_s=0.05)


def taker_trades(condition_id: str, max_offset: int = MAX_OFFSET) -> tuple[pl.DataFrame, bool]:
    """All taker trades of a market and whether the offset ceiling truncated them."""
    rows, off, truncated = [], 0, False
    while True:
        params = {"market": condition_id, "limit": PAGE, "offset": off, "takerOnly": "true"}
        page = _client.get_json(f"{DATA_API}/trades", params)
        rows += [
            (
                int(t["timestamp"]),
                t["side"],
                int(t["outcomeIndex"]),
                float(t["price"]),
                float(t["size"]),
                t.get("transactionHash"),
            )
            for t in page
        ]
        if len(page) < PAGE:
            break
        off += PAGE
        if off >= max_offset:
            truncated = True
            break
    df = pl.DataFrame(rows, schema=["t", "side", "outcome_index", "price", "size", "tx"], orient="row")
    if df.is_empty():
        return pl.DataFrame(schema=TRADE_SCHEMA), truncated
    df = df.with_columns(
        ts=pl.from_epoch("t", time_unit="s").dt.replace_time_zone("UTC"),
        outcome_index=pl.col("outcome_index").cast(pl.Int8),
    )
    return df.select(list(TRADE_SCHEMA)).sort("ts"), truncated


def outcome0_view(trades: pl.DataFrame) -> pl.DataFrame:
    """Express every trade in outcome-0 terms.

    `px0`: the equivalent outcome-0 price; `buys0`: the taker gained outcome-0 exposure (bought outcome 0 or sold
    outcome 1); `usd`: notional paid.
    """
    return trades.with_columns(
        px0=pl.when(pl.col("outcome_index") == 0).then(pl.col("price")).otherwise(1 - pl.col("price")),
        buys0=(pl.col("side") == "BUY") == (pl.col("outcome_index") == 0),
        usd=pl.col("size") * pl.col("price"),
    )
