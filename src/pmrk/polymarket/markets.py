"""Normalize raw Gamma events into one row per market.

Every Polymarket market is a two-outcome binary; multi-outcome events are groups of binaries (negRisk). Outcome
labels vary ("Yes/No", "Over/Under", "Up/Down", team names), so the table is keyed on *outcome index*: `p` always
means the price of outcome 0 (the first CLOB token) and `winner` is the index of the outcome that paid 1.

Timestamps are kept separate because they mean different things:
- `end_ts`: scheduled end date shown on the market;
- `closed_ts`: when trading actually closed (`closedTime`), often long before `end_ts` for "by date" markets;
- `uma_end_ts`: when the UMA oracle round ended (resolution);
- `known_ts`: `min(end, closed, uma_end)`, the earliest moment the outcome may have been public. Snapshot and
  decision guards are anchored on it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from datetime import datetime
from typing import Any

import polars as pl

from pmrk.polymarket.categories import DEFAULT_CATEGORIES, CategoryMap, categorize

MARKET_SCHEMA: dict[str, pl.DataType] = {
    "event_id": pl.Utf8,
    "event_slug": pl.Utf8,
    "event_title": pl.Utf8,
    "series_slug": pl.Utf8,
    "category": pl.Utf8,
    "neg_risk": pl.Boolean,
    "market_id": pl.Utf8,
    "condition_id": pl.Utf8,
    "question": pl.Utf8,
    "group_item_title": pl.Utf8,
    "outcome0": pl.Utf8,
    "outcome1": pl.Utf8,
    "n_outcomes": pl.Int32,
    "token0": pl.Utf8,
    "token1": pl.Utf8,
    "created_ts": pl.Datetime("us", "UTC"),
    "start_ts": pl.Datetime("us", "UTC"),
    "end_ts": pl.Datetime("us", "UTC"),
    "closed_ts": pl.Datetime("us", "UTC"),
    "uma_end_ts": pl.Datetime("us", "UTC"),
    "closed": pl.Boolean,
    "outcome_prices": pl.Utf8,
    "winner": pl.Int8,
    "disputed": pl.Boolean,
    "archived_copy": pl.Boolean,
    "volume": pl.Float64,
    "fees_enabled": pl.Boolean,
    "fee_rate": pl.Float64,
    "fee_exponent": pl.Float64,
}


def parse_ts(value: str | None) -> datetime | None:
    """Gamma timestamps come as '...Z', '... +00' or '...+00:00'; return aware UTC datetimes."""
    if not value:
        return None
    v = value.strip().replace(" ", "T").replace("Z", "+00:00")
    if re.search(r"[+-]\d{2}$", v):
        v += ":00"
    return datetime.fromisoformat(v)


def _json_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    try:
        out = json.loads(value) if value else []
    except (TypeError, ValueError):
        return []
    return out if isinstance(out, list) else []


def winner_index(outcome_prices: Any, closed: bool) -> int | None:
    """0 or 1 if exactly one outcome settled at 1; None for open, 50-50/voided or partial resolutions."""
    prices = _json_list(outcome_prices)
    if not closed or len(prices) != 2:
        return None
    try:
        values = [float(p) for p in prices]
    except (TypeError, ValueError):
        return None
    if values == [1.0, 0.0]:
        return 0
    if values == [0.0, 1.0]:
        return 1
    return None


def market_row(event: dict[str, Any], market: dict[str, Any], category: str) -> dict[str, Any]:
    outcomes = _json_list(market.get("outcomes"))
    tokens = _json_list(market.get("clobTokenIds"))
    fee = market.get("feeSchedule") or {}
    statuses = _json_list(market.get("umaResolutionStatuses"))
    closed = bool(market.get("closed"))
    return {
        "event_id": str(event["id"]),
        "event_slug": event.get("slug"),
        "event_title": event.get("title"),
        "series_slug": event.get("seriesSlug"),
        "category": category,
        "neg_risk": bool(event.get("negRisk") or event.get("enableNegRisk")),
        "market_id": str(market["id"]),
        "condition_id": market.get("conditionId"),
        "question": market.get("question"),
        "group_item_title": market.get("groupItemTitle"),
        "outcome0": outcomes[0] if outcomes else None,
        "outcome1": outcomes[1] if len(outcomes) > 1 else None,
        "n_outcomes": len(outcomes),
        "token0": tokens[0] if tokens else None,
        "token1": tokens[1] if len(tokens) > 1 else None,
        "created_ts": parse_ts(market.get("createdAt")),
        "start_ts": parse_ts(market.get("startDate") or event.get("startDate")),
        "end_ts": parse_ts(market.get("endDate") or event.get("endDate")),
        "closed_ts": parse_ts(market.get("closedTime")),
        "uma_end_ts": parse_ts(market.get("umaEndDate")),
        "closed": closed,
        "outcome_prices": json.dumps(_json_list(market.get("outcomePrices"))),
        "winner": winner_index(market.get("outcomePrices"), closed),
        "disputed": "disputed" in statuses,
        "archived_copy": (event.get("slug") or "").startswith("arch-"),
        "volume": float(market.get("volumeNum") or 0.0),
        "fees_enabled": bool(market.get("feesEnabled")),
        "fee_rate": float(fee.get("rate") or 0.0),
        "fee_exponent": float(fee.get("exponent") or 1.0),
    }


def markets_frame(events: Iterable[dict[str, Any]], categories: CategoryMap = DEFAULT_CATEGORIES) -> pl.DataFrame:
    """One row per market with structure, clean-resolution and timing columns (duplicates dropped)."""
    rows = []
    for ev in events:
        category = categorize([t.get("slug") for t in ev.get("tags") or []], categories)
        rows.extend(market_row(ev, m, category) for m in ev.get("markets") or [])
    df = pl.DataFrame(rows, schema=MARKET_SCHEMA).unique("market_id", keep="last", maintain_order=True)
    return with_derived(df)


def with_derived(df: pl.DataFrame) -> pl.DataFrame:
    """Structure, `known_ts`, early-resolution flag and the `clean` filter."""
    df = df.drop(
        [c for c in ("markets_in_event", "structure", "known_ts", "resolved_early_1h", "clean") if c in df.columns]
    )
    df = df.with_columns(markets_in_event=pl.len().over("event_id").cast(pl.Int32))
    return df.with_columns(
        structure=pl.when(pl.col("neg_risk"))
        .then(pl.lit("negrisk"))
        .when(pl.col("markets_in_event") > 1)
        .then(pl.lit("multi_market"))
        .otherwise(pl.lit("binary")),
        known_ts=pl.min_horizontal("end_ts", "closed_ts", "uma_end_ts"),
        resolved_early_1h=(pl.col("end_ts") - pl.col("closed_ts")) > pl.duration(hours=1),
        clean=(pl.col("n_outcomes") == 2)
        & pl.col("closed")
        & pl.col("winner").is_not_null()
        & ~pl.col("disputed")
        & ~pl.col("archived_copy"),
    )
