"""Synthetic Gamma payloads and stored price/trade files for offline tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import polars as pl

T0 = datetime(2026, 3, 1, tzinfo=UTC)


def iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def market(
    mid: str,
    *,
    prices=("1", "0"),
    outcomes=("Yes", "No"),
    created=T0,
    end=T0 + timedelta(days=10),
    closed=None,
    uma_end=None,
    fees=False,
    rate=None,
    statuses=("proposed",),
    title=None,
) -> dict:
    closed = closed if closed is not None else end + timedelta(hours=2)
    return {
        "id": mid,
        "conditionId": f"0x{mid}",
        "question": f"Question {mid}?",
        "groupItemTitle": title,
        "outcomes": json.dumps(list(outcomes)),
        "outcomePrices": json.dumps(list(prices)),
        "clobTokenIds": json.dumps([f"tok{mid}a", f"tok{mid}b"]),
        "closed": True,
        "createdAt": iso(created),
        "startDate": iso(created),
        "endDate": iso(end),
        "closedTime": closed.strftime("%Y-%m-%d %H:%M:%S+00"),
        "umaEndDate": iso(uma_end or closed),
        "volumeNum": 1000.0,
        "feesEnabled": fees,
        "feeSchedule": {"exponent": 1, "rate": rate, "takerOnly": True} if rate is not None else None,
        "umaResolutionStatuses": json.dumps(list(statuses)),
    }


def event(eid: str, markets: list[dict], *, neg_risk=False, tags=("politics",), slug=None) -> dict:
    return {
        "id": eid,
        "slug": slug or f"event-{eid}",
        "title": f"Event {eid}",
        "negRisk": neg_risk,
        "tags": [{"slug": t} for t in tags],
        "markets": markets,
    }


def write_prices(data_dir, event_id: str, rows: list[tuple[str, datetime, float]]) -> None:
    (data_dir / "prices").mkdir(exist_ok=True)
    pl.DataFrame(
        rows, schema={"market_id": pl.Utf8, "ts": pl.Datetime("us", "UTC"), "p": pl.Float64}, orient="row"
    ).write_parquet(data_dir / "prices" / f"{event_id}.parquet")


def write_trades(data_dir, event_id: str, rows: list[tuple[str, datetime, str, int, float, float]]) -> None:
    (data_dir / "trades").mkdir(exist_ok=True)
    schema = {
        "market_id": pl.Utf8,
        "ts": pl.Datetime("us", "UTC"),
        "side": pl.Utf8,
        "outcome_index": pl.Int8,
        "price": pl.Float64,
        "size": pl.Float64,
    }
    pl.DataFrame(rows, schema=schema, orient="row").with_columns(
        tx=pl.lit(None, pl.Utf8), truncated=pl.lit(False)
    ).write_parquet(data_dir / "trades" / f"{event_id}.parquet")
