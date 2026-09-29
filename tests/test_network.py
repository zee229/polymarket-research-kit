"""Live API smoke tests. Skipped by default; run with `uv run pytest -m network`."""

from datetime import UTC, datetime, timedelta

import pytest

from pmrk.polymarket import clob, data_api, gamma
from pmrk.polymarket.markets import markets_frame

pytestmark = pytest.mark.network


def test_fetch_resolved_event_prices_and_trades():
    m = markets_frame([gamma.get_event("fed-decision-in-october")])
    assert m.height > 1 and m["neg_risk"].all() and m["clean"].any()
    row = m.filter("clean").row(0, named=True)
    end = row["known_ts"]
    px = clob.price_history(row["token0"], end - timedelta(days=6), end, fidelity_min=60)
    assert px.height > 0 and px["p"].is_between(0, 1).all()
    trades, _ = data_api.taker_trades(row["condition_id"], max_offset=500)
    assert trades.height > 0


def test_keyset_crawl_window():
    day = datetime.now(UTC).date() - timedelta(days=30)
    events = list(gamma.iter_events(day, day + timedelta(days=1), tag_slug="politics"))
    assert all("markets" in e for e in events)
