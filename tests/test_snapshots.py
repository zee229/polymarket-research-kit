from datetime import timedelta

import polars as pl
import pytest
from synth import T0, event, market

from pmrk.polymarket.markets import markets_frame
from pmrk.snapshots.horizons import decision_grid, market_rows, parse_duration, snapshot_targets
from pmrk.snapshots.prices import normalize_event, overround, price_at

H = timedelta(hours=1)


def test_parse_duration():
    assert parse_duration("90m") == timedelta(minutes=90)
    assert parse_duration("7d") == timedelta(days=7)
    with pytest.raises(ValueError):
        parse_duration("7 weeks")


def test_targets_never_at_or_after_known_minus_buffer():
    early = T0 + timedelta(days=5)  # "by date" market that resolved 5 days before its end date
    m = markets_frame([event("1", [market("a", created=T0, end=T0 + timedelta(days=10), closed=early)])])
    t = snapshot_targets(m)
    assert t.height > 0
    assert (t["target_ts"] <= early - H).all()
    # the 1d/6h/1h horizons are anchored on end_ts and all fall after the early close: dropped
    assert not set(t["snapshot"]) & {"1d", "6h", "1h"}


def test_targets_not_before_creation():
    m = markets_frame([event("1", [market("a", created=T0, end=T0 + timedelta(days=2))])])
    t = snapshot_targets(m)
    assert "30d" not in set(t["snapshot"]) and "7d" not in set(t["snapshot"])
    assert (t["target_ts"] >= T0).all()


def test_decision_grid_and_per_market_guard():
    # negRisk event: one bucket closes early (became impossible), the rest at the end
    ms = [
        market("a", end=T0 + timedelta(days=2), closed=T0 + timedelta(days=1)),
        market("b", prices=("0", "1"), end=T0 + timedelta(days=2)),
    ]
    m = markets_frame([event("1", ms, neg_risk=True)])
    grid = decision_grid(m)
    assert grid["decision_time"].min() == T0 + H
    rows = market_rows(grid, m)
    last_a = rows.filter(market_id="a")["decision_time"].max()
    assert last_a <= T0 + timedelta(days=1) - H
    assert rows.filter(market_id="b")["decision_time"].max() > last_a


def _rows(ps: list[tuple[str, float | None]], neg: bool = True) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "event_id": "e",
            "market_id": [m for m, _ in ps],
            "decision_time": T0,
            "neg_risk": neg,
            "p": [p for _, p in ps],
        },
        schema_overrides={"decision_time": pl.Datetime("us", "UTC"), "p": pl.Float64},
    )


def test_price_at_respects_staleness():
    prices = pl.DataFrame(
        {"market_id": ["a", "a", "b"], "ts": [T0 - 3 * H, T0 - timedelta(minutes=5), T0 - 2 * H], "p": [0.2, 0.3, 0.9]},
        schema_overrides={"ts": pl.Datetime("us", "UTC")},
    )
    rows = _rows([("a", None), ("b", None)]).drop("p")
    out = price_at(rows, prices).sort("market_id")
    assert out["p"].to_list() == [0.3, None]  # b's last print is 2 h old


def test_price_at_never_uses_future_points():
    prices = pl.DataFrame(
        {"market_id": ["a"], "ts": [T0 + timedelta(seconds=1)], "p": [0.99]},
        schema_overrides={"ts": pl.Datetime("us", "UTC")},
    )
    assert price_at(_rows([("a", None)]).drop("p"), prices)["p"].to_list() == [None]


def test_normalize_event_removes_overround():
    out = normalize_event(_rows([("a", 0.5), ("b", 0.3), ("c", 0.3)]))
    assert out["p_sum"][0] == pytest.approx(1.1)
    assert out["q_market"].sum() == pytest.approx(1.0)


def test_incomplete_negrisk_group_dropped_binary_kept():
    assert normalize_event(_rows([("a", 0.5), ("b", None)])).is_empty()
    out = normalize_event(_rows([("a", 0.7)], neg=False))
    assert out["q_market"].to_list() == [0.7]


def test_overround_requires_complete_event():
    rows = (
        _rows([("a", 0.6), ("b", 0.5)]).rename({"decision_time": "target_ts"}).with_columns(markets_in_event=pl.lit(2))
    )
    assert overround(rows)["sum_p"].to_list() == [pytest.approx(1.1)]
    assert overround(rows.head(1)).is_empty()
