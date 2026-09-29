from datetime import timedelta

import polars as pl
import pytest
from synth import T0

from pmrk.execution.costs import attach_cost, band, band_labels, cost_table, trade_costs
from pmrk.execution.fees import taker_fee_per_share
from pmrk.execution.fills import first_print, reaction_minutes, window_volume
from pmrk.polymarket.data_api import outcome0_view

M = timedelta(minutes=1)


def _fee(price, enabled, rate, exponent=1.0):
    df = pl.DataFrame({"price": [price], "fees_enabled": [enabled], "fee_rate": [rate], "fee_exponent": [exponent]})
    return df.select(fee=taker_fee_per_share(pl.col("price")))["fee"][0]


def test_fee_matches_documented_formula():
    assert _fee(0.5, True, 0.05) == pytest.approx(0.05 * 0.25)
    assert _fee(0.05, True, 0.05) == pytest.approx(0.05 * 0.05 * 0.95)


def test_fee_zero_when_disabled_or_zero_rate():
    assert _fee(0.5, False, 0.05) == 0.0
    assert _fee(0.5, True, 0.0) == 0.0


def test_band_labels_and_edges():
    assert band_labels([0.01, 0.5])[0] == "0-1" and band_labels([0.01, 0.5])[-1] == "50-100"
    b = pl.DataFrame({"p": [0.0, 0.01, 0.349, 0.35, 0.995]}).select(band("p"))["p"].to_list()
    assert b == ["0-1", "1-3", "20-35", "35-50", "99-100"]


def _trades(rows):
    schema = {
        "market_id": pl.Utf8,
        "ts": pl.Datetime("us", "UTC"),
        "side": pl.Utf8,
        "outcome_index": pl.Int8,
        "price": pl.Float64,
        "size": pl.Float64,
    }
    return outcome0_view(pl.DataFrame(rows, schema=schema, orient="row"))


def test_outcome0_view_maps_all_four_trade_types():
    t = _trades(
        [
            ("a", T0, "BUY", 0, 0.4, 1),
            ("a", T0, "SELL", 0, 0.4, 1),
            ("a", T0, "BUY", 1, 0.6, 1),
            ("a", T0, "SELL", 1, 0.6, 1),
        ]
    )
    assert t["buys0"].to_list() == [True, False, False, True]
    assert t["px0"].to_list() == pytest.approx([0.4, 0.4, 0.4, 0.4])


def test_trade_costs_sign_positive_when_worse_than_mid():
    mids = pl.DataFrame(
        {"market_id": ["a"], "ts": [T0], "p": [0.50]}, schema_overrides={"ts": pl.Datetime("us", "UTC")}
    )
    t = _trades([("a", T0 + M, "BUY", 0, 0.52, 10), ("a", T0 + M, "BUY", 1, 0.53, 10)])
    c = trade_costs(t, mids)
    assert c["cost"].to_list() == pytest.approx([0.02, 0.03])  # buying NO at .53 = selling YES at .47
    table = cost_table(c)
    assert table["median_cost"][0] == pytest.approx(0.025)


def test_trade_costs_ignore_stale_mid():
    mids = pl.DataFrame({"market_id": ["a"], "ts": [T0], "p": [0.5]}, schema_overrides={"ts": pl.Datetime("us", "UTC")})
    assert trade_costs(_trades([("a", T0 + 10 * M, "BUY", 0, 0.52, 1)]), mids).is_empty()


def test_attach_cost_falls_back_to_pooled_then_default():
    table = pl.DataFrame({"category": ["sports"], "band": ["35-50"], "median_cost": [0.01]})
    rows = pl.DataFrame({"category": ["sports", "politics", "sports"], "p": [0.4, 0.4, 0.9]})
    out = attach_cost(rows, table, by=["category"], default=0.02)
    assert out["cost"].to_list() == [0.01, 0.01, 0.02]


def test_window_volume_and_first_print():
    rows = pl.DataFrame(
        {"market_id": ["a", "a"], "decision_time": [T0, T0 + timedelta(hours=5)], "side": "YES", "limit": 0.5},
        schema_overrides={"decision_time": pl.Datetime("us", "UTC")},
    )
    t = _trades(
        [
            ("a", T0 - 30 * M, "BUY", 0, 0.45, 10),
            ("a", T0 + 5 * M, "SELL", 0, 0.44, 7),
            ("a", T0 + 10 * M, "BUY", 0, 0.48, 20),
            ("a", T0 + 20 * M, "BUY", 0, 0.55, 5),
        ]
    )
    wv = window_volume(rows, t).sort("decision_time")
    assert wv["vol_window"].to_list() == [42.0, 0.0] and wv["traded_near"].to_list() == [True, False]
    fp = first_print(rows, t, limit_col="limit").sort("decision_time")
    first = fp.row(0, named=True)
    # the SELL at +5 min is the other side; the print before T is ignored
    assert first["print_px"] == 0.48 and first["fillable_shares"] == 20.0
    assert fp["print_px"][1] is None


def test_first_print_for_no_side_uses_outcome1_price():
    rows = pl.DataFrame(
        {"market_id": ["a"], "decision_time": [T0], "side": "NO"},
        schema_overrides={"decision_time": pl.Datetime("us", "UTC")},
    )
    fp = first_print(rows, _trades([("a", T0 + M, "BUY", 1, 0.30, 5)]))
    assert fp["print_px"][0] == pytest.approx(0.30)


def test_reaction_minutes():
    mids = pl.DataFrame(
        {"market_id": "a", "ts": [T0 - M, T0 + 3 * M, T0 + 12 * M], "p": [0.9, 0.88, 0.5]},
        schema_overrides={"ts": pl.Datetime("us", "UTC")},
    )
    ev = pl.DataFrame(
        {"market_id": ["a"], "t0": [T0], "winner": [1]}, schema_overrides={"t0": pl.Datetime("us", "UTC")}
    )
    assert reaction_minutes(ev, mids)["reaction_min"][0] == pytest.approx(12.0)
