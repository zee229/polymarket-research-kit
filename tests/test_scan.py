from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from synth import T0, event, market

from pmrk.interfaces import settlement_agreement
from pmrk.polymarket.data_api import outcome0_view
from pmrk.polymarket.markets import markets_frame
from pmrk.scan import discovery, holdout

SPLIT = date(2026, 4, 1)


def _synthetic(n_events: int = 600, seed: int = 3) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Binary 'culture' markets priced at 0.40 that resolve YES 25% of the time (overpriced YES)."""
    rng = np.random.default_rng(seed)
    evs, snaps = [], []
    for i in range(n_events):
        end = T0 + timedelta(days=i % 60)
        won = rng.random() < 0.25
        evs.append(
            event(
                str(i),
                [market(f"m{i}", prices=("1", "0") if won else ("0", "1"), created=end - timedelta(days=10), end=end)],
                tags=("pop-culture",),
            )
        )
        snaps.append(
            {
                "market_id": f"m{i}",
                "snapshot": "1d",
                "target_ts": end - timedelta(days=1),
                "price_ts": end - timedelta(days=1),
                "p": 0.40,
            }
        )
    m = markets_frame(evs)
    s = pl.DataFrame(
        snaps, schema_overrides={"target_ts": pl.Datetime("us", "UTC"), "price_ts": pl.Datetime("us", "UTC")}
    )
    return m, s


def test_discovery_freezes_mispriced_cell_and_ignores_holdout(tmp_path):
    m, s = _synthetic()
    a = discovery.assemble(s, m, SPLIT)
    assert set(a["split"]) == {"discovery", "holdout"}
    cells, cand = discovery.discover(a, None, min_events=50)
    assert cand.height > 0 and set(cand["side"]) == {"NO"}
    # corrupting holdout outcomes must not change discovery
    flipped = a.with_columns(y=pl.when(pl.col("split") == "holdout").then(1 - pl.col("y")).otherwise(pl.col("y")))
    cells2, _ = discovery.discover(flipped, None, min_events=50)
    assert_frame_equal(cells, cells2, rel_tol=1e-9)  # float sums may differ in the last bits
    digest = discovery.freeze(cand, cells, tmp_path, SPLIT)
    assert len(digest) == 16 and discovery.load_frozen(tmp_path)


def test_frozen_candidates_are_tamper_evident(tmp_path):
    m, s = _synthetic()
    cells, cand = discovery.discover(discovery.assemble(s, m, SPLIT), None, min_events=50)
    discovery.freeze(cand, cells, tmp_path, SPLIT)
    p = tmp_path / "candidates.json"
    p.write_text(p.read_text().replace("NO", "YES"))
    with pytest.raises(ValueError):
        discovery.load_frozen(tmp_path)
    with pytest.raises(FileNotFoundError):
        discovery.load_frozen(tmp_path / "missing")


def test_holdout_mid_vs_real_prints():
    m, s = _synthetic()
    a = discovery.assemble(s, m, SPLIT)
    _, cand = discovery.discover(a, None, min_events=50)
    hold = a.filter(pl.col("split") == "holdout")
    # the only NO prints are far worse than mid + cost: executable ROI must collapse
    trades = outcome0_view(
        pl.DataFrame(
            {
                "market_id": hold["market_id"],
                "ts": hold["target_ts"] + timedelta(minutes=10),
                "side": "BUY",
                "outcome_index": pl.Series([1] * hold.height, dtype=pl.Int8),
                "price": 0.95,
                "size": 10.0,
            }
        )
    )
    r = cand.row(0, named=True)
    spec = [{"category": r["category"], "band": r["band"], "snapshot": r["snapshot"], "side": r["side"]}]
    res = holdout.evaluate(hold, spec, trades, None).row(0, named=True)
    assert res["mid_roi"] > 0.1
    assert res["print_roi"] < 0 and res["print_lim_n"] == 0 and not res["pass"]


class FakeSettlement:
    name = "fake"

    def reproduce(self, markets):
        assert "winner" not in markets.columns
        return markets.select("market_id", winner_reproduced=pl.lit(0, pl.Int8))


def test_settlement_agreement():
    m = markets_frame([event("1", [market("a")]), event("2", [market("b", prices=("0", "1"))])])
    out = settlement_agreement(m, FakeSettlement())
    assert out["reproduced"][0] == 0.5 and out["markets"][0] == 2


def test_split_uses_event_end():
    m, s = _synthetic(n_events=10)
    a = discovery.assemble(s, m, date(2026, 3, 5))
    early = a.filter(pl.col("event_end") < datetime(2026, 3, 5, tzinfo=UTC))
    assert (early["split"] == "discovery").all()
