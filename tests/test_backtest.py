from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest
from synth import T0, event, market, write_prices, write_trades

from pmrk.backtest.engine import SimConfig, metrics, simulate, without_top
from pmrk.backtest.run import build_panel, run
from pmrk.interfaces import OUTCOME_COLUMNS, load_model
from pmrk.polymarket.markets import markets_frame
from pmrk.polymarket.store import save_markets

H = timedelta(hours=1)
EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "models" / "baselines.py"


def _panel(**over) -> pl.DataFrame:
    row = {
        "event_id": "e",
        "market_id": "m",
        "decision_time": T0,
        "day": date(2026, 3, 1),
        "p": 0.40,
        "p_model": 0.60,
        "winner": 0,
        "fees_enabled": True,
        "fee_rate": 0.05,
        "fee_exponent": 1.0,
        "vol_window": 1000.0,
        "traded_near": True,
    }
    row.update(over)
    return pl.DataFrame([row], schema_overrides={"decision_time": pl.Datetime("us", "UTC")})


def test_simulate_pnl_arithmetic_with_fee():
    tr = simulate(_panel(), SimConfig(threshold=0.05, cost=0.02, flat_stake=20.0))
    assert tr.height == 1 and tr["side"][0] == "YES"
    price = 0.42
    fee = 0.05 * price * (1 - price)
    shares = 20.0 / (price + fee)
    assert tr["shares"][0] == pytest.approx(shares)
    assert tr["pnl"][0] == pytest.approx(shares * (1 - price - fee))


def test_no_fee_where_disabled_unless_fee_everywhere():
    p = _panel(fees_enabled=False)
    assert simulate(p, SimConfig(cost=0.0))["fee"][0] == 0.0
    assert simulate(p, SimConfig(cost=0.0, fee_everywhere=True))["fee"][0] > 0


def test_no_fill_without_prints_and_volume_cap():
    assert simulate(_panel(traded_near=False), SimConfig()).is_empty()
    tr = simulate(_panel(vol_window=30.0), SimConfig(vol_frac=0.1))
    assert tr["shares"][0] == pytest.approx(3.0)


def test_one_entry_per_market_side_and_daily_cap():
    rows = pl.concat([_panel(decision_time=T0 + i * H) for i in range(5)])
    assert simulate(rows, SimConfig()).height == 1
    many = pl.concat([_panel(market_id=f"m{i}") for i in range(10)])
    tr = simulate(many, SimConfig(flat_stake=20.0, max_per_day=50.0))
    assert tr["spend"].sum() <= 50.0 and tr.height == 2


def test_losing_no_side_and_metrics():
    tr = simulate(_panel(p=0.6, p_model=0.3, winner=0), SimConfig(cost=0.01))
    assert tr["side"][0] == "NO" and not tr["won"][0] and tr["pnl"][0] < 0
    m = metrics(tr)
    assert m["n_trades"] == 1 and m["roi"] == pytest.approx(-1.0)
    assert without_top(tr, k=1)["pnl_without_top1"] == 0.0


class SpyModel:
    """Records what it was shown; predicts 0.9 for outcome 0 of market 'a'."""

    name = "spy"

    def __init__(self):
        self.seen: list[pl.DataFrame] = []

    def predict(self, q: pl.DataFrame) -> pl.DataFrame:
        self.seen.append(q)
        return q.select(
            "event_id",
            "market_id",
            "decision_time",
            p_model=pl.when(pl.col("market_id") == "a").then(0.9).otherwise(0.1),
        )


def _store(data_dir):
    end = T0 + timedelta(days=2)
    ms = [market("a", end=end), market("b", prices=("0", "1"), end=end)]
    m = save_markets(markets_frame([event("1", ms, neg_risk=True)]))
    hours = [T0 + i * H for i in range(0, 60)]
    write_prices(data_dir, "1", [("a", t, 0.5) for t in hours] + [("b", t, 0.55) for t in hours])
    write_trades(
        data_dir,
        "1",
        [("a", t + timedelta(minutes=5), "BUY", 0, 0.5, 100.0) for t in hours]
        + [("b", t + timedelta(minutes=5), "BUY", 0, 0.55, 100.0) for t in hours],
    )
    return m


def test_build_panel_hides_outcomes_and_respects_guard(data_dir):
    m = _store(data_dir)
    spy = SpyModel()
    panel = build_panel(m, spy)
    assert panel.height > 0
    for q in spy.seen:
        assert not set(q.columns) & set(OUTCOME_COLUMNS)
    known = m["known_ts"].min()
    assert (panel["decision_time"] <= known - H).all()
    # both buckets priced -> q_market and p_model normalized within each decision
    s = panel.group_by("decision_time").agg(pl.col("q_market").sum(), pl.col("p_model").sum())
    assert s["q_market"].to_list() == pytest.approx([1.0] * s.height)
    assert s["p_model"].to_list() == pytest.approx([1.0] * s.height)


def test_market_price_model_never_trades(data_dir):
    m = _store(data_dir)
    panel = build_panel(m, load_model(f"{EXAMPLES}:MarketPrice"))
    res, trades = run(panel, SimConfig(cost=0.01), cut=date(2026, 3, 1))
    assert trades.is_empty() and res["test_main"]["n_trades"] == 0


def test_load_model_rejects_non_models(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("class NotAModel:\n    pass\n")
    with pytest.raises(TypeError):
        load_model(f"{bad}:NotAModel")
    with pytest.raises(ValueError):
        load_model("no_colon")
