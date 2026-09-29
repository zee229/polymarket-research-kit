"""Two tiny non-weather models that show the ProbabilityModel plug-in flow.

Both only use `p`, the market's own mid at the decision time, so they are lookahead-free by construction.

    uv run pmrk backtest --model examples/models/baselines.py:MarketPrice
    uv run pmrk backtest --model examples/models/baselines.py:LogitSharpen
"""

from __future__ import annotations

import polars as pl


class MarketPrice:
    """The market itself. Its edge is always minus the cost, so it should never trade: a sanity check."""

    name = "market_price"

    def predict(self, queries: pl.DataFrame) -> pl.DataFrame:
        return queries.select("event_id", "market_id", "decision_time", p_model=pl.col("p"))


class LogitSharpen:
    """Push prices away from 0.5: `p_model = sigmoid(k * logit(p))`.

    With k > 1 favorites get more likely and longshots less likely, i.e. a bet on the favorite-longshot bias.
    The calibration scan found that tilt in mid prices, mostly in thin books; this model lets you check whether any
    of it survives execution in your market set.
    """

    name = "logit_sharpen"

    def __init__(self, k: float = 1.3, eps: float = 1e-4):
        self.k = k
        self.eps = eps

    def predict(self, queries: pl.DataFrame) -> pl.DataFrame:
        p = pl.col("p").clip(self.eps, 1 - self.eps)
        logit = (p / (1 - p)).log()
        return queries.select("event_id", "market_id", "decision_time", p_model=1 / (1 + (-self.k * logit).exp()))
