# 03. Backtest your own model

Any class with a `name` and a `predict(queries) -> DataFrame` method works (see [docs/plugins.md](../docs/plugins.md)).
Two tiny examples live in [models/baselines.py](models/baselines.py).

```bash
uv run pmrk fetch markets --tag fed --start 2026-01-01 --end 2026-09-01
uv run pmrk fetch prices --clean-only --lookback 14d
uv run pmrk fetch trades --clean-only

# sanity check: the market itself can never beat the market after costs, so it should make zero trades
uv run pmrk backtest --model examples/models/baselines.py:MarketPrice

# a bet on the favorite-longshot bias
uv run pmrk backtest --model examples/models/baselines.py:LogitSharpen --cost 0.01

# the same with measured taker cost by price band instead of a flat cost
uv run pmrk costs --by category
uv run pmrk backtest --model examples/models/baselines.py:LogitSharpen --band-cost median --plot  # --plot needs uv sync --extra plots
```

What happens:

1. An hourly decision grid per event, plus any times your model adds via `decision_times`.
2. Every market that passes the guard at that time (nothing within 1 h of the earliest end / close / resolution).
3. The last mid at or before the decision (at most 60 min stale), normalized within negRisk events.
4. Your model is called with resolution columns removed.
5. Both sides are priced at mid + cost + fee; a fill needs a real print within ±60 min and is capped at 10% of that
   window's volume.
6. The edge threshold is tuned on the first half of events and applied to the second half; results at 1, 3 and 5¢
   with block-bootstrap CIs, ROI without the top 5 days, and breakdowns go to
   `data/reports/backtest/<model name>/results.json`.

A small market set gives wide CIs. That is the point: the CI tells you whether you have learned anything.
