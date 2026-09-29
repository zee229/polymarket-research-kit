# Plugins: bring your own model or settlement source

The core knows nothing about any domain. Domain code plugs in through two small protocols in `pmrk.interfaces`,
both batch-shaped (polars in, polars out) because a backtest asks for millions of probabilities.

## `ProbabilityModel`

```python
class ProbabilityModel(Protocol):
    name: str

    def predict(self, queries: pl.DataFrame) -> pl.DataFrame: ...
```

**Input**: one row per `(event_id, market_id, decision_time)` that passed the lookahead guard, with:

| Column | Meaning |
|---|---|
| `event_id`, `market_id`, `decision_time` | keys; `decision_time` is UTC |
| `p` | the market's own outcome-0 mid at `decision_time` (at most 60 min stale) |
| `q_market`, `p_sum` | `p` normalized within a negRisk event, and the raw sum of mids |
| `question`, `group_item_title`, `outcome0`, `outcome1` | market text and outcome labels (any language) |
| `category`, `event_slug`, `series_slug`, `neg_risk`, `end_ts` | metadata |
| `fees_enabled`, `fee_rate`, `fee_exponent`, `trigger` | fee schedule; which decision grid produced the row |

Resolution columns (`winner`, `outcome_prices`, close and resolution timestamps, `known_ts`, `clean`, ...) are
removed before the call. `tests/test_backtest.py` checks that.

**Output**: `event_id`, `market_id`, `decision_time`, `p_model` (probability that outcome 0 wins). Rules:

- Use only information available at `decision_time`. The core cannot check what your model reads from its own
  data sources, so write a lookahead test for every feature (see `tests/test_weather_lookahead.py`).
- Rows you cannot price can be omitted. For negRisk events, a decision is only kept if you priced every live
  market of the event, and `p_model` is floored at 1e-4 and renormalized to sum to 1.
- Extra columns pass through to the trade log, so you can break results down by them later (the weather model
  returns `hgroup`, `city`, `station_icao`).
- A `day` column, if you return one, replaces the UTC date as the block for daily caps and the bootstrap (the
  weather model uses the local observation date).

### Optional: `decision_times(markets)`

By default decisions happen on an hourly grid. If your information arrives at specific moments, return extra rows
`(event_id, decision_time[, trigger])`. They pass through the same guards. The weather model adds one decision a
minute after each forecast run becomes available and one `obs_lag` after each METAR.

### Running it

```bash
uv run pmrk backtest --model my_models.py:MyModel                  # file path
uv run pmrk backtest --model mypkg.models:MyModel --band-cost median  # importable module, measured costs
```

A class is instantiated without arguments; to pass parameters, expose a configured instance
(`my_model = MyModel(k=2)`) and point at it: `--model my_models.py:my_model`.

Results go to `<data>/reports/backtest/<model.name>/`: `results.json` (tuning grid, test results at 1/3/5¢ for flat
and Kelly sizing, bootstrap CIs, ROI without the top 5 days, breakdowns) and `trades.parquet`.

In Python:

```python
from pmrk.backtest.engine import SimConfig
from pmrk.backtest.run import build_panel, run
from pmrk.polymarket.store import load_markets

panel = build_panel(load_markets(), MyModel())
results, trades = run(panel, SimConfig(cost=0.03))
```

## `SettlementSource`

```python
class SettlementSource(Protocol):
    name: str

    def reproduce(self, markets: pl.DataFrame) -> pl.DataFrame: ...  # market_id, winner_reproduced (0/1/null)
```

Implement it when outcomes can be rebuilt from external data (weather observations, sports box scores, official
statistics, prices at a fixed time). Then:

```python
from pmrk.interfaces import settlement_agreement

settlement_agreement(load_markets(), MySource(), by=["category"])
```

gives the share of cleanly resolved markets whose outcome you reproduce. Anything below the high 90s means you are
modelling a different quantity than the market resolves on.

## Example plugins

- `examples/models/baselines.py`: `MarketPrice` (the market itself, which should never trade, as a sanity check)
  and `LogitSharpen` (a one-parameter bet on the favorite-longshot bias).
- `pmrk.weather`: a full case study. `WeatherModel` implements `ProbabilityModel` with `decision_times`, and
  `MetarSettlement` implements `SettlementSource`. It only imports the public core API.

## Rules for a new domain package

- Keep everything domain-specific (units, stations, tickers, team names, text parsing) inside your package.
- If you parse market text, prefer structured fields and stable identifiers. Rules text can be in any language
  and templates change over time; parse per market and surface every failure instead of silently dropping it.
- If your plugin needs a hack in the core, change the core design instead.
