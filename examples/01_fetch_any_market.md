# 01. Fetch any market

Everything lands under `./data` (or `$PMRK_DATA_DIR`). Downloads are cached and resumable: rerun a command after an
interruption and it continues where it stopped.

## Markets

```bash
# one or more events by slug or numeric id
uv run pmrk fetch markets --event fed-decision-in-october

# every closed event with a tag, by end date
uv run pmrk fetch markets --tag fed --start 2026-01-01 --end 2026-09-01

# every closed event in a window (all categories; large)
uv run pmrk fetch markets --start 2026-09-01 --end 2026-09-08
```

Each fetch is merged into `data/markets.parquet`, one row per market:

- `outcome0` / `outcome1` and `winner` (index of the outcome that paid 1; null for voided or 50-50);
- `neg_risk`, `markets_in_event`, `structure` (`negrisk`, `multi_market`, `binary`);
- `end_ts`, `closed_ts`, `uma_end_ts` and `known_ts` (the earliest of the three);
- `fees_enabled`, `fee_rate`, `fee_exponent` per market;
- `clean` (two outcomes, closed, exact 1/0, not disputed, not an `arch-` copy) and `resolved_early_1h`;
- `category` from tag slugs (`pmrk.polymarket.categories`, override with your own mapping).

## Prices and trades

```bash
uv run pmrk fetch prices --clean-only --lookback 14d              # 1-minute outcome-0 history
uv run pmrk fetch prices --category politics --fidelity 60        # hourly, full life
uv run pmrk fetch trades --clean-only                             # taker trades
```

Stored per event in `data/prices/<event_id>.parquet` and `data/trades/<event_id>.parquet`.

## From Python

```python
from datetime import date

from pmrk.polymarket import gamma
from pmrk.polymarket.markets import markets_frame
from pmrk.polymarket.store import download_prices, load_prices, save_markets

events = gamma.iter_events(date(2026, 8, 1), date(2026, 9, 1), tag_slug="nba")
markets = save_markets(markets_frame(events))
download_prices(markets.filter("clean"), fidelity_min=5)
prices = load_prices(markets["event_id"].unique())
```
