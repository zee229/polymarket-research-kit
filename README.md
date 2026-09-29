# polymarket-research-kit

Research toolkit for Polymarket: data loaders, execution-realistic backtests and calibration scans.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)
[![CI](https://github.com/zee229/polymarket-research-kit/actions/workflows/ci.yml/badge.svg)](https://github.com/zee229/polymarket-research-kit/actions/workflows/ci.yml)

## TL;DR

- Loaders, snapshotting, cost measurement, statistics and a backtester for **any** Polymarket market: binary or
  multi-outcome (negRisk), sports, politics, crypto, mentions, weather.
- I used it for two studies. The first tried to beat daily temperature markets with calibrated weather forecasts.
  The second scanned every category for miscalibrated prices.
- Both were a no-go. The market won twice.
- The lesson: **apparent mispricings live inside the spread. At prices you can actually trade, Polymarket is well
  calibrated.** Most of this repo exists to stop you from fooling yourself with mid prices.
- Research only. There is no order placement, no wallet and no keys, and there never will be.

## What's inside

| Module | What it does | Why it exists |
|---|---|---|
| `pmrk.polymarket` | Gamma events and markets (by id, slug, series, tag, end-date window), CLOB price history, Data API taker trades, per-market fee schedules, on-disk cache, resumable downloads | The public APIs have sharp edges (see [gotchas](docs/polymarket-api-gotchas.md)). The loaders handle them once. |
| `pmrk.snapshots` | Prices at fixed horizons before resolution, hourly decision grids, negRisk normalization and overround | Every time is guarded: nothing at or after the earliest of end / close / resolution, and nothing in the hour before it. |
| `pmrk.execution` | Measured taker cost vs mid by price band, fills against real prints, volume caps, the Polymarket fee formula | Mid prices are not executable. This is where both studies died. |
| `pmrk.stats` | Log loss, Brier, CRPS, reliability cells with event-clustered errors, block bootstrap, Benjamini-Hochberg | Snapshots of one event are not independent, and a screen over hundreds of cells will always find something. |
| `pmrk.scan` | Calibration scanner: discovery/holdout split, frozen (hash-checked) candidate list, holdout at mid vs real prints | Separates "looks miscalibrated" from "you could have made money". |
| `pmrk.backtest` | Event-driven taker backtest: decision times, guarded market rows, model, fills, fees, PnL, tuning half vs test half | Works with any model that implements `ProbabilityModel`. |
| `pmrk.weather` | The weather case study as a plugin: METAR settlement reproduction, exact ECMWF/GFS runs from AWS, EMOS | Proof that the core is domain-agnostic. It only uses the public interfaces. |

## Install

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/zee229/polymarket-research-kit
cd polymarket-research-kit
uv sync                      # core
uv sync --extra weather      # + weather case study (ecCodes for GRIB decoding)
uv run pytest -q             # offline tests
```

Data goes to `./data` (git-ignored). Set `PMRK_DATA_DIR` or pass `--data-dir` to put it elsewhere.

## Quickstart

Fed-related markets that ended between January and September 2026:

```bash
uv run pmrk fetch markets --tag fed --start 2026-01-01 --end 2026-09-01
uv run pmrk fetch prices --clean-only --lookback 14d     # 1-minute history, last 14 days before resolution
uv run pmrk fetch trades --clean-only                    # every taker print
uv run pmrk snapshot --from-stored                       # prices 7d / 1d / 6h / 1h before the end, guarded
uv run pmrk costs --by category                          # what takers really paid vs mid, by price band
uv run pmrk backtest --model examples/models/baselines.py:LogitSharpen --cost 0.01
```

For a calibration scan you need many more events (thousands per category). The workflow is in
[examples/02_calibration_scan.md](examples/02_calibration_scan.md).

## Bring your own model

A model gets one row per (event, market, decision time) with the market's metadata and its own mid `p` at that
time. Resolution columns are stripped before the call. It returns the probability that outcome 0 wins.

```python
import polars as pl


class LogitSharpen:
    """Bet on the favorite-longshot bias: push prices away from 0.5."""

    name = "logit_sharpen"

    def __init__(self, k: float = 1.3):
        self.k = k

    def predict(self, queries: pl.DataFrame) -> pl.DataFrame:
        p = pl.col("p").clip(1e-4, 1 - 1e-4)
        logit = (p / (1 - p)).log()
        return queries.select("event_id", "market_id", "decision_time", p_model=1 / (1 + (-self.k * logit).exp()))
```

```bash
uv run pmrk backtest --model path/to/model.py:LogitSharpen
```

Models can also add their own decision times (for example "one minute after a new forecast is published") and
extra output columns for breakdowns. See [docs/plugins.md](docs/plugins.md).

## Case studies

### 1. Daily highest-temperature markets

Hypothesis: calibrated public weather forecasts plus live METAR observations should beat the market on 10k+
daily max-temperature events. Full write-up: [docs/case-study-weather.md](docs/case-study-weather.md).

**Settlement first.** Before modelling anything, I checked whether the resolved outcome can be rebuilt from raw
METARs.

| Unit / source | Events | Reproduced |
|---|---|---|
| °F / Wunderground | 2,719 | 100.0% |
| °F / NOAA timeseries | 385 | 99.7% |
| °C / Wunderground | 5,649 | 97.4% |
| °C / NOAA timeseries | 1,709 | 99.3% |
| **All clean events** | **10,462** | **98.45%** (~99.8% excluding two stations) |

**Model minus market log loss** (final version, exact forecast runs, 7 unreliable-settlement stations excluded,
date-block bootstrap 95% CI). Positive means the market was better.

| Horizon | Model - market |
|---|---|
| ≥ 48 h | +0.162 (0.147, 0.178) |
| 36-48 h | +0.229 (0.214, 0.244) |
| 24-36 h | +0.264 (0.250, 0.278) |
| Day D 00-10 local | +0.255 (0.240, 0.270) |
| Day D 10-16 local | +0.242 (0.229, 0.256) |
| Day D 16-24 local | +0.044 (0.033, 0.061) |

The model was well calibrated and far better than raw forecasts or climatology. The market was better at every
horizon.

**Backtest** (taker only, out-of-sample half: 109 days, threshold chosen on the first half):

| Cost above mid | Trades | ROI (95% CI) |
|---|---|---|
| 1¢ | 18,810 | +6.6% (+1.9%, +12.5%) |
| **3¢** | **16,444** | **-7.5% (-10.9%, -3.9%)** |
| 5¢ | 14,453 | -16.3% (-19.2%, -13.2%) |

The measured taker cost was a median 1.2-2.0¢ and a mean 1.8-3.2¢ above mid in 5-85¢ buckets, so 3¢ is realistic.

![Out-of-sample ROI, threshold x cost](docs/figures/weather-v2-roi-heatmap.png)

**The one residual, and why it's an artifact.** With the cost charged per price band at its median, the model
showed +6.8% (+2.9%, +10.8%). Nearly all of it came from resolution-day trades fired by a fresh METAR (6,115 trades,
+18.5%). Three checks killed it:

- *Observation lag.* The backtest treated a METAR as public 10 minutes after its observation time. At 20 minutes
  the edge halved (CI touching 0), and at 30 minutes it was gone.
- *Fillability.* Only 20.3% of those trades had a real print at or below our price within 5 minutes. Capped to
  that volume, the PnL was $2.2k over 109 days, about $20/day.
- *Market reaction.* The median time from observation to the first 5¢ mid move was 31.2 minutes. That points to
  publication lag, not a slow market: the "edge" was knowing the METAR before anyone could.

### 2. Cross-category calibration scan

Are Polymarket prices systematically miscalibrated anywhere, in a way you could trade? Sample: 63,416 markets,
27,144 events, 328,909 snapshots from all categories since 2025. Full write-up:
[docs/case-study-calibration-scan.md](docs/case-study-calibration-scan.md).

On discovery data (events ending before 2026-04-01), 599 cells had at least 100 events, 232 were BH-significant and
**200** had positive expected ROI after measured cost and fees. That list was frozen and hashed before the
holdout was read.

| Execution model on the holdout | Candidates passing every criterion |
|---|---|
| Mid + measured cost (not executable) | **25** |
| First real same-side print within 60 min | **1** |
| Same, and the print is at or below mid + cost (limit order) | **0** |

The mid-price pattern replicated: buying NO against cheap YES buckets in thin negRisk books looked like +34% to
+37% in some categories. At real prints it mostly disappeared (crypto 35-50¢ NO: +37% at mid, +5.5% at the print;
culture: +34% at mid, -9.7% at the print).

The single survivor was crypto favorites at 97-99¢ (buy YES): +0.65% per trade (CI +0.30%, +1.04%), about $4.6/day
of expected profit. Its holdout p ≈ 5e-4 is above the 2.5e-4 multiple-testing threshold for 200 candidates, and
about 5 false passes are expected by chance at that CI level. It doesn't count.

![Holdout ROI at mid vs at real prints](docs/figures/scan-discovery-vs-holdout.png)

## Methodology in brief

Details: [docs/methodology.md](docs/methodology.md).

1. **Reproduce settlement before modelling.** If you cannot rebuild the outcome from source data, you are modelling
   the wrong quantity.
2. **Lookahead guards everywhere.** Features only use data published before the decision time. Snapshots and
   decisions stop an hour before the earliest of end, close and resolution. Each guard has a test.
3. **Walk-forward refits.** Monthly fits trained strictly before the test month, with a gap.
4. **The market price is the baseline to beat.** Log loss against the market at the same timestamps, not against
   climatology.
5. **Discovery / holdout with frozen candidates.** Screen, correct for multiple testing, freeze with a hash, then
   test once.
6. **Execute against real prints, not the mid.** Measure the cost you would have paid. Require a print on your
   side, and report how many dollars a day actually fit.

## Polymarket API gotchas

Short list of what we verified. More in [docs/polymarket-api-gotchas.md](docs/polymarket-api-gotchas.md).

- `prices-history` for closed markets returns fine-grained points only with explicit `startTs`/`endTs`.
  `interval=max` with a small fidelity comes back empty, and windows longer than about 15 days are rejected.
- `p` in `prices-history` is undocumented. It behaves like a mid: takers paid a median 1.2-2.0¢ worse than `p` in
  5-85¢ weather buckets. Treat it as a reference, not a price you can trade at.
- Resolution rules change per market and over time: weather markets moved from Wunderground to NOAA around
  2026-08-23, stations switched (Paris LFPG to LFPB, Denver KDEN to KBKF), London changed from °F to °C, and at
  least one slug carries the wrong date. Parse every market, not every series.
- Fees are `shares × rate × p × (1 - p)`, taker only. Rates differ within a category, and fees were switched on at
  different dates per category (sports from 2025-11-29, weather from 2026-03-30). Read `feesEnabled` and
  `feeSchedule` per market.
- "By date" markets close early. 25% of the sampled markets closed more than an hour before their scheduled end,
  so anchor guards on the actual close, not on `endDate`.
- Data API taker trades (`takerOnly=true`) sum exactly to Gamma `volumeNum`, which makes them a trustworthy tape.

## Limitations

- **No historical order books.** Execution uses mids plus measured costs and real prints. Depth and queue position
  are unknown.
- **Maker strategies are untested.** The one plausible way to monetize the thin-book pattern is quoting it, and
  that needs a live order-book recorder.
- **Sample periods.** Weather markets from 2025-01-22 to 2026-09-29 with a 109-day test half; scan events since
  2025-01-01 with the split at 2026-04-01.
- **No METAR receipt timestamps.** The IEM archive has observation times only, so a genuine latency edge can
  neither be ruled out nor validated historically.
- The measured taker cost comes from markets that actually traded, so it understates the cost in thin books.
- Categories come from tag slugs. The mapping is a heuristic and easy to override.

## Data sources and attribution

No data is redistributed. The loaders fetch it from the original sources at runtime.

- [Polymarket](https://docs.polymarket.com) public Gamma, CLOB and Data APIs.
- [ECMWF open data](https://www.ecmwf.int/en/forecasts/datasets/open-data), IFS 0.25°, via the
  [AWS archive](https://registry.opendata.aws/ecmwf-forecasts/). CC BY 4.0.
- [NOAA GFS](https://registry.opendata.aws/noaa-gfs-bdp-pds/) on the AWS Open Data Registry.
- [Open-Meteo](https://open-meteo.com) for station timezones. CC BY 4.0.
- [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/request/download.phtml) ASOS/METAR archive.
- [aviationweather.gov](https://aviationweather.gov/data/api/) for station coordinates.

## Disclaimer

For research and education only. Not financial advice. This software has no trading functionality. Prediction
markets may be restricted where you live; check local rules and Polymarket's terms of service.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Reproductions, new case studies and honest negative results are welcome.

## License

[MIT](LICENSE)
