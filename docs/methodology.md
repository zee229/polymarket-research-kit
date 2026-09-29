# Methodology

How the toolkit tries to keep you from finding edges that are not there. Every rule below comes from something that
went wrong, or nearly went wrong, in one of the two case studies.

## 1. Reproduce settlement before you model

Know exactly which number the market resolves on, then rebuild it from source data. In the weather study this
turned "daily max temperature" into a precise rule: local-day max over all METAR reports including SPECIs, degF
markets from the tenths-degC T-group rounded half-up, degC markets from the integer group. With that rule, 98.45%
of 10,462 clean events reproduced. The misses were informative: two stations where the archive cannot match the
settlement page, one station that must use routine reports only, and a batch of administratively resolved events.

If your domain allows it, implement a `SettlementSource` and run `pmrk.interfaces.settlement_agreement` before
anything else. If you cannot reproduce outcomes, say so and treat the market's own resolution as ground truth.

## 2. Lookahead guards

Three kinds of lookahead, and the guard for each:

| Leak | Guard | Where |
|---|---|---|
| A feature uses data published after the decision time | Every input carries an availability time (forecast `init + delay`, METAR `obs_time + lag`, trailing statistics with a gap) and is filtered by `available <= decision_time` | model code, e.g. `pmrk.weather.features` |
| A price or decision after the outcome was known | No snapshot or decision at or after `known_ts - buffer`, where `known_ts = min(end, close, oracle resolution)` and the buffer defaults to 1 h | `pmrk.snapshots.horizons` |
| The model sees the answer | Resolution columns (`winner`, `outcome_prices`, close and resolution times, ...) are dropped before `predict` is called | `pmrk.backtest.run` |

The early-close guard matters more than it looks. 25% of the markets sampled in the scan closed more than an hour
before their scheduled end. Anchoring on `endDate` alone would put many snapshots after the outcome was public.

Each guard has a test that feeds in data published one minute too late and checks that it is ignored.

A caveat about the guard itself: excluding the final hour before an early close uses the close time, which is not
known in advance. That removes trades rather than adding them, so it can only make a strategy look worse. It also
means early resolvers are a selected subsample: in the scan they resolved YES more often than priced, and that
difference is not tradeable because you do not know in advance which markets will close early.

## 3. Walk-forward

Models are refit on a fixed schedule (monthly in the weather study), each fit using only data that was fully
observed before the first decision it will be used for. Trailing statistics (station bias, climatologies) respect
the same cutoff. Hyperparameters, thresholds and station filters are chosen on the first half of the sample and
frozen for the second half.

## 4. The market is the baseline

Beating climatology or a raw forecast is easy and says nothing about money. The primary test is log loss against
the market's own normalized mid at the same timestamps, with a block bootstrap by date. In the weather study the
model's CRPS was far better than raw forecasts and climatology and its 90% intervals covered about 90%, and it
still lost to the market at every horizon.

negRisk events need care: the mids of all buckets usually sum to a bit more than 1 (median 1.007-1.019 in the
scan), so the market distribution is normalized before scoring. Model probabilities are floored and renormalized
the same way.

## 5. Discovery, holdout and frozen candidates

A screen over hundreds of cells will always find something. The scan protocol:

1. Split events by end date. Discovery comes first in time, holdout second.
2. Test every cell on discovery only, with standard errors clustered by event.
3. Apply Benjamini-Hochberg and require positive expected ROI after measured cost and fees.
4. Freeze the survivors to `candidates.json` with a sha256 fingerprint. The holdout step refuses to run if the
   file is missing or has been edited.
5. Test each frozen candidate once on the holdout. Report how many pass, and compare the p-values of the passes
   with a Bonferroni threshold for the number of candidates tested.

## 6. Execution realism

No historical order books exist for Polymarket, so execution is modelled from what can be measured:

- **Measured cost.** For every taker trade, the price paid minus the prevailing 1-minute mid, by price band. In the
  weather study the median was 1.2-2.0¢ and the mean 1.8-3.2¢ in 5-85¢ buckets. Run flat costs of 1, 3 and 5¢
  and the measured band costs (median, mean, p75) side by side.
- **Real prints.** A backtest fill requires that the market actually traded near the decision time, and size is
  capped at a fraction of that window's volume. The strictest check requires a real same-side print at or below
  your price within minutes.
- **Fees** from each market's own schedule, and a sensitivity run with fees applied everywhere.
- **Capacity.** Report fillable dollars per day. An edge worth $5/day is a curiosity, not a strategy.
- **Concentration.** Report ROI without the top 5 days or events, and the share of PnL from the top series.

## 7. Kill criteria, agreed before looking

The weather study wrote its go / no-go criteria down before the backtest. Out-of-sample log loss had to beat the
market. PnL had to stay positive after fees and a 3¢ cost, with a bootstrap CI above zero. The result had to
survive removing the top 5 days. It failed the first criterion, and the rest of the backtest was run only to
quantify the failure.

## Open questions

Things these studies could not test. They are open questions, not promises:

- **A live order-book recorder.** Spreads, depth and reaction latency can only be measured forward, and every day
  not recorded is lost.
- **Maker execution.** Quoting the cheap side of thin negRisk books is the only way the mid-price pattern could be
  monetized. Adverse selection around news can only be measured with recorded books.
- **Latency.** Whether anyone can act on an observation before the settlement source and the market see it needs
  receipt timestamps, which historical archives do not have.
