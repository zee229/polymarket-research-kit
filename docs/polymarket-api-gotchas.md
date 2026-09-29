# Polymarket API gotchas

What we ran into while pulling a few million markets from the public APIs. Everything here was verified against
live responses during the two studies (September 2026). APIs change, so re-check anything you depend on.

## Gamma (events and markets)

- **Keyset pagination and long ranges.** A single `events/keyset` or `markets/keyset` cursor over a long end-date
  range hit a persistent 403. Splitting the crawl into monthly `end_date` windows fixed it.
  `gamma.iter_events` does this for you.
- **Series vs tags.** Tags are added later than markets exist. The weather tag `daily-temperature` only exists from
  2025-12-28, so tag-based discovery misses 2025 markets; enumerating the recurring series (`<city>-daily-weather`)
  does not.
- **Everything is a binary.** Multi-outcome events are groups of Yes/No markets (`negRisk`). Outcome labels are not
  always Yes/No ("Over/Under", "Up/Down", team names), so the toolkit keys everything on outcome index.
- **Clean resolution.** `outcomePrices` is `["1","0"]` or `["0","1"]` for a normal resolution. 50-50 and voided
  markets show other values (56,501 of 3,577,448 markets in the scan were not exactly 1/0). Disputes show up in
  `umaResolutionStatuses`; admin resolutions are only partly detectable.
- **`arch-` copies.** Some events were re-created with an `arch-` slug prefix. In May 2026 a batch of weather
  events was resolved administratively and did not match the observation data (the `arch-` events of
  2026-05-17..19 reproduce at 81%). The toolkit marks them as not clean.
- **Three timestamps.** `endDate` is the scheduled end, `closedTime` is when trading stopped and `umaEndDate` is
  when the oracle round ended. They differ: 25% of the sampled markets closed more than an hour before the scheduled
  end. Some close after it (an announcement after the nominal end time). Guards use the earliest of the three.
- **Slugs lie.** At least one weather slug carries the wrong date. Parse dates from the rules text or use the
  structured fields.
- **Resolution rules drift.** For daily temperature markets: the source moved from Wunderground to NOAA around
  2026-08-23, stations switched (Paris LFPG to LFPB from 2026-04-19, Denver KDEN to KBKF from 2026-03-29, Taipei
  RCTP to RCSS from 2026-04-05) and London changed from degF to degC on 2025-12-10. Read the rules per market.

## CLOB `/prices-history`

- **Closed markets need explicit windows.** With explicit `startTs`/`endTs` and `fidelity=1` we got 1-minute
  points for markets resolved as far back as January 2025. With `interval=max` and a small fidelity the response
  is empty. This contradicts a widely cited claim that closed markets return nothing below 12 h fidelity.
- **Window size.** Windows longer than about 15 days are rejected at any fidelity. The weather study used 5-day
  chunks at 1-minute fidelity; `clob.price_history` chunks automatically.
- **What is `p`?** Undocumented. It behaves like a mid or last-trade proxy, not an executable price: takers paid a
  median 1.2-2.0¢ worse than `p` in 5-85¢ weather buckets. The scan showed what happens if you trust it: patterns
  that are real in `p` vanish at real prints.
- **Size.** One-minute history adds up fast. Store it per event and load it in event batches.

## Data API `/trades`

- `takerOnly=true` returns the taker side of every fill. Summed `size` equals Gamma `volumeNum` exactly, so this is
  a reliable tape for volume, costs and "did anyone actually trade at this price".
- Pagination stops at a maximum offset. Very active markets get truncated; the loader flags them.
- Trades come with `side` and `outcomeIndex`. A taker selling outcome 1 is economically buying outcome 0; the
  toolkit converts every trade to outcome-0 terms (`px0`, `buys0`).

## Fees

- Formula (docs.polymarket.com/trading/fees): `fee = shares × feeRate × p × (1 - p)`, taker only. Makers pay nothing
  and receive a rebate.
- Each market has its own `feesEnabled` and `feeSchedule`. Rates vary within a category (sports: 0, 0.0175, 0.03,
  0.04, 0.05; crypto up to 0.07, and 0.25 for some short-horizon crypto markets).
- Fees started on different dates per category (first fee-enabled end date): sports 2025-11-29; crypto, finance and
  culture 2025-12-31; crypto up/down 2026-01-07; esports 2026-01-20; politics 2026-02-28; weather 2026-03-30;
  mentions 2026-04-01; economics 2026-04-03. Most geopolitics markets are fee-free.

## Rate limits

All loaders are rate-limited, retry 429/5xx with exponential backoff, and resume from what is already on disk.
Keyset pages sometimes return 403 under load and succeed on retry with the same cursor.

## Non-Polymarket sources used by the weather case study

- ECMWF open data on AWS moved the 06/18Z runs from stream `scda` to `oper` on 2026-05-12. Sustained downloads get
  503 SlowDown; rerun with fewer workers.
- Iowa Environmental Mesonet throttles hard ("Too many requests"). Its `valid` field is the observation time, and
  there are no receipt times.
- Open-Meteo's Historical Forecast API stitches the first hours of successive runs. It is fine for climatology but
  leaks the future into any decision-time feature. The Previous Runs API only exposes leads at 24 h offsets.
