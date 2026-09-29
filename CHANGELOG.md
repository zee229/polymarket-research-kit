# Changelog

## v0.1.0

First public release.

- Core Polymarket loaders: Gamma events/markets (by id, slug, series, tag, end-date window), CLOB price history
  (chunked, resumable), Data API taker trades, per-market fee schedules, on-disk cache.
- Snapshots at fixed horizons and decision grids with lookahead guards (earliest of end / close / resolution,
  minus a buffer), bounded price staleness, negRisk normalization and overround.
- Execution realism: measured taker cost vs mid by price band, fills against real prints, window-volume caps,
  Polymarket taker fee formula.
- Statistics: log loss, Brier, CRPS, cluster-robust calibration cells, block bootstrap, Benjamini-Hochberg.
- Calibration scanner with discovery/holdout split and tamper-evident frozen candidates.
- Generic taker backtester driven by any `ProbabilityModel`, with two example models.
- Weather case study (`[weather]` extra): METAR settlement reproduction, exact ECMWF/GFS runs from AWS, EMOS.
