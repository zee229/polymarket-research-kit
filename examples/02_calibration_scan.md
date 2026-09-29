# 02. Calibration scan

Question: are prices in some category, price band or horizon systematically off, in a way you could have traded?
The protocol is in [docs/methodology.md](../docs/methodology.md); the full study is in
[docs/case-study-calibration-scan.md](../docs/case-study-calibration-scan.md).

```bash
# 1. a broad universe of closed markets
uv run pmrk fetch markets --start 2025-01-01 --end 2026-10-01

# 2. guarded snapshots for a stratified sample (targeted price windows, no full history)
uv run pmrk snapshot --sample 3000 --per-event 3

# 3. measured taker cost by category and band, from a subset with trades and 1-minute mids
uv run pmrk fetch trades --category sports crypto politics culture
uv run pmrk fetch prices --category sports crypto politics culture --lookback 5d
uv run pmrk costs --by category

# 4. discovery only: test cells, apply BH, freeze candidates with a hash
uv run pmrk calibration discover --split 2026-04-01

# 5. holdout: frozen cells only, at mid vs at real prints
uv run pmrk calibration holdout --split 2026-04-01 --fetch-trades
```

Outputs in `data/reports/calibration/`:

- `discovery_cells.csv`: every tested cell with `diff = frequency - price`, clustered CI, p-value, BH flag and
  expected ROI after cost and fees;
- `candidates.json` / `candidates.sha256` / `candidates.md`: the frozen list. Editing the JSON makes the holdout step
  refuse to run;
- `holdout_results.csv`: per candidate, ROI with event- and date-block CIs for three execution models (`mid`,
  `print`, `print_lim`), fill rate, fillable USD per day and the pass flag.

Read the `mid` vs `print` columns together. A large `mid_roi` that collapses at `print_roi` is a spread artifact.
