# 04. Weather case study

Rebuilds the daily highest-temperature study end to end. Needs the extra: `uv sync --extra weather`.
Results and discussion: [docs/case-study-weather.md](../docs/case-study-weather.md).

```bash
uv run pmrk weather markets       # all <city>-daily-weather series; parses station, source, unit, buckets per event
uv run pmrk weather stations      # coordinates (aviationweather.gov), timezones (Open-Meteo)
uv run pmrk weather prices        # 1-minute price history of every settleable closed event
uv run pmrk weather trades        # taker trades
uv run pmrk weather metar         # IEM METAR/SPECI archive, decoded; IEM throttles, so this is slow
uv run pmrk weather settlement    # reproduction rate by unit and source, worst stations
uv run pmrk weather forecasts     # exact ECMWF IFS 0.25 / GFS 0.5 runs from AWS, 2 m temperature only
uv run pmrk weather fit           # walk-forward monthly EMOS fits
uv run pmrk weather eval          # model / climatology / raw ECMWF vs market log loss by horizon
uv run pmrk costs --category weather
uv run pmrk weather backtest --band-cost median                   # triggers: hourly, forecast runs, METARs
uv run pmrk weather backtest --band-cost median --obs-lag 20m     # the residual check
```

The weather package is also the reference plugin. It uses only the public core:

- `WeatherModel` (`pmrk.weather.model`) implements `ProbabilityModel` and its `decision_times` hook;
- `MetarSettlement` (`pmrk.weather.settlement`) implements `SettlementSource`;
- everything temperature-specific (stations, METAR decoding, units, bucket edges, NWP, EMOS) stays in
  `pmrk.weather`.
