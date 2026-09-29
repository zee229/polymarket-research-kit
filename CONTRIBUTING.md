# Contributing

Bug reports, reproductions of the studies, and new case studies are welcome.

## Setup

```bash
uv sync                    # core
uv sync --extra weather    # + weather case study (ecCodes)
uv run pytest -q           # offline tests
uv run pytest -q -m network  # live API smoke tests (optional)
uv run ruff check . && uv run ruff format --check .
```

## Ground rules

- **Research only.** No order placement, wallets, keys or anything that trades. PRs adding them will be closed.
- **No data in git.** Tests use small synthetic fixtures built in code; loaders fetch real data at runtime.
- **The core stays domain-agnostic.** Anything specific to one kind of market (units, stations, team names,
  tickers) goes in its own subpackage and talks to the core only through `pmrk.interfaces` and the public API.
  If a plugin needs a hack in the core, change the core design instead.
- **Every new time-dependent feature needs a lookahead test**: show that information published after the decision
  time cannot reach it (see `tests/test_weather_lookahead.py`).
- Tests must run offline and fast. Mark anything that hits the network with `@pytest.mark.network`.

## Adding a model or a case study

Implement `ProbabilityModel` (and `SettlementSource` if outcomes can be reproduced from external data), see
`docs/plugins.md` and `examples/models/`. A negative result with honest execution assumptions is a perfectly good
contribution.
