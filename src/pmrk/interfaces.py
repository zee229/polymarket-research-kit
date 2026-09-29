"""Public plugin protocols. Domain code (e.g. `pmrk.weather`) plugs into the core only through these.

Both protocols are batch-shaped (polars in, polars out): a backtest asks for millions of (market, time)
probabilities, so one call per market and time would be far too slow.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Protocol, runtime_checkable

import polars as pl

# Columns a model may never see: they encode the resolution.
OUTCOME_COLUMNS = (
    "winner",
    "outcome_prices",
    "closed",
    "closed_ts",
    "uma_end_ts",
    "known_ts",
    "disputed",
    "clean",
    "resolved_early_1h",
)


@runtime_checkable
class ProbabilityModel(Protocol):
    """Probability that outcome 0 of a market wins, using only information available at the decision time.

    `predict` receives one row per (event_id, market_id, decision_time) with market metadata (question, outcome
    labels, category, end_ts, ...) and `p`, the market's own outcome-0 mid at the decision time. Resolution columns
    are stripped before the call. It returns event_id, market_id, decision_time and `p_model` in [0, 1]; rows it
    cannot price may be omitted, and extra columns are carried through to the trade log (useful for breakdowns).

    Optionally, `decision_times(markets)` returns extra (event_id, decision_time[, trigger]) rows, e.g. "one
    minute after a new forecast run was published". They are added to the default hourly grid and pass through the
    same lookahead guards.
    """

    name: str

    def predict(self, queries: pl.DataFrame) -> pl.DataFrame: ...


@runtime_checkable
class SettlementSource(Protocol):
    """Reproduce market outcomes from external data (for domains where that is possible).

    `reproduce(markets)` returns market_id and `winner_reproduced` (0/1, null if the data is missing). Reproducing
    settlement before modelling is the cheapest way to find out whether you are modelling the right quantity.
    """

    name: str

    def reproduce(self, markets: pl.DataFrame) -> pl.DataFrame: ...


def settlement_agreement(markets: pl.DataFrame, source: SettlementSource, by: list[str] | None = None) -> pl.DataFrame:
    """Share of clean resolved markets whose outcome the source reproduces (per optional group)."""
    rep = source.reproduce(markets.drop([c for c in OUTCOME_COLUMNS if c in markets.columns]))
    j = markets.filter(pl.col("clean")).join(rep, on="market_id", how="inner").drop_nulls("winner_reproduced")
    return j.group_by(by or pl.lit("all").alias("group")).agg(
        markets=pl.len(),
        events=pl.col("event_id").n_unique(),
        reproduced=(pl.col("winner") == pl.col("winner_reproduced")).mean(),
    )


def load_model(spec: str) -> ProbabilityModel:
    """Load `module:attr` or `path/to/file.py:attr`; a class is instantiated without arguments."""
    target, _, attr = spec.partition(":")
    if not attr:
        raise ValueError(f"model spec {spec!r} must look like 'package.module:Name' or 'file.py:Name'")
    if target.endswith(".py"):
        path = Path(target).resolve()
        mod_spec = importlib.util.spec_from_file_location(path.stem, path)
        if mod_spec is None or mod_spec.loader is None:
            raise ImportError(f"cannot load {path}")
        module = importlib.util.module_from_spec(mod_spec)
        sys.modules[path.stem] = module
        mod_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    obj = getattr(module, attr)
    model = obj() if isinstance(obj, type) else obj
    if not isinstance(model, ProbabilityModel):
        raise TypeError(f"{spec} does not implement ProbabilityModel (needs `name` and `predict`)")
    return model
