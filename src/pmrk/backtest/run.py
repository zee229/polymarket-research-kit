"""Event-driven backtest: decision times -> guarded market rows -> mids -> model -> fills -> PnL.

Protocol: events are split by date into a tuning half and a test half. The edge threshold is chosen on the tuning
half at the base cost; every reported number comes from the test half.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from pmrk.backtest.engine import SimConfig, bootstrap, breakdown, metrics, simulate, sweep, without_top
from pmrk.execution.fills import window_volume
from pmrk.interfaces import OUTCOME_COLUMNS, ProbabilityModel
from pmrk.polymarket.store import load_prices, load_trades
from pmrk.snapshots.horizons import LOOKAHEAD_BUFFER, decision_grid, market_rows
from pmrk.snapshots.prices import MAX_STALE, normalize_event, price_at
from pmrk.stats.scores import EPS

log = logging.getLogger(__name__)

THRESHOLDS = [0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
COSTS = [0.01, 0.02, 0.03, 0.05]
REPORT_COSTS = [0.01, 0.03, 0.05]
MIN_TRAIN_TRADES = 100
FILL_WINDOW = timedelta(minutes=60)
META = [
    "event_id",
    "market_id",
    "event_slug",
    "series_slug",
    "category",
    "neg_risk",
    "question",
    "group_item_title",
    "outcome0",
    "outcome1",
    "end_ts",
    "fees_enabled",
    "fee_rate",
    "fee_exponent",
    "winner",
]


def decision_times(markets: pl.DataFrame, model: ProbabilityModel, every: timedelta, buffer: timedelta) -> pl.DataFrame:
    """Regular grid plus the model's own triggers (if it defines `decision_times`), de-duplicated."""
    grid = decision_grid(markets, every, buffer).with_columns(trigger=pl.lit("grid"))
    hook = getattr(model, "decision_times", None)
    if hook is None:
        return grid
    extra = hook(markets.drop([c for c in OUTCOME_COLUMNS if c in markets.columns]))
    if "trigger" not in extra.columns:
        extra = extra.with_columns(trigger=pl.lit("model"))
    extra = extra.select("event_id", pl.col("decision_time").dt.cast_time_unit("us"), "trigger")
    return pl.concat([grid, extra]).unique(["event_id", "decision_time"], keep="first")


def _normalize_model(rows: pl.DataFrame) -> pl.DataFrame:
    """negRisk: floor model probabilities at EPS and renormalize within each (event, decision)."""
    clipped = pl.col("p_model").clip(EPS, 1.0)
    norm = clipped / clipped.sum().over("event_id", "decision_time")
    return rows.with_columns(p_model=pl.when(pl.col("neg_risk")).then(norm).otherwise(pl.col("p_model")))


def build_batch(
    markets: pl.DataFrame,
    model: ProbabilityModel,
    every: timedelta = timedelta(hours=1),
    buffer: timedelta = LOOKAHEAD_BUFFER,
    max_stale: timedelta = MAX_STALE,
    fill_window: timedelta = FILL_WINDOW,
) -> pl.DataFrame:
    """Panel rows for one batch of events (all markets of each event must be in `markets`)."""
    events = decision_times(markets, model, every, buffer)
    rows = market_rows(events, markets, buffer=buffer).drop("created_ts", "known_ts")
    rows = rows.join(markets.select(META).drop("event_id"), on="market_id")
    ids = markets["event_id"].unique().to_list()
    rows = normalize_event(price_at(rows, load_prices(ids), max_stale=max_stale))
    if rows.is_empty():
        return rows
    preds = model.predict(rows.drop([c for c in OUTCOME_COLUMNS if c in rows.columns]))
    keys = ["event_id", "market_id", "decision_time"]
    extra = [c for c in preds.columns if c not in rows.columns or c in keys]
    n_before = pl.len().over("event_id", "decision_time")
    rows = rows.with_columns(_n=n_before).join(preds.select(extra).drop_nulls("p_model"), on=keys, how="inner")
    # A negRisk distribution is only comparable if the model priced every live market of the event.
    rows = rows.filter(~pl.col("neg_risk") | (n_before == pl.col("_n"))).drop("_n")
    rows = _normalize_model(rows)
    if "day" not in rows.columns:
        rows = rows.with_columns(day=pl.col("decision_time").dt.date())
    return window_volume(rows, load_trades(ids), fill_window)


def build_panel(
    markets: pl.DataFrame, model: ProbabilityModel, batch_events: int = 500, out: Path | None = None, **kwargs: Any
) -> pl.DataFrame:
    """Panel for all clean markets, built in event batches (price history can be large)."""
    markets = markets.filter(pl.col("clean").all().over("event_id"))
    event_ids = sorted(markets["event_id"].unique().to_list())
    parts = []
    for i in range(0, len(event_ids), batch_events):
        batch = markets.filter(pl.col("event_id").is_in(event_ids[i : i + batch_events]))
        part = build_batch(batch, model, **kwargs)
        log.info("panel batch %d: %d events -> %d rows", i // batch_events, batch["event_id"].n_unique(), part.height)
        if part.height:
            parts.append(part)
    panel = pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        panel.write_parquet(out)
    return panel


def split_date(panel: pl.DataFrame) -> date:
    """Median event date: the first half of events tunes, the second half tests."""
    ev = panel.group_by("event_id").agg(d=pl.col("day").max()).sort("d")["d"]
    return ev[len(ev) // 2]


def choose_threshold(train: pl.DataFrame, base: SimConfig, thresholds: list[float]) -> tuple[float, pl.DataFrame]:
    """Best tuning-half ROI among thresholds with enough trades; falls back to `base.threshold` if none trade."""
    grid = sweep(train, base, thresholds, [base.cost])
    ok = grid.filter(pl.col("n_trades") >= MIN_TRAIN_TRADES)
    if ok.is_empty():
        ok = grid.filter(pl.col("n_trades") > 0)
    if ok.is_empty():
        return base.threshold, grid
    return float(ok.sort("roi", descending=True)["threshold"][0]), grid


def run(
    panel: pl.DataFrame,
    base: SimConfig | None = None,
    cut: date | None = None,
    thresholds: list[float] | None = None,
    breakdowns: tuple[str, ...] = ("trigger", "category", "side"),
) -> tuple[dict[str, Any], pl.DataFrame]:
    """Tune on the first half, report on the second. Returns (results, main-config test trades)."""
    base = base or SimConfig()
    thresholds = thresholds or THRESHOLDS
    cut = cut or split_date(panel)
    ev_day = pl.col("day").max().over("event_id")
    train, test = panel.filter(ev_day < cut), panel.filter(ev_day >= cut)
    thr, train_grid = choose_threshold(train, base, thresholds)
    res: dict[str, Any] = {
        "split_date": str(cut),
        "chosen_threshold": thr,
        "base_config": asdict(base),
        "train_events": train["event_id"].n_unique(),
        "test_events": test["event_id"].n_unique(),
        "train_grid": train_grid.to_dicts(),
    }
    for sizing in ("flat", "kelly"):
        for cost in REPORT_COSTS:
            tr = simulate(test, replace(base, threshold=thr, cost=cost, sizing=sizing))
            res[f"test_{sizing}_cost{cost}"] = {**metrics(tr), "ci": bootstrap(tr), **without_top(tr)}
    main_cfg = replace(base, threshold=thr)
    main = simulate(test, main_cfg)
    res["test_main"] = {**metrics(main), "ci": bootstrap(main), **without_top(main)}
    res["test_main_fee_everywhere"] = metrics(simulate(test, replace(main_cfg, fee_everywhere=True)))
    for by in breakdowns:
        if by in main.columns:
            res[f"by_{by}"] = breakdown(main, by).to_dicts()
    res["test_grid"] = sweep(test, base, thresholds, COSTS).to_dicts()
    return res, main


def plot_heatmap(grid: list[dict[str, Any]], path: Path, value: str = "roi") -> None:
    """Threshold x cost heatmap of a sweep (needs the [plots] extra)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    g = pl.DataFrame(grid).pivot(on="cost", index="threshold", values=value).sort("threshold")
    mat = g.drop("threshold").to_numpy().astype(float)
    vmax = float(np.nanmax(np.abs(mat))) if np.isfinite(mat).any() else 1.0
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto", origin="lower")
    fig.colorbar(im, label=value)
    ax.set_xticks(range(mat.shape[1]), [c for c in g.columns if c != "threshold"])
    ax.set_yticks(range(mat.shape[0]), g["threshold"].to_list())
    ax.set_xlabel("cost above mid (USD/share)")
    ax.set_ylabel("edge threshold")
    for i, j in np.ndindex(mat.shape):
        if np.isfinite(mat[i, j]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title(f"Out-of-sample {value}: threshold x cost")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
