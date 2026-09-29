"""Taker-only trading simulation on a decision panel.

Panel: one row per (event_id, market_id, decision_time) with `p` (outcome-0 mid), `p_model`, `winner`, the fee
columns, `vol_window` / `traded_near` (see `execution.fills.window_volume`) and `day` (the block for daily caps and
bootstrap). An optional `cost` column (e.g. measured cost by band) overrides the flat `SimConfig.cost`.

Execution model (no historical order books exist):
- buying outcome 0 ("YES") pays `p + cost`, buying outcome 1 ("NO") pays `1 - p + cost`;
- taker fee from each market's own schedule (`fee_everywhere=True` applies it even where disabled);
- no fill unless the market printed within the fill window, price within [min_price, max_price];
- size capped at `vol_frac` of the window's taker volume, per-market and per-day stake caps;
- one entry per (market, side): the first decision whose edge clears the threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import polars as pl

from pmrk.execution.fees import taker_fee_per_share
from pmrk.stats.bootstrap import ratio_ci, sum_ci


@dataclass(frozen=True)
class SimConfig:
    threshold: float = 0.05
    cost: float = 0.03
    vol_frac: float = 0.10
    sizing: Literal["flat", "kelly"] = "flat"
    flat_stake: float = 20.0
    kelly_frac: float = 0.15
    bankroll: float = 10_000.0
    max_per_market: float = 200.0
    max_per_day: float = 2_000.0
    fee_everywhere: bool = False
    min_price: float = 0.01
    max_price: float = 0.99


def candidate_orders(panel: pl.DataFrame, cfg: SimConfig) -> pl.DataFrame:
    """Both sides of every row with execution price, fee and expected edge per share."""
    cost = pl.col("cost") if "cost" in panel.columns else pl.lit(cfg.cost)
    yes = panel.with_columns(side=pl.lit("YES"), p_win=pl.col("p_model"), price=pl.col("p") + cost)
    no = panel.with_columns(side=pl.lit("NO"), p_win=1.0 - pl.col("p_model"), price=1.0 - pl.col("p") + cost)
    c = pl.concat([yes, no]).filter(pl.col("price").is_between(cfg.min_price, cfg.max_price))
    enabled = pl.lit(True) if cfg.fee_everywhere else None
    c = c.with_columns(fee=taker_fee_per_share(pl.col("price"), enabled=enabled))
    return c.with_columns(
        cost_per_share=pl.col("price") + pl.col("fee"), edge=pl.col("p_win") - pl.col("price") - pl.col("fee")
    )


def _stake(cfg: SimConfig) -> pl.Expr:
    if cfg.sizing == "flat":
        return pl.lit(cfg.flat_stake)
    kelly = (pl.col("edge") / (1.0 - pl.col("cost_per_share"))).clip(0.0, 1.0)
    return (kelly * cfg.kelly_frac * cfg.bankroll).clip(0.0, cfg.max_per_market)


def simulate(panel: pl.DataFrame, cfg: SimConfig) -> pl.DataFrame:
    """Trade log: one row per entered (market, side) with shares, spend, won, pnl."""
    c = candidate_orders(panel, cfg).filter((pl.col("edge") > cfg.threshold) & pl.col("traded_near"))
    first = c.sort("decision_time").group_by("market_id", "side", maintain_order=True).first()
    first = first.with_columns(stake=_stake(cfg)).with_columns(
        shares=pl.min_horizontal(pl.col("stake") / pl.col("cost_per_share"), cfg.vol_frac * pl.col("vol_window"))
    )
    first = (
        first.filter(pl.col("shares") > 0.5)
        .sort("decision_time")
        .with_columns(spend=pl.col("shares") * pl.col("cost_per_share"))
    )
    first = first.with_columns(cum_day=pl.col("spend").cum_sum().over("day")).filter(
        pl.col("cum_day") <= cfg.max_per_day
    )
    won = pl.when(pl.col("side") == "YES").then(pl.col("winner") == 0).otherwise(pl.col("winner") == 1)
    return first.with_columns(won=won).with_columns(
        pnl=pl.col("shares") * (pl.col("won").cast(pl.Float64) - pl.col("cost_per_share"))
    )


def metrics(trades: pl.DataFrame) -> dict[str, float]:
    if trades.is_empty():
        nan = float("nan")
        return {
            "n_trades": 0,
            "pnl": 0.0,
            "turnover": 0.0,
            "roi": nan,
            "hit_rate": nan,
            "max_dd": 0.0,
            "avg_edge": nan,
            "realized_edge": nan,
            "n_days": 0,
        }
    daily = trades.group_by("day").agg(pl.col("pnl").sum()).sort("day")
    equity = daily["pnl"].cum_sum().to_numpy()
    peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:]
    turnover = float(trades["spend"].sum())
    return {
        "n_trades": trades.height,
        "pnl": float(trades["pnl"].sum()),
        "turnover": turnover,
        "roi": float(trades["pnl"].sum()) / turnover if turnover else float("nan"),
        "hit_rate": float(trades["won"].mean()),
        "max_dd": float(np.max(peak - equity)),
        "avg_edge": float(trades["edge"].mean()),
        "realized_edge": float((trades["pnl"] / trades["shares"]).mean()),
        "n_days": daily.height,
    }


def bootstrap(trades: pl.DataFrame, block: str = "day") -> dict[str, tuple[float, float]]:
    """95% CIs for total PnL and ROI, resampling whole blocks (dates by default)."""
    if trades.is_empty():
        nan = float("nan")
        return {"pnl": (nan, nan), "roi": (nan, nan)}
    _, plo, phi = sum_ci(trades, "pnl", block)
    _, rlo, rhi = ratio_ci(trades, "pnl", "spend", block)
    return {"pnl": (plo, phi), "roi": (rlo, rhi)}


def without_top(trades: pl.DataFrame, k: int = 5, block: str = "day") -> dict[str, float]:
    """PnL and ROI after removing the k most profitable blocks (concentration check)."""
    g = trades.group_by(block).agg(pl.col("pnl").sum(), pl.col("spend").sum()).sort("pnl", descending=True).slice(k)
    spend = float(g["spend"].sum())
    return {
        f"pnl_without_top{k}": float(g["pnl"].sum()),
        f"roi_without_top{k}": float(g["pnl"].sum()) / spend if spend else float("nan"),
    }


def sweep(panel: pl.DataFrame, base: SimConfig, thresholds: list[float], costs: list[float]) -> pl.DataFrame:
    rows = [
        {"threshold": t, "cost": c, **metrics(simulate(panel, replace(base, threshold=t, cost=c)))}
        for t in thresholds
        for c in costs
    ]
    return pl.DataFrame(rows)


def breakdown(trades: pl.DataFrame, by: str) -> pl.DataFrame:
    return (
        trades.group_by(by)
        .agg(
            n=pl.len(),
            pnl=pl.col("pnl").sum(),
            spend=pl.col("spend").sum(),
            hit=pl.col("won").mean(),
            avg_edge=pl.col("edge").mean(),
        )
        .with_columns(roi=pl.col("pnl") / pl.col("spend"))
        .sort("pnl", descending=True)
    )
