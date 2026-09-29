"""Measured taker cost: what takers actually paid relative to the prevailing mid, by price band.

`cost = paid - mid` for a taker who gained outcome-0 exposure, `mid - received` for one who shed it (both in
outcome-0 terms, positive = worse than mid). Measured on real prints, so it reflects the half-spread plus slippage
that small takers actually paid. Caveat: prints happen mostly where books are liquid, so this understates the cost
in thin markets.
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl

BAND_EDGES = [0.01, 0.03, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95, 0.97, 0.99]


def band_labels(edges: list[float]) -> list[str]:
    pts = [0.0, *edges, 1.0]
    return [f"{round(a * 100):g}-{round(b * 100):g}" for a, b in zip(pts[:-1], pts[1:], strict=True)]


def band(col: str = "p", edges: list[float] = BAND_EDGES) -> pl.Expr:
    """Price band label, left-closed: [a, b)."""
    return pl.col(col).cut(edges, labels=band_labels(edges), left_closed=True).cast(pl.Utf8)


def trade_costs(trades: pl.DataFrame, mids: pl.DataFrame, tolerance: timedelta = timedelta(minutes=5)) -> pl.DataFrame:
    """Per trade: prevailing mid (last point at or before the trade, within tolerance) and signed cost."""
    px = mids.select("market_id", "ts", pl.col("p").alias("mid")).sort("ts")
    j = trades.sort("ts").join_asof(
        px, on="ts", by="market_id", strategy="backward", check_sortedness=False, tolerance=tolerance
    )
    return j.drop_nulls("mid").with_columns(
        cost=pl.when(pl.col("buys0")).then(pl.col("px0") - pl.col("mid")).otherwise(pl.col("mid") - pl.col("px0"))
    )


def cost_table(costs: pl.DataFrame, by: list[str] | None = None, edges: list[float] = BAND_EDGES) -> pl.DataFrame:
    """Median / mean / p75 taker cost by mid band (and optional extra keys such as category)."""
    keys = [*(by or []), "band"]
    return (
        costs.with_columns(band=band("mid", edges))
        .group_by(keys)
        .agg(
            n=pl.len(),
            median_cost=pl.col("cost").median(),
            mean_cost=pl.col("cost").mean(),
            p75_cost=pl.col("cost").quantile(0.75),
            shares=pl.col("size").sum(),
        )
        .sort([*(by or []), pl.col("band").str.split("-").list.first().cast(pl.Float64)])
    )


def attach_cost(
    rows: pl.DataFrame,
    table: pl.DataFrame,
    stat: str = "median_cost",
    price_col: str = "p",
    by: list[str] | None = None,
    default: float = 0.02,
    edges: list[float] = BAND_EDGES,
) -> pl.DataFrame:
    """Add a `cost` column from a cost table: keyed cell, else the band's pooled median, else `default`."""
    keys = [*(by or []), "band"]
    pooled = table.group_by("band").agg(_pooled=pl.col(stat).median())
    out = rows.with_columns(band=band(price_col, edges)).join(
        table.select(*keys, pl.col(stat).alias("_cell")), on=keys, how="left"
    )
    out = out.join(pooled, on="band", how="left")
    return out.with_columns(cost=pl.coalesce("_cell", "_pooled", pl.lit(default)).clip(0.0, None)).drop(
        "_cell", "_pooled"
    )
