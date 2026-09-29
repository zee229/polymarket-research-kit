"""Fill realism from the historical tape (no historical order books exist).

- `window_volume`: taker shares printed within +-window of a decision (feasibility flag and size cap);
- `first_print`: the first real same-side taker print after a decision, i.e. what someone actually paid;
- `reaction_minutes`: how long the mid took to move toward the final outcome after an external event.
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl


def window_volume(
    rows: pl.DataFrame, trades: pl.DataFrame, window: timedelta = timedelta(minutes=60), time_col: str = "decision_time"
) -> pl.DataFrame:
    """Add `vol_window` (taker shares in [t - window, t + window]) and `traded_near` (> 0)."""
    tr = trades.sort("ts").with_columns(cum=pl.col("size").cum_sum().over("market_id")).select("market_id", "ts", "cum")
    keys = (
        rows.select("market_id", time_col)
        .unique()
        .with_columns(_lo=pl.col(time_col) - window, _hi=pl.col(time_col) + window)
    )
    a = keys.sort("_lo").join_asof(
        tr.rename({"ts": "_lo_ts", "cum": "_c_lo"}),
        left_on="_lo",
        right_on="_lo_ts",
        by="market_id",
        strategy="backward",
        check_sortedness=False,
    )
    b = a.sort("_hi").join_asof(
        tr.rename({"ts": "_hi_ts", "cum": "_c_hi"}),
        left_on="_hi",
        right_on="_hi_ts",
        by="market_id",
        strategy="backward",
        check_sortedness=False,
    )
    vol = b.select("market_id", time_col, vol_window=pl.col("_c_hi").fill_null(0.0) - pl.col("_c_lo").fill_null(0.0))
    return (
        rows.join(vol, on=["market_id", time_col], how="left")
        .with_columns(vol_window=pl.col("vol_window").fill_null(0.0))
        .with_columns(traded_near=pl.col("vol_window") > 0)
    )


def first_print(
    rows: pl.DataFrame,
    trades: pl.DataFrame,
    window: timedelta = timedelta(minutes=60),
    time_col: str = "decision_time",
    limit_col: str | None = None,
) -> pl.DataFrame:
    """First same-side taker print in (t, t + window].

    `rows` need market_id, `time_col` and `side` ("YES" = buy outcome 0, "NO" = buy outcome 1). Adds `print_px` (the
    side's price of that print), `print_ts`, `print_usd` (all same-side notional in the window) and, if `limit_col`
    is given, `fillable_usd` / `fillable_shares` of same-side prints at or below that limit.
    """
    keys = ["market_id", time_col, "side", *([limit_col] if limit_col else [])]
    k = rows.select(keys).unique()
    j = k.join(trades.select("market_id", "ts", "px0", "buys0", "size"), on="market_id").filter(
        (pl.col("ts") > pl.col(time_col))
        & (pl.col("ts") <= pl.col(time_col) + window)
        & (pl.col("buys0") == (pl.col("side") == "YES"))
    )
    j = j.with_columns(side_px=pl.when(pl.col("side") == "YES").then(pl.col("px0")).otherwise(1 - pl.col("px0")))
    aggs = [
        pl.col("side_px").first().alias("print_px"),
        pl.col("ts").first().alias("print_ts"),
        (pl.col("side_px") * pl.col("size")).sum().alias("print_usd"),
    ]
    if limit_col:
        ok = pl.col("side_px") <= pl.col(limit_col)
        aggs += [
            (pl.col("side_px") * pl.col("size")).filter(ok).sum().alias("fillable_usd"),
            pl.col("size").filter(ok).sum().alias("fillable_shares"),
        ]
    first = j.sort("ts").group_by(keys).agg(aggs)
    return rows.join(first, on=keys, how="left")


def reaction_minutes(
    events: pl.DataFrame, mids: pl.DataFrame, move: float = 0.05, horizon: timedelta = timedelta(hours=6)
) -> pl.DataFrame:
    """Minutes from `t0` until the outcome-0 mid first moved more than `move` toward the resolved outcome.

    `events`: market_id, t0 (e.g. an observation time), winner (0/1). The reference mid is the last one at or before
    t0. Adds `reaction_min` (null if no such move within `horizon`).
    """
    px = mids.select("market_id", "ts", "p").sort("ts")
    ref = events.sort("t0").join_asof(
        px.rename({"ts": "_ref_ts", "p": "_p0"}),
        left_on="t0",
        right_on="_ref_ts",
        by="market_id",
        strategy="backward",
        check_sortedness=False,
    )
    j = ref.join(px, on="market_id").filter((pl.col("ts") > pl.col("t0")) & (pl.col("ts") <= pl.col("t0") + horizon))
    toward = pl.when(pl.col("winner") == 0).then(pl.col("p") - pl.col("_p0")).otherwise(pl.col("_p0") - pl.col("p"))
    hit = j.filter(toward > move).group_by("market_id", "t0").agg(_first=pl.col("ts").min())
    out = events.join(hit, on=["market_id", "t0"], how="left")
    return out.with_columns(reaction_min=(pl.col("_first") - pl.col("t0")).dt.total_seconds() / 60.0).drop("_first")
