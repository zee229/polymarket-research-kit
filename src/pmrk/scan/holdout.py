"""Calibration scan, holdout side: test the FROZEN candidates with three execution models.

- `mid`: side mid + measured median cost (what discovery assumed; not executable in thin books);
- `print`: the first real same-side taker print within `entry_window` after the snapshot (no print = no trade);
- `print_lim`: as `print`, but only if that print is at or below the mid-based price (a limit order at our
  assumption).
The gap between `mid` and `print` is the part of an apparent edge that lives inside the spread.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import polars as pl

from pmrk.execution.costs import attach_cost
from pmrk.execution.fees import taker_fee_per_share
from pmrk.execution.fills import first_print
from pmrk.stats.bootstrap import ratio_ci

MAX_EVENTS = 400
ENTRY_WINDOW = timedelta(minutes=60)
SEED = 20260930
VARIANTS = ("mid", "print", "print_lim")


def candidate_rows(
    holdout: pl.DataFrame, spec: dict, costs: pl.DataFrame | None, max_events: int = MAX_EVENTS, seed: int = SEED
) -> pl.DataFrame:
    """Holdout snapshots in the candidate's cell (up to `max_events` random events) with the mid-based price."""
    sel = holdout.filter(pl.col("band") == spec["band"])
    if spec["category"] != "ALL":
        sel = sel.filter(pl.col("category") == spec["category"])
    if spec["snapshot"] != "ALL":
        sel = sel.filter(pl.col("snapshot") == spec["snapshot"])
    if costs is not None and costs.height:
        by = ["category"] if "category" in costs.columns else None
        sel = attach_cost(sel, costs, by=by)
    else:
        sel = sel.with_columns(cost=pl.lit(0.02))
    ev = sel.select("event_id").unique().sort("event_id")
    ev = ev.filter(pl.int_range(pl.len()).shuffle(seed=seed) < max_events)
    yes = spec["side"] == "YES"
    side_mid = pl.col("p") if yes else 1 - pl.col("p")
    return sel.join(ev, on="event_id").with_columns(
        side=pl.lit(spec["side"]), mid_price=side_mid + pl.col("cost"), win=pl.col("y") if yes else 1 - pl.col("y")
    )


def _score(v: pl.DataFrame, price: str) -> dict[str, Any]:
    v = v.with_columns(fee=taker_fee_per_share(pl.col(price))).with_columns(
        pnl=pl.col("win") - pl.col(price) - pl.col("fee"),
        spend=pl.col(price) + pl.col("fee"),
        day=pl.col("event_end").dt.date(),
    )
    roi, lo, hi = ratio_ci(v, "pnl", "spend", "event_id")
    _, dlo, dhi = ratio_ci(v, "pnl", "spend", "day")
    ev = v.group_by("event_id").agg(pl.col("pnl").sum(), pl.col("spend").sum()).sort("pnl", descending=True).slice(5)
    ser = (
        v.group_by(pl.coalesce("series_slug", "event_slug").alias("s"))
        .agg(pl.col("pnl").sum())
        .sort("pnl", descending=True)
    )
    tot = float(v["pnl"].sum())
    return {
        "n": v.height,
        "events": v["event_id"].n_unique(),
        "roi": roi,
        "ci_event": [lo, hi],
        "ci_date": [dlo, dhi],
        "roi_wo_top5": float(ev["pnl"].sum() / ev["spend"].sum()) if ev.height else None,
        "top_series_share": float(ser["pnl"][0] / tot) if tot > 0 else None,
    }


def evaluate_candidate(rows: pl.DataFrame, trades: pl.DataFrame, entry_window: timedelta = ENTRY_WINDOW) -> dict:
    """Scores of all three execution variants plus fill rate and fillable USD per day."""
    rows = first_print(rows, trades, entry_window, time_col="target_ts", limit_col="mid_price")
    variants = {
        "mid": (rows, "mid_price"),
        "print": (rows.filter(pl.col("print_px").is_not_null()), "print_px"),
        "print_lim": (rows.filter(pl.col("print_px") <= pl.col("mid_price")), "print_px"),
    }
    out: dict[str, Any] = {}
    for name, (v, price) in variants.items():
        v = v.filter(pl.col(price).is_between(0.001, 0.999))
        scores = _score(v, price) if v.height else {"n": 0}
        out.update({f"{name}_{k}": val for k, val in scores.items()})
    days = max(rows["event_end"].dt.date().n_unique(), 1)
    out["fill_rate"] = float(rows["print_px"].is_not_null().mean()) if rows.height else None
    out["fillable_usd_per_day"] = float(rows["fillable_usd"].fill_null(0).sum() / days) if rows.height else 0.0
    return out


def passes(r: dict, prefix: str = "print_lim", min_events: int = 200, min_usd_per_day: float = 50.0) -> bool:
    """All criteria: ROI > 0, event- and date-block CIs above 0, enough events, survives removing the top 5 events,
    no single series carries half the PnL, and enough fillable volume to matter."""
    ci, cid = r.get(f"{prefix}_ci_event") or [None], r.get(f"{prefix}_ci_date") or [None]
    share = r.get(f"{prefix}_top_series_share")
    return bool(
        r.get(f"{prefix}_n")
        and (r.get(f"{prefix}_roi") or -1) > 0
        and ci[0] is not None
        and ci[0] > 0
        and cid[0] is not None
        and cid[0] > 0
        and (r.get(f"{prefix}_events") or 0) >= min_events
        and (r.get(f"{prefix}_roi_wo_top5") or -1) > 0
        and (share is None or share < 0.5)
        and (r.get("fillable_usd_per_day") or 0) > min_usd_per_day
    )


def evaluate(
    holdout: pl.DataFrame,
    frozen: list[dict],
    trades: pl.DataFrame,
    costs: pl.DataFrame | None,
    max_events: int = MAX_EVENTS,
) -> pl.DataFrame:
    """One result row per frozen candidate (input must already be restricted to split == 'holdout')."""
    results = []
    for i, spec in enumerate(frozen):
        rows = candidate_rows(holdout, spec, costs, max_events, SEED + i)
        res = {"cand": i + 1, **spec, **(evaluate_candidate(rows, trades) if rows.height else {"mid_n": 0})}
        res["pass"] = passes(res)
        results.append(res)
    return pl.DataFrame(results, infer_schema_length=None)
