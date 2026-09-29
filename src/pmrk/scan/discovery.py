"""Calibration scan, discovery side: sample, assemble snapshots, screen cells, freeze candidates.

Protocol against fooling yourself:
1. split events by end date into discovery and holdout;
2. `discover` reads only discovery rows, tests every (category x band x horizon) cell plus pooled cells, applies
   Benjamini-Hochberg, keeps cells with positive expected ROI after measured cost and fees, and freezes them to
   `candidates.json` with a sha256 fingerprint (`candidates.md` is the human-readable copy);
3. `holdout.evaluate` refuses to run without that file and tests only the frozen cells.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from pmrk.execution.costs import band
from pmrk.execution.fees import taker_fee_per_share
from pmrk.stats.multiple import benjamini_hochberg
from pmrk.stats.reliability import cell_stats

MIN_EVENTS_CELL = 100
BH_Q = 0.05
DEFAULT_COST = 0.02


def sample_markets(
    markets: pl.DataFrame,
    events_per_category: int = 3000,
    markets_per_event: int = 3,
    seed: int = 20260928,
    neg_risk_only: bool = False,
) -> pl.DataFrame:
    """Stratified random sample of clean markets: up to N events per category, up to K markets per event (0 = all)."""
    t = markets.filter(pl.col("clean") & pl.col("token0").is_not_null())
    if neg_risk_only:
        t = t.filter(pl.col("neg_risk"))
    ev = t.select("category", "event_id").unique().sort("event_id")
    ev = ev.filter(pl.int_range(pl.len()).shuffle(seed=seed).over("category") < events_per_category)
    m = t.join(ev.select("event_id"), on="event_id").sort("market_id")
    if markets_per_event:
        m = m.filter(pl.int_range(pl.len()).shuffle(seed=seed).over("event_id") < markets_per_event)
    return m


def assemble(snaps: pl.DataFrame, markets: pl.DataFrame, split: date) -> pl.DataFrame:
    """Join snapshot prices with market metadata, outcome `y` (outcome 0 won), band and discovery/holdout split."""
    meta = markets.with_columns(event_end=pl.col("end_ts").max().over("event_id")).select(
        "market_id",
        "event_id",
        "category",
        "neg_risk",
        "structure",
        "markets_in_event",
        "series_slug",
        "event_slug",
        "end_ts",
        "closed_ts",
        "created_ts",
        "volume",
        "fees_enabled",
        "fee_rate",
        "fee_exponent",
        "winner",
        "resolved_early_1h",
        "event_end",
        life_days=(pl.col("closed_ts") - pl.col("created_ts")).dt.total_hours() / 24.0,
    )
    split_ts = datetime(split.year, split.month, split.day, tzinfo=UTC)
    return snaps.join(meta, on="market_id", how="inner").with_columns(
        y=(pl.col("winner") == 0).cast(pl.Int8),
        hours_to_end=(pl.col("end_ts") - pl.col("target_ts")).dt.total_minutes() / 60.0,
        split=pl.when(pl.col("event_end") < split_ts).then(pl.lit("discovery")).otherwise(pl.lit("holdout")),
        band=band("p"),
        vol_per_day=pl.col("volume") / pl.col("life_days").clip(1 / 24, None),
    )


def all_cells(d: pl.DataFrame) -> pl.DataFrame:
    """Every (category | ALL) x band x (horizon | ALL) cell, with the mean effective fee rate of the cell."""
    d = d.with_columns(eff_rate=pl.when(pl.col("fees_enabled")).then(pl.col("fee_rate")).otherwise(0.0))
    fee = [pl.col("eff_rate").mean().alias("fee_rate_mean")]
    parts = []
    for keys, fill in (
        (["category", "band", "snapshot"], {}),
        (["category", "band"], {"snapshot": "ALL"}),
        (["band", "snapshot"], {"category": "ALL"}),
        (["band"], {"category": "ALL", "snapshot": "ALL"}),
    ):
        c = cell_stats(d, keys).join(d.group_by(keys).agg(fee), on=keys)
        parts.append(c.with_columns(**{k: pl.lit(v) for k, v in fill.items()}))
    cols = [
        "category",
        "band",
        "snapshot",
        "n",
        "events",
        "p_mean",
        "freq",
        "diff",
        "se",
        "z",
        "pval",
        "ci_lo",
        "ci_hi",
        "fee_rate_mean",
    ]
    return pl.concat([p.select(cols) for p in parts])


def expected_roi(cells: pl.DataFrame, costs: pl.DataFrame | None) -> pl.DataFrame:
    """ROI per $ of buying the underpriced side at the cell's mean price + measured median cost + fee."""
    if costs is not None and costs.height:
        pooled = costs.group_by("band").agg(_pooled=pl.col("median_cost").median())
        keyed = costs.select("category", "band", _cell=pl.col("median_cost")) if "category" in costs.columns else None
        if keyed is not None:
            cells = cells.join(keyed, on=["category", "band"], how="left")
        else:
            cells = cells.with_columns(_cell=pl.lit(None, pl.Float64))
        cells = cells.join(pooled, on="band", how="left")
    else:
        cells = cells.with_columns(_cell=pl.lit(None, pl.Float64), _pooled=pl.lit(None, pl.Float64))
    cells = cells.with_columns(cost=pl.coalesce("_cell", "_pooled", pl.lit(DEFAULT_COST)).clip(0.0, None)).drop(
        "_cell", "_pooled"
    )
    buy_yes = pl.col("diff") > 0
    price = pl.when(buy_yes).then(pl.col("p_mean")).otherwise(1 - pl.col("p_mean")) + pl.col("cost")
    win = pl.when(buy_yes).then(pl.col("freq")).otherwise(1 - pl.col("freq"))
    fee = taker_fee_per_share(price, enabled=pl.lit(True), rate=pl.col("fee_rate_mean"), exponent=pl.lit(1.0))
    return cells.with_columns(
        side=pl.when(buy_yes).then(pl.lit("YES")).otherwise(pl.lit("NO")),
        exec_price=price,
        exp_roi=(win - price - fee) / (price + fee),
    )


def discover(
    assembled: pl.DataFrame, costs: pl.DataFrame | None, min_events: int = MIN_EVENTS_CELL, q: float = BH_Q
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(all tested cells, candidates). Reads ONLY discovery rows."""
    d = assembled.filter(pl.col("split") == "discovery")
    cells = all_cells(d).filter(pl.col("events") >= min_events).sort("category", "band", "snapshot")
    cells = cells.with_columns(bh_pass=pl.Series(benjamini_hochberg(cells["pval"].to_numpy(), q)))
    cells = expected_roi(cells, costs)
    cand = cells.filter(pl.col("bh_pass") & (pl.col("exp_roi") > 0)).sort("exp_roi", descending=True)
    return cells, cand


def _digest(payload: str) -> str:
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def freeze(cand: pl.DataFrame, cells: pl.DataFrame, out_dir: Path, split: date) -> str:
    """Write candidates.json (+ .sha256) and candidates.md; returns the fingerprint."""
    spec = [
        {
            "category": r["category"],
            "band": r["band"],
            "snapshot": r["snapshot"],
            "side": r["side"],
            "disc_diff": r["diff"],
            "disc_exp_roi": r["exp_roi"],
            "disc_events": r["events"],
            "disc_pval": r["pval"],
        }
        for r in cand.iter_rows(named=True)
    ]
    payload = json.dumps(spec, indent=1, default=str)
    digest = _digest(payload)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "candidates.json").write_text(payload)
    (out_dir / "candidates.sha256").write_text(digest + "\n")
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    lines = [
        "# Candidates frozen on discovery data (before holdout)",
        "",
        f"Frozen {stamp}, sha256[:16] of candidates.json: `{digest}`",
        "",
        f"Discovery = events ending before {split}. Cells tested: {cells.height} (>= {MIN_EVENTS_CELL} events); "
        f"BH q = {BH_Q}; BH-significant: {int(cells['bh_pass'].sum())}; with expected ROI > 0 after measured "
        f"cost and fees: **{len(spec)}**.",
        "",
        "| # | Category | Band | Horizon | Side | freq - price | Exp. ROI | Events | p-value |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {i} | {s['category']} | {s['band']} | {s['snapshot']} | {s['side']} | {s['disc_diff']:+.3f} | "
        f"{s['disc_exp_roi']:+.1%} | {s['disc_events']} | {s['disc_pval']:.1e} |"
        for i, s in enumerate(spec, 1)
    ]
    (out_dir / "candidates.md").write_text("\n".join(lines) + "\n")
    return digest


def load_frozen(out_dir: Path) -> list[dict]:
    """Frozen candidates; raises if missing or modified after freezing."""
    path, sha = out_dir / "candidates.json", out_dir / "candidates.sha256"
    if not path.exists() or not sha.exists():
        raise FileNotFoundError(f"no frozen candidates in {out_dir}: run discovery first")
    payload = path.read_text()
    if _digest(payload) != sha.read_text().strip():
        raise ValueError("candidates.json changed after it was frozen; re-run discovery instead of editing it")
    return json.loads(payload)


def summary_by_band(assembled: pl.DataFrame) -> pl.DataFrame:
    """Discovery-only favorite-longshot table (price band vs realized frequency)."""
    d = assembled.filter(pl.col("split") == "discovery")
    return cell_stats(d, ["band"]).sort("p_mean")
