"""Calibration cells (price vs realized frequency) and model-vs-market head-to-head scoring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import norm

from pmrk.stats.bootstrap import mean_ci
from pmrk.stats.scores import EPS


def cell_stats(
    df: pl.DataFrame, by: list[str], price: str = "p", outcome: str = "y", cluster: str = "event_id"
) -> pl.DataFrame:
    """Mean price vs realized frequency per cell with a cluster-robust (by event) z-test of `freq - price`.

    `diff < 0` means the priced side was overpriced. The variance treats each cluster's summed residual as one
    observation (ratio estimator), because snapshots of the same event are not independent.
    """
    ev = df.group_by(*by, cluster).agg(
        r=(pl.col(outcome) - pl.col(price)).sum(), n=pl.len(), p_sum=pl.col(price).sum(), y_sum=pl.col(outcome).sum()
    )
    g = (
        ev.group_by(by)
        .agg(
            n=pl.col("n").sum(),
            events=pl.len(),
            p_mean=pl.col("p_sum").sum() / pl.col("n").sum(),
            freq=pl.col("y_sum").sum() / pl.col("n").sum(),
            r_list=pl.col("r"),
            n_list=pl.col("n"),
        )
        .with_columns(diff=pl.col("freq") - pl.col("p_mean"))
    )
    se = []
    for r, n, d in zip(g["r_list"].to_list(), g["n_list"].to_list(), g["diff"].to_list(), strict=True):
        r, n = np.asarray(r, float), np.asarray(n, float)
        k = len(r)
        u = r - d * n
        se.append(float(np.sqrt(k / (k - 1) * np.sum(u**2)) / np.sum(n)) if k > 1 else float("nan"))
    g = (
        g.drop("r_list", "n_list")
        .with_columns(se=pl.Series(se, dtype=pl.Float64))
        .with_columns(
            z=pl.col("diff") / pl.col("se"),
            ci_lo=pl.col("diff") - 1.96 * pl.col("se"),
            ci_hi=pl.col("diff") + 1.96 * pl.col("se"),
        )
    )
    z = g["z"].fill_nan(None).fill_null(0.0).to_numpy()
    pval = np.where(g["se"].fill_nan(None).is_null().to_numpy(), np.nan, 2 * norm.sf(np.abs(z)))
    return g.with_columns(pval=pl.Series(pval))


def side_view(df: pl.DataFrame, price: str = "p", outcome: str = "y") -> pl.DataFrame:
    """Each binary as two rows: side YES (price p, win y) and side NO (price 1-p, win 1-y)."""
    yes = df.with_columns(side=pl.lit("YES"))
    no = df.with_columns(side=pl.lit("NO"), **{price: 1 - pl.col(price), outcome: 1 - pl.col(outcome)})
    return pl.concat([yes, no])


def outcome_losses(panel: pl.DataFrame, model_col: str = "p_model") -> pl.DataFrame:
    """Per (event, decision) log loss of the market and the model on what happened.

    negRisk events: `-log` of the normalized probability of the winning market (`q_market`, model column normalized
    the same way by the caller). Other markets: binary log loss of `p` and the model on `winner == 0`.
    """
    clip = lambda c: pl.col(c).clip(EPS, 1.0)  # noqa: E731
    neg = panel.filter(pl.col("neg_risk") & (pl.col("winner") == 0)).with_columns(
        ll_market=-clip("q_market").log(), ll_model=-clip(model_col).log()
    )
    y = pl.col("winner") == 0
    other = panel.filter(~pl.col("neg_risk")).with_columns(
        ll_market=-pl.when(y).then(clip("p")).otherwise((1 - pl.col("p")).clip(EPS, 1.0)).log(),
        ll_model=-pl.when(y).then(clip(model_col)).otherwise((1 - pl.col(model_col)).clip(EPS, 1.0)).log(),
    )
    return pl.concat([neg, other], how="diagonal_relaxed").with_columns(diff=pl.col("ll_model") - pl.col("ll_market"))


def head_to_head(losses: pl.DataFrame, group: str | None = None, block: str = "block") -> pl.DataFrame:
    """Mean market / model log loss and `model - market` with a block-bootstrap 95% CI (per `group`)."""
    groups = losses.partition_by(group, as_dict=True) if group else {("all",): losses}
    rows = []
    for key, sub in sorted(groups.items(), key=lambda kv: str(kv[0])):
        d, lo, hi = mean_ci(sub, "diff", block)
        rows.append(
            {
                "group": str(key[0]),
                "n": sub.height,
                "events": sub["event_id"].n_unique(),
                "ll_market": sub["ll_market"].mean(),
                "ll_model": sub["ll_model"].mean(),
                "model_minus_market": d,
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )
    return pl.DataFrame(rows)


def plot_reliability(cells: pl.DataFrame, path: Path, facet: str | None = None, title: str = "") -> None:
    """Reliability diagram from `cell_stats` output (needs the [plots] extra)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    facets = sorted(cells[facet].unique().to_list()) if facet else [None]
    cols = min(4, len(facets))
    rows = int(np.ceil(len(facets) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows), squeeze=False)
    for ax, f in zip(axes.flat, facets, strict=False):
        sub = (cells if f is None else cells.filter(pl.col(facet) == f)).sort("p_mean")
        ax.errorbar(sub["p_mean"], sub["freq"], yerr=1.96 * sub["se"].fill_null(0), fmt="o-", ms=3, lw=1)
        ax.plot([0, 1], [0, 1], color="0.7", lw=0.6)
        ax.set_title(str(f) if f is not None else "all", fontsize=9)
    for ax in list(axes.flat)[len(facets) :]:
        ax.axis("off")
    fig.suptitle(title or "price vs realized frequency")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
