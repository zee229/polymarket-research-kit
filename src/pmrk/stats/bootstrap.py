"""Block bootstrap: resample whole blocks (events or dates), never single rows.

Rows inside one event (several snapshots, several buckets) or one date (weather, news) are strongly correlated;
resampling rows would produce CIs that are far too narrow.
"""

from __future__ import annotations

import numpy as np
import polars as pl

N_BOOT = 2000
SEED = 7


def ratio_ci(
    df: pl.DataFrame, num: str, den: str, block: str, n_boot: int = N_BOOT, seed: int = SEED, level: float = 0.95
) -> tuple[float, float, float]:
    """Point estimate and CI of sum(num) / sum(den), resampling `block` (e.g. ROI = PnL / spend by date)."""
    g = df.group_by(block).agg(pl.col(num).sum(), pl.col(den).sum())
    a, b = g[num].to_numpy(), g[den].to_numpy()
    point = float(a.sum() / b.sum()) if b.sum() else float("nan")
    if len(a) < 2:
        return point, float("nan"), float("nan")
    idx = np.random.default_rng(seed).integers(0, len(a), (n_boot, len(a)))
    boot = a[idx].sum(1) / b[idx].sum(1)
    q = (1 - level) / 2 * 100
    return point, float(np.percentile(boot, q)), float(np.percentile(boot, 100 - q))


def mean_ci(
    df: pl.DataFrame, value: str, block: str, n_boot: int = N_BOOT, seed: int = SEED, level: float = 0.95
) -> tuple[float, float, float]:
    """Mean of `value` over rows with a block-bootstrap CI."""
    return ratio_ci(df.with_columns(_one=pl.lit(1.0)), value, "_one", block, n_boot, seed, level)


def sum_ci(
    df: pl.DataFrame, value: str, block: str, n_boot: int = N_BOOT, seed: int = SEED, level: float = 0.95
) -> tuple[float, float, float]:
    """Total of `value` with a block-bootstrap CI (e.g. total PnL by date)."""
    s = df.group_by(block).agg(pl.col(value).sum())[value].to_numpy()
    if len(s) < 2:
        return float(s.sum()), float("nan"), float("nan")
    idx = np.random.default_rng(seed).integers(0, len(s), (n_boot, len(s)))
    boot = s[idx].sum(1)
    q = (1 - level) / 2 * 100
    return float(s.sum()), float(np.percentile(boot, q)), float(np.percentile(boot, 100 - q))
