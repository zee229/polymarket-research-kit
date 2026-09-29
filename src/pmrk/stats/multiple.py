"""Multiple-testing control for screens over many cells."""

from __future__ import annotations

import numpy as np


def benjamini_hochberg(pvals: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Boolean mask of hypotheses rejected at FDR q (NaN p-values never pass)."""
    p = np.asarray(pvals, float)
    p = np.where(np.isnan(p), 1.0, p)
    n = len(p)
    if n == 0:
        return np.zeros(0, bool)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    below = np.nonzero(ranked <= q)[0]
    out = np.zeros(n, bool)
    if below.size:
        out[order[: below.max() + 1]] = True
    return out


def bonferroni_threshold(alpha: float, n_tests: int) -> float:
    return alpha / max(n_tests, 1)
