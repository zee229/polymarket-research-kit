"""Proper scoring rules. Lower is better for all of them."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

EPS = 1e-4


def log_loss(p_outcome: np.ndarray, eps: float = EPS) -> np.ndarray:
    """-log of the probability assigned to what actually happened (clipped at eps)."""
    return -np.log(np.clip(np.asarray(p_outcome, float), eps, 1.0))


def binary_log_loss(p: np.ndarray, y: np.ndarray, eps: float = EPS) -> np.ndarray:
    p, y = np.asarray(p, float), np.asarray(y, float)
    return log_loss(np.where(y == 1, p, 1 - p), eps)


def brier(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (np.asarray(p, float) - np.asarray(y, float)) ** 2


def crps_gaussian(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Closed-form CRPS of N(mu, sigma) at observation y."""
    z = (np.asarray(y, float) - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
