"""Gaussian EMOS for the settlement daily max (degC), fitted per horizon bin by interval-censored MLE.

- pre-day (no observation of day D yet): X ~ N(mu, sigma), pooled over stations;
- intraday: X = max(M, R) with M the observed max so far (hard lower bound) and R ~ N(mu_R, sigma_R) the max of
  the remaining reports; per-station intercepts on mean and log-sigma, ridge-shrunk toward their mean (unseen
  stations get the mean effect).
Targets are interval-censored at reporting precision (0.5 degC integer METAR, 0.05 degC T-group).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.stats import norm

PRE_BINS = [(24.0, 36.0), (36.0, 48.0), (48.0, 60.0), (60.0, 200.0)]
MIN_SIGMA = 0.15
RIDGE = 2.0
LOG_SIG_CLIP = (-5.0, 4.0)
MIN_TRAIN = 300


def horizon_bin(hours_to_eod: pl.Expr, local_hour: pl.Expr, obs_n: pl.Expr) -> pl.Expr:
    """'pre_<lo>' before day D; 'intra_<hour>' on day D once an observation exists (else 'pre_24')."""
    expr = pl.lit(None, dtype=pl.Utf8)
    for lo, hi in reversed(PRE_BINS):
        expr = pl.when((hours_to_eod >= lo) & (hours_to_eod < hi)).then(pl.lit(f"pre_{int(lo)}")).otherwise(expr)
    intra = pl.lit("intra_") + local_hour.floor().cast(pl.Int32).cast(pl.Utf8).str.zfill(2)
    on_day = (hours_to_eod < 24.0) & (hours_to_eod > 0.0)
    return pl.when(on_day & (obs_n > 0)).then(intra).when(on_day).then(pl.lit("pre_24")).otherwise(expr)


def interval_prob(a: np.ndarray, b: np.ndarray, mu: np.ndarray, sigma: np.ndarray, m: np.ndarray) -> np.ndarray:
    """P(a <= max(m, R) < b) with R ~ N(mu, sigma); m = -inf before day D."""
    fb = np.where(b > m, norm.cdf((b - mu) / sigma), 0.0)
    fa = np.where(a > m, norm.cdf((a - mu) / sigma), 0.0)
    return np.clip(fb - fa, 0.0, 1.0)


def _col(df: pl.DataFrame, name: str, fill: float | np.ndarray | None = None) -> np.ndarray:
    base = df[name].cast(pl.Float64).to_numpy().copy() if name in df.columns else np.full(df.height, np.nan)
    return base if fill is None else np.where(np.isnan(base), fill, base)


def design(df: pl.DataFrame, intraday: bool) -> tuple[np.ndarray, np.ndarray]:
    """Shared (non-station) columns of the mean and log-sigma designs."""
    ens = _col(df, "ens_mean")
    mse_cols = [c for c in df.columns if c.startswith("mse_")]
    rmse = np.full(df.height, 1.5)
    if mse_cols:
        mse = df.select(mse_cols).cast(pl.Float64).to_numpy()
        n = (~np.isnan(mse)).sum(axis=1)
        rmse = np.where(n > 0, np.sqrt(np.nansum(mse, axis=1) / np.maximum(n, 1)), 1.5)
    age = _col(df, "age_h_ecmwf_ifs025", _col(df, "age_h_gfs_050", 24.0)) / 24.0
    ecd, sd = _col(df, "ec_minus_gfs", 0.0), _col(df, "ens_sd", 1.0)
    one = np.ones(df.height)
    if not intraday:
        return (np.column_stack([one, ens, ecd]), np.column_stack([one, np.log(sd + 0.3), np.log(rmse + 0.3), age]))
    last, m = _col(df, "obs_last"), _col(df, "obs_max")
    rem = _col(df, "ens_rem", last)
    cloud, conv = _col(df, "cloud", 0.4), _col(df, "convective", 0.0)
    wind, trend = _col(df, "wind_kt", 8.0) / 10.0, _col(df, "trend_3h", 0.0)
    rise = _col(df, "clim_rise", 1.0)
    to_peak = _col(df, "clim_peak_hour", 14.0) - _col(df, "local_hour")
    xm = np.column_stack([rem, ens, ecd, last, m, trend, cloud, conv, wind, rise, to_peak, age])
    xs = np.column_stack(
        [
            np.log(sd + 0.3),
            np.log(rmse + 0.3),
            cloud,
            np.log(np.clip(rise, 0, None) + 0.5),
            np.log(np.clip(rem - last, 0, None) + 0.5),
        ]
    )
    return xm, xs


def _station_matrix(df: pl.DataFrame, codes: tuple[str, ...]) -> np.ndarray:
    idx = {s: i for i, s in enumerate(codes)}
    col = np.array([idx.get(s, -1) for s in df["station_icao"].to_list()])
    mat = np.zeros((df.height, len(codes)))
    ok = col >= 0
    mat[np.arange(df.height)[ok], col[ok]] = 1.0
    mat[~ok] = 1.0 / len(codes)  # unseen station: mean effect
    return mat


def _nll_grad(
    theta: np.ndarray, xm: np.ndarray, xs: np.ndarray, lo: np.ndarray, hi: np.ndarray, n_st: int, ridge: float
) -> tuple[float, np.ndarray]:
    km = xm.shape[1]
    mu = xm @ theta[:km]
    sig = np.maximum(np.exp(np.clip(xs @ theta[km:], *LOG_SIG_CLIP)), MIN_SIGMA)
    zh, zl = (hi - mu) / sig, (lo - mu) / sig
    p = np.clip(norm.cdf(zh) - norm.cdf(zl), 1e-12, None)
    ph, pl_ = norm.pdf(zh), norm.pdf(zl)
    d_mu = (ph - pl_) / (sig * p)
    d_ls = (zh * ph - zl * pl_) / p
    nll = -np.sum(np.log(p))
    g = np.concatenate([xm.T @ d_mu, xs.T @ d_ls])
    for start in (0, km) if n_st else ():  # station blocks lead both designs
        dev = theta[start : start + n_st] - theta[start : start + n_st].mean()
        nll += ridge * np.sum(dev**2)
        g[start : start + n_st] += 2 * ridge * dev
    return nll, g


@dataclass(frozen=True)
class FittedBin:
    name: str
    params: np.ndarray
    stations: tuple[str, ...]
    n_train: int

    @property
    def intraday(self) -> bool:
        return self.name.startswith("intra")

    def matrices(self, df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        xm, xs = design(df, self.intraday)
        if not self.stations:
            return xm, xs
        st = _station_matrix(df, self.stations)
        return np.column_stack([st, xm]), np.column_stack([st, xs])

    def predict(self, df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        xm, xs = self.matrices(df)
        km = xm.shape[1]
        return xm @ self.params[:km], np.maximum(np.exp(np.clip(xs @ self.params[km:], *LOG_SIG_CLIP)), MIN_SIGMA)


def fit_bin(name: str, df: pl.DataFrame) -> FittedBin | None:
    intraday = name.startswith("intra")
    target = "rem_max" if intraday else "y_c"
    d = df.drop_nulls(["ens_mean", target, "y_halfwidth", *(["obs_max", "obs_last"] if intraday else [])])
    if d.height < MIN_TRAIN:
        return None
    codes = tuple(sorted(d["station_icao"].unique().to_list())) if intraday else ()
    xm, xs = FittedBin(name, np.zeros(0), codes, d.height).matrices(d)
    y, hw = d[target].to_numpy(), d["y_halfwidth"].to_numpy()
    beta, *_ = np.linalg.lstsq(xm, y, rcond=None)
    gamma0 = np.zeros(xs.shape[1])
    gamma0[: len(codes) or 1] = np.log(max(np.std(y - xm @ beta), 0.3))
    res = minimize(
        _nll_grad,
        np.concatenate([beta, gamma0]),
        args=(xm, xs, y - hw, y + hw, len(codes), RIDGE),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 500},
    )
    return FittedBin(name, res.x, codes, d.height) if np.all(np.isfinite(res.x)) else None


def fit_all(train: pl.DataFrame) -> dict[str, FittedBin]:
    fits = {name: fit_bin(name, sub) for (name,), sub in train.group_by("hbin") if name is not None}
    return {k: v for k, v in fits.items() if v is not None}


def predict_all(models: dict[str, FittedBin], df: pl.DataFrame) -> pl.DataFrame:
    """Adds mu, sigma, lower_bound (observed max on day D, else -inf)."""
    parts = []
    for (name,), sub in df.group_by("hbin"):
        model = models.get(name)
        sub = sub.drop_nulls(["ens_mean"])
        if model is not None and model.intraday:
            sub = sub.drop_nulls(["obs_max", "obs_last"])
        if model is None or sub.is_empty():
            continue
        mu, sigma = model.predict(sub)
        lb = sub["obs_max"].to_numpy().astype(float) if model.intraday else np.full(sub.height, -np.inf)
        parts.append(sub.with_columns(mu=pl.Series(mu), sigma=pl.Series(sigma), lower_bound=pl.Series(lb)))
    return pl.concat(parts, how="diagonal_relaxed") if parts else df.clear()
