import numpy as np
import polars as pl
import pytest
from scipy import integrate
from scipy.stats import norm

from pmrk.stats.bootstrap import mean_ci, ratio_ci
from pmrk.stats.multiple import benjamini_hochberg, bonferroni_threshold
from pmrk.stats.reliability import cell_stats, head_to_head, outcome_losses, side_view
from pmrk.stats.scores import binary_log_loss, brier, crps_gaussian


def test_benjamini_hochberg_step_up():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    assert benjamini_hochberg(p, 0.05).sum() == 2
    assert not benjamini_hochberg(np.array([np.nan, 0.9]), 0.05).any()
    assert bonferroni_threshold(0.05, 200) == pytest.approx(2.5e-4)


def test_crps_gaussian_matches_numeric_integral():
    y, mu, s = 1.3, 0.2, 0.8
    num, _ = integrate.quad(lambda x: (norm.cdf(x, mu, s) - (x >= y)) ** 2, -20, 20, points=[y])
    assert crps_gaussian(np.array([y]), mu, s)[0] == pytest.approx(num, rel=1e-6)


def test_scores():
    assert binary_log_loss([0.8, 0.8], [1, 0]).tolist() == pytest.approx([-np.log(0.8), -np.log(0.2)])
    assert brier([0.8], [1]).tolist() == pytest.approx([0.04])


def test_ratio_ci_resamples_blocks():
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"day": np.repeat(np.arange(50), 20), "pnl": rng.normal(0.1, 1, 1000), "spend": 1.0})
    roi, lo, hi = ratio_ci(df, "pnl", "spend", "day")
    assert lo < roi < hi
    _, lo1, hi1 = ratio_ci(df.with_columns(day=pl.int_range(1000)), "pnl", "spend", "day")
    assert (hi - lo) > 0 and (hi1 - lo1) > 0


def test_cell_stats_detects_overpricing_with_cluster_se():
    rng = np.random.default_rng(1)
    n_ev = 400
    y = (rng.random(n_ev) < 0.3).astype(int)  # priced at 0.4, happens 30% of the time
    df = pl.DataFrame({"event_id": np.repeat(np.arange(n_ev), 3), "p": 0.4, "y": np.repeat(y, 3), "g": "x"})
    c = cell_stats(df, ["g"]).row(0, named=True)
    assert c["events"] == n_ev and c["diff"] < 0 and c["pval"] < 0.01
    # three identical snapshots per event must not triple the evidence
    c1 = cell_stats(df.unique("event_id"), ["g"]).row(0, named=True)
    assert c["se"] == pytest.approx(c1["se"], rel=0.05)


def test_side_view_mirrors():
    sv = side_view(pl.DataFrame({"p": [0.3], "y": [1]}))
    assert sv.sort("side")["p"].to_list() == pytest.approx([0.7, 0.3]) and sv.sort("side")["y"].to_list() == [0, 1]


def test_outcome_losses_negrisk_and_binary():
    panel = pl.DataFrame(
        {
            "event_id": ["e", "e", "b"],
            "decision_time": 0,
            "neg_risk": [True, True, False],
            "winner": [0, 1, 1],
            "p": [0.6, 0.5, 0.2],
            "q_market": [0.6 / 1.1, 0.5 / 1.1, 0.2],
            "p_model": [0.5, 0.5, 0.1],
            "block": 1,
        }
    )
    ll = outcome_losses(panel).sort("event_id")
    assert ll["ll_market"].to_list() == pytest.approx([-np.log(0.8), -np.log(0.6 / 1.1)])
    assert ll["ll_model"].to_list() == pytest.approx([-np.log(0.9), -np.log(0.5)])
    h = head_to_head(ll)
    assert h["n"][0] == 2


def test_mean_ci():
    df = pl.DataFrame({"v": [1.0, 2.0, 3.0, 4.0], "b": [1, 2, 3, 4]})
    m, lo, hi = mean_ci(df, "v", "b")
    assert m == 2.5 and lo <= m <= hi
