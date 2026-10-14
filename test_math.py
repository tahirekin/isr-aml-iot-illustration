"""Checks of the statistical building blocks shared by the Section 4 scripts.
Run from the code folder:  python tests/test_math.py
"""
import importlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import alg3_cusum_t2 as A   # any script exposes the shared helpers
import alg5_ara_choice_of_k as B

rng = np.random.default_rng(7)


def test_t2_limit_size():
    """Phase II F limit: averaged over training draws, the false-alarm rate on
    new i.i.d. normal observations should be close to alpha."""
    n, p, alpha, rates = 2000, 50, 0.01, []
    lim = A.t2_control_limit(p, n, alpha)
    for _ in range(30):
        X = rng.normal(size=(n, p)); m, S = X.mean(0), np.cov(X, rowvar=False)
        rates.append(np.mean(A.hotelling_t2(rng.normal(size=(4000, p)), m, S) > lim))
    r = float(np.mean(rates)); print(f"size check: mean FAR {r:.4f} (nominal {alpha})")
    assert abs(r - alpha) < 0.002


def test_null_moments():
    n, p, sims = 400, 20, []
    for _ in range(200):
        X = rng.normal(size=(n, p)); m, S = X.mean(0), np.cov(X, rowvar=False)
        sims.append(A.hotelling_t2(rng.normal(size=(200, p)), m, S))
    sims = np.concatenate(sims); mu, sd = A.t2_null_moments(p, n)
    print(f"moments: theory {mu:.2f}/{sd:.2f}, simulation {sims.mean():.2f}/{sims.std():.2f}")
    assert abs(sims.mean() - mu) / mu < 0.02 and abs(sims.std() - sd) / sd < 0.05


def test_projection_equals_pseudoinverse():
    """T^2 in the retained subspace equals the Moore-Penrose T^2 on standardized data."""
    n, r, p = 600, 12, 20
    X = rng.normal(size=(n, r)) @ rng.normal(size=(r, p)) * 10 ** rng.uniform(-3, 5, size=p)
    tr = A.StandardizeProject().fit(X); Y = tr.transform(X)
    new = X[:50] + 0.1 * (X[50:100] - X[:50])
    t_proj = A.hotelling_t2(tr.transform(new), Y.mean(0), np.cov(Y, rowvar=False))
    Z = (X - X.mean(0)) / X.std(0, ddof=1); Zn = (new - X.mean(0)) / X.std(0, ddof=1)
    Sp = np.linalg.pinv(np.cov(Z, rowvar=False), rcond=1e-8)
    t_pinv = np.einsum("ij,jk,ik->i", Zn, Sp, Zn)
    assert tr.p_eff == r and np.allclose(t_proj, t_pinv, rtol=1e-5)
    print(f"projection: p_eff = {tr.p_eff}, matches pseudo-inverse T^2")


def test_cusum_recursion():
    z = np.array([0.0, 2.0, 2.0, 2.0, -3.0, -3.0, 4.0, 0.5])
    S, alarms = A.cusum_upper(z, k=0.5, h=2.5)
    # S: 0, 1.5, 3.0 -> alarm at 2, reset; 1.5, 0, 0, 3.5 -> alarm at 6, reset; 0
    assert list(alarms) == [2, 6], alarms
    assert np.allclose(S, [0, 1.5, 3.0, 1.5, 0, 0, 3.5, 0])
    s = A.summarize_alarms(alarms, onset=4)
    assert s["false_alarm_before_onset"] and s["n_false_alarms_pre_onset"] == 1
    assert s["detected"] and s["detection_delay_rows"] == 2
    # a series far below its reference never signals (one-sided)
    assert A.cusum_upper(-5.0 * np.ones(50), k=0.5, h=2.5)[1].size == 0
    print("one-sided CUSUM recursion and alarm bookkeeping: ok")


def test_log_t2_null_moments():
    n, p = 400, 20
    sims = []
    for _ in range(200):
        X = rng.normal(size=(n, p)); m, S = X.mean(0), np.cov(X, rowvar=False)
        sims.append(np.log(A.hotelling_t2(rng.normal(size=(200, p)), m, S)))
    sims = np.concatenate(sims); mu, sd = A.log_t2_null_moments(p, n)
    print(f"log T^2 moments: theory {mu:.3f}/{sd:.3f}, simulation {sims.mean():.3f}/{sims.std():.3f}")
    assert abs(sims.mean() - mu) < 0.01 and abs(sims.std() - sd) / sd < 0.03


def test_calibration_reference():
    x = rng.normal(3.0, 2.0, size=5000)
    m, s = A.cusum_reference(x, None, None, None, "calibration", "log")
    assert abs(m - x.mean()) < 1e-12 and abs(s - x.std(ddof=1)) < 1e-12
    print("calibration reference = calibration-block mean and SD: ok")


def test_restart_rows():
    cols = ["MI_dir_L0.01_weight", "x"]
    X = np.array([[1.0, 0], [1.0, 0], [2.0, 0], [1.0, 0], [3.0, 0]])
    assert A.n_restart_rows(X, cols) == 2
    assert A.n_restart_rows(X[2:], cols) == 0
    assert A.n_restart_rows(X, ["a", "b"]) == 0
    print("restart-row rule: ok")


def test_robust_t2_truncation():
    """Robust distances with full-rank scatter equal the usual Mahalanobis form,
    and directions with zero robust scatter are dropped rather than inverted."""
    C = importlib.import_module("alg4_contamination_mcd")
    Y = rng.normal(size=(400, 6)) @ np.diag([3, 2, 1, 1, 0.5, 0.2])
    fit = C.fit_mcd(Y, None, 0, 1e-8)
    from sklearn.covariance import MinCovDet
    ref = MinCovDet(random_state=0).fit(Y).mahalanobis(Y[:20])
    assert fit["rank"] == 6 and np.allclose(C.robust_t2(Y[:20], fit), ref, rtol=1e-6)
    # exact fit: more than half the rows have the last coordinate fixed at 0
    Y2 = Y.copy(); Y2[:300, 5] = 0.0
    fit2 = C.fit_mcd(Y2, None, 0, 1e-8)
    assert fit2["rank"] == 5 and np.all(np.isfinite(C.robust_t2(Y2, fit2)))
    const, med = C.constant_within(Y2, fit2["support"])
    assert const.tolist() == [False] * 5 + [True] and med[5] == 0.0
    st = C.in_state(Y2, const, med)
    assert st[:300].all() and not st[300:].any()
    print("robust T^2 rank rule and exact-fit case: ok")


def test_throttle_identity():
    n, p = 500, 15
    X = rng.normal(size=(n, p)) * 10 ** rng.uniform(-2, 4, size=p)
    tr = B.StandardizeProject().fit(X); Y = tr.transform(X); m, S = Y.mean(0), np.cov(Y, rowvar=False)
    Xa = X[:40] * 1.7 + 5
    t0 = B.hotelling_t2(tr.transform(Xa), m, S)
    for d in (0.0, 0.3, 0.9, 1.0):
        td = B.hotelling_t2(tr.transform(B.throttle(Xa, X.mean(0), d)), m, S)
        assert np.allclose(td, (1 - d) ** 2 * t0, atol=1e-8)
    print("throttle identity T^2(delta) = (1 - delta)^2 T^2: ok")


def test_vectorized_cusum_matches_scalar():
    """cusum_alarm_summary (many series at once) = cusum_upper (one series)."""
    Z = rng.normal(size=(6, 300)) + np.r_[0, 0.5, 1, 1.5, 2, 3][:, None] * (np.arange(300) >= 150)
    for k, h in ((0.5, 4.0), (1.0, 2.0), (2.0, 0.5)):
        pre, first = A.cusum_alarm_summary(Z, k, h, 150)
        for r in range(Z.shape[0]):
            s = A.summarize_alarms(A.cusum_upper(Z[r], k, h)[1], 150)
            assert pre[r] == s["n_false_alarms_pre_onset"]
            assert (first[r] if first[r] >= 0 else None) == s["detection_delay_rows"]
    print("vectorized CUSUM = scalar CUSUM: ok")


def test_bootstrap_and_arl():
    """Stationary bootstrap draws values of x, keeps runs (mean length close to the
    block length), and the alarm rate agrees with a direct count; calibrate_h
    returns the smallest h (to 0.01) meeting the target, and on i.i.d. N(0,1)
    data the ARL is close to the value known for the CUSUM (Siegmund's
    approximation) at k = 0.5, h = 4: about 336."""
    x = rng.normal(size=1000)
    b = A.stationary_bootstrap(x, 5000, 20, 25.0, np.random.default_rng(1))
    assert b.shape == (20, 5000) and np.isin(b, x).all()
    where = {v: i for i, v in enumerate(x)}
    ib = np.array([[where[v] for v in row] for row in b])
    run_cont = np.mean((ib[:, 1:] - ib[:, :-1]) % len(x) == 1)   # steps that continue a block
    assert abs(run_cont - (1 - 1 / 25)) < 0.03, run_cont
    r = A.alarm_rate(b[:3], 0.5, 3.0)
    direct = sum(A.cusum_upper(b[i], 0.5, 3.0)[1].size for i in range(3)) / b[:3].size
    assert abs(r - direct) < 1e-12
    iid = np.random.default_rng(2).normal(size=(200, 20000))
    arl = 1 / A.alarm_rate(iid, 0.5, 4.0)
    assert 290 < arl < 390, arl
    c = A.calibrate_h(iid[:50, :5000], 0.5, 200.0)
    assert A.alarm_rate(iid[:50, :5000], 0.5, c["h"]) <= 1 / 200
    assert A.alarm_rate(iid[:50, :5000], 0.5, c["h"] - 0.02) > 1 / 200
    print(f"stationary bootstrap, alarm rate, ARL (iid k=0.5,h=4: {arl:.0f}), calibrate_h: ok")


def test_block_intervals():
    idx = A.moving_block_indices(100, 10, 50, np.random.default_rng(3))
    assert idx.shape == (50, 100) and np.all(np.diff(idx[:, :10], axis=1) % 100 == 1)
    flags = np.r_[np.ones(10), np.zeros(90)]
    lo, hi = A.rate_interval(flags, type("a", (), {"ci_block": 10, "ci_replicates": 2000})(),
                             np.random.default_rng(4))
    assert lo <= 0.1 <= hi
    print("moving-block bootstrap intervals: ok")


if __name__ == "__main__":
    for f in [test_t2_limit_size, test_null_moments, test_log_t2_null_moments,
              test_projection_equals_pseudoinverse, test_calibration_reference,
              test_cusum_recursion, test_restart_rows, test_robust_t2_truncation,
              test_throttle_identity, test_vectorized_cusum_matches_scalar,
              test_bootstrap_and_arl, test_block_intervals]:
        f()
    print("all checks passed")
