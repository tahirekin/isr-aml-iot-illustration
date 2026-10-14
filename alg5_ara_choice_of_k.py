"""Algorithm 5 (Section 4.4): the defender's choice of the CUSUM reference value
k against an adaptive attacker, by adversarial risk analysis (ARA).

The defender chooses k; the threshold h is then set for that k as in
Algorithm 3 (by default, the smallest h whose bootstrap in-control ARL reaches
the target, so every k is held to the same false-alarm standard). The attacker
chooses a throttle delta in [0, 1] that moves each attack row toward the benign
training mean,
    x(delta) = (1 - delta) x_attack + delta * mu_benign,
trading effectiveness e = 1 - delta for stealth. Because the standardization
and projection are affine and mu_benign maps to the origin, T^2 of a throttled
row equals (1 - delta)^2 times T^2 of the original row; the code applies the
throttle in raw feature space for a few delta and checks this identity.

Attacker: U_A = w (1 - delta) + (1 - w) D(k, delta), where D is the detection
delay divided by a horizon H, set to 1 if the attack is not detected within the
attack segment, and w is the weight on effectiveness relative to time
undetected.

Defender: U_D = -c_damage (1 - delta) D - c_delay D - c_fa F(k), evaluated at
the attacker's response. The first term is the damage done before detection;
the second (off by default) charges time undetected regardless of
effectiveness; F(k) = min(1, a(k) / fa_cap) is the false-alarm burden of a(k)
alarms per horizon. In the defender's model, which uses only Phase I data, a(k)
is the expected number of in-control alarms per horizon (equal to H / ARL0 for
every k the bootstrap can calibrate) (with --cusum-h-mode fixed, the alarms counted on the
calibration block), and D is computed with the calibration block as the benign
stretch before the attack, averaged over several onset times. The chosen k is then evaluated on the held-out test
sequence, with the alarms actually raised on the test block.

Two attacker models are solved:
  * types (the limiting case): a discrete prior over three attacker types with
    known weights; each type best-responds to the true k. This is a Bayesian
    Stackelberg game.
  * ARA (the main analysis): the defender is uncertain about the attacker's
    preferences and beliefs. A draw picks a type from the prior, then a weight
    w ~ Beta(kappa m_t, kappa (1 - m_t)) around the type's weight m_t, and a
    perceived reference value k~ = k + N(0, sigma_k^2) (clipped to the grid):
    the attacker best-responds to what it believes the detector is. Monte Carlo
    over draws gives the defender's predictive distribution p_D(delta | k) and
    its expected utility, with Monte Carlo standard errors from common random
    numbers across k. k* maximizes the expected utility; k values whose
    expected utility is within two standard errors of k*'s are reported as
    near-optimal.
The script also reports k* over a grid of false-alarm costs c_fa and caps
fa_cap, because the decision depends on how a false alarm is valued relative
to damage.

The delta grid is uniform in log(1 - delta), from 1 - delta = 1 down to
1 - delta = e_min (10 values per decade), plus delta = 1. e_min is 1e-6 unless
the attack rows are so far from the benign data that a smaller value is needed
for the throttled attack to become invisible; it is then lowered until the
strongest throttled row has T^2 at most 1% of the median benign T^2.

This is a discretized approximation to the ARA problem of Appendix B, not a
general solver: one scalar attacker action, a three-type prior with Beta
uncertainty, and grid search. Every weight is an explicit input.

Usage:
    python alg5_ara_choice_of_k.py --data-dir /path/to/Danmini_Doorbell --out-dir results
Outputs:
    results/alg5_ara.json, results/section4_ara_tradeoff.png,
    results/section4_ara_cost.png (Supplementary Figure S1)
Requires: numpy, pandas, scipy, matplotlib
"""
import argparse
# =============================================================================
# SHARED DATA SETUP (identical in every script so that each runs on its own)
# =============================================================================
import glob
import json
import os

# One BLAS thread. FastMCD on these near-degenerate data is sensitive to the
# floating-point order of operations, which changes with the number of BLAS
# threads: the same random_state then gives different MCD fits on different
# machines. Pinning the thread count makes every result reproducible.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import numpy as np                                 # noqa: E402
import pandas as pd                                # noqa: E402
from scipy import stats                            # noqa: E402
from threadpoolctl import threadpool_limits        # noqa: E402

threadpool_limits(1)    # also applies if numpy was imported before this module

ATTACK_FILES = {
    # attack family -> (UCI sub-folder, subtype file stem)
    "gafgyt": ("gafgyt_attacks", "combo"),
    "mirai": ("mirai_attacks", "syn"),
}


def find_csv(data_dir, subfolder, stem):
    """Locate an N-BaIoT CSV under either common distribution layout.

    UCI layout:    <device>/benign_traffic.csv, <device>/gafgyt_attacks/combo.csv
    Kaggle layout: <dir>/1.benign.csv, <dir>/1.gafgyt.combo.csv (device-number prefix)
    """
    candidates = []
    if subfolder is None:
        candidates.append(os.path.join(data_dir, "benign_traffic.csv"))
        candidates += sorted(glob.glob(os.path.join(data_dir, "*benign*.csv")))
    else:
        candidates.append(os.path.join(data_dir, subfolder, f"{stem}.csv"))
        family = subfolder.split("_")[0]
        candidates += sorted(glob.glob(os.path.join(data_dir, f"*{family}*{stem}*.csv")))
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        f"Could not find {'benign traffic' if subfolder is None else subfolder + '/' + stem}"
        f" CSV under {data_dir}. "
        f"Tried: {candidates[:3]}")


def load_matrix(path):
    """Read an N-BaIoT CSV and return a float array of its numeric feature columns."""
    df = pd.read_csv(path)
    df = df.select_dtypes(include=[np.number]).dropna(axis=0, how="any")
    return df.to_numpy(dtype=np.float64), list(df.columns)


def load_device(data_dir, n_train, n_test_benign, n_attack_used, train_offset=None,
                families=("gafgyt", "mirai"), n_calib=1000, skip_restart=True):
    """Split one device's data into the blocks used in Section 4.

    Rows are kept in their recorded (time) order. N-BaIoT's benign file joins
    several captures end to end, and the feature extractor restarts at each
    capture boundary. By default (train_offset=None) the training, calibration,
    and test blocks are taken from the start of the first capture that is long
    enough to hold all three, so that no block crosses a boundary.
      train  : benign rows [offset, offset + n_train)          -> estimate mu, Sigma
      calib  : the next n_calib benign rows (Phase I)          -> in-control behaviour
               of the monitoring statistic (empirical limits, CUSUM reference);
               never used to estimate mu or Sigma
      test   : the next n_test_benign benign rows (Phase II)   -> false-alarm rate
      attack : n_attack_used rows of each attack file           -> appended after test,
               so the attack onset is at row index n_test_benign of each test sequence
      spare  : attack rows after those                          -> used only to
               contaminate the training block (no overlap with the test sequence)

    Each N-BaIoT file is a separate capture, so the Kitsune feature extractor
    restarts at the first row of each attack file: its damped-window statistics
    are computed from a single packet (window weight 1). Such restart rows look
    the same in benign captures (benign restart rows have T^2 of the same size),
    so a detector flagging one is reacting to the restart, not to the attack.
    Unless skip_restart is False, leading attack rows whose longest-window
    weight (MI_dir_L0.01_weight) is <= 1 are dropped before the attack block is
    taken; the number dropped is recorded.
    """
    benign_path = find_csv(data_dir, None, None)
    benign, cols = load_matrix(benign_path)
    width = n_train + n_calib + n_test_benign
    if train_offset is None:               # default: the first capture long enough
        train_offset = first_within_capture_offset(benign, cols, width)
    c0 = train_offset + n_train            # first calibration row
    t0 = c0 + n_calib                      # first test row
    need = t0 + n_test_benign
    if benign.shape[0] < need:
        raise ValueError(f"benign file has {benign.shape[0]} rows; need {need}")
    out = {
        "benign_path": benign_path,
        "columns": cols,
        "train": benign[train_offset:c0],
        "calib": benign[c0:t0],
        "test_benign": benign[t0:need],
        "row_ranges": {"train": [train_offset, c0], "calib": [c0, t0], "test_benign": [t0, need]},
        "benign_capture_restarts_in_window":
            [int(train_offset + r) for r in capture_restarts(benign[train_offset:need], cols)],
        "attack": {}, "attack_spare": {}, "attack_path": {}, "attack_restart_rows_skipped": {},
    }
    out["train_offset"] = int(train_offset)
    rep_ = repeated_rows(benign)
    out["benign_rows_repeating_earlier_rows"] = {
        "in_file": int(rep_.sum()), "in_window": int(rep_[train_offset:need].sum())}
    late = [r for r in out["benign_capture_restarts_in_window"] if r > train_offset]
    if late:
        print(f"WARNING: the benign window crosses a capture boundary (extractor restart at "
              f"row(s) {late}); benign behaviour changes there, which is not an attack. "
              f"Consider another --train-offset.")
    for fam in families:
        sub, stem = ATTACK_FILES[fam]
        path = find_csv(data_dir, sub, stem)
        X, acols = load_matrix(path)
        if acols != cols:
            raise ValueError(f"{path}: columns differ from the benign file")
        if X.shape[0] < n_attack_used:
            raise ValueError(f"{path} has {X.shape[0]} rows; need {n_attack_used}")
        s0 = n_restart_rows(X, cols) if skip_restart else 0
        if X.shape[0] < s0 + n_attack_used:
            raise ValueError(f"{path} has {X.shape[0]} rows; need {s0 + n_attack_used}")
        out["attack"][fam] = X[s0:s0 + n_attack_used]
        out["attack_spare"][fam] = X[s0 + n_attack_used:]
        out["attack_path"][fam] = path
        out["attack_restart_rows_skipped"][fam] = s0
    return out


RESTART_COLUMN = "MI_dir_L0.01_weight"


def n_restart_rows(X, cols):
    """Number of leading rows in the feature extractor's restart state."""
    if RESTART_COLUMN not in cols:
        return 0
    w = X[:, cols.index(RESTART_COLUMN)]
    s = 0
    while s < len(w) and w[s] <= 1.0 + 1e-9:
        s += 1
    return s


def capture_restarts(X, cols):
    """Rows at which every long-window weight is <= 1: the feature extractor has
    restarted, i.e. a new capture begins (N-BaIoT files join several captures)."""
    J = [cols.index(c) for c in cols if c.endswith("L0.01_weight")]
    if not J:
        return np.array([], dtype=int)
    return np.flatnonzero(np.all(X[:, J] <= 1.0 + 1e-9, axis=1))


def repeated_rows(X):
    """Rows identical to an earlier row of X. (The Doorbell's benign file holds
    several captures that begin with the same packets, so about 18% of its rows
    repeat earlier ones.)"""
    _, first = np.unique(X, axis=0, return_index=True)
    mask = np.ones(len(X), dtype=bool); mask[first] = False
    return mask


def first_within_capture_offset(benign, cols, width):
    """Start of the first capture segment with at least `width` rows."""
    edges = [int(b) for b in capture_restarts(benign, cols)]
    if not edges or edges[0] != 0:
        edges = [0] + edges
    edges.append(benign.shape[0])
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a >= width:
            return a
    raise ValueError(f"no capture in the benign file has {width} rows; "
                     f"set --train-offset explicitly")


class StandardizeProject:
    """z-score each feature with training moments, then project onto the
    non-degenerate principal subspace of the training covariance.

    N-BaIoT's 115 Kitsune features differ in raw scale by many orders of magnitude,
    and several are exact or near-exact linear combinations of others, so the
    sample covariance is singular. Hotelling's T^2 is computed in the retained
    p_eff-dimensional subspace, where the training covariance is diagonal.
    """

    def __init__(self, rank_tol=1e-8):
        self.rank_tol = rank_tol

    def fit(self, X):
        self.mu = X.mean(axis=0)
        sd = X.std(axis=0, ddof=1)
        self.keep_cols = sd > 0
        self.sd = sd[self.keep_cols]
        Z = (X[:, self.keep_cols] - self.mu[self.keep_cols]) / self.sd
        S = np.cov(Z, rowvar=False)
        evals, evecs = np.linalg.eigh(S)
        order = np.argsort(evals)[::-1]
        evals, evecs = evals[order], evecs[:, order]
        keep = evals > self.rank_tol * evals[0]
        self.V = evecs[:, keep]
        self.n_constant_cols = int((~self.keep_cols).sum())
        self.p_eff = int(keep.sum())
        return self

    def transform(self, X):
        Z = (X[:, self.keep_cols] - self.mu[self.keep_cols]) / self.sd
        return Z @ self.V   # training mean of Z is 0, so no further centering

    def info(self):
        return {"n_constant_columns_dropped": self.n_constant_cols,
                "p_eff": self.p_eff, "rank_tol": self.rank_tol}


def hotelling_t2(Y, mean, cov):
    """T^2 = (y - mean)' cov^{-1} (y - mean) for each row of Y."""
    D = Y - mean
    L = np.linalg.cholesky(cov)
    W = np.linalg.solve(L, D.T)
    return np.sum(W * W, axis=0)


def t2_control_limit(p, n, alpha):
    """Phase II limit for a new observation (Appendix A):
    p (n+1)(n-1) / (n^2 - n p) * F_{p, n-p, alpha}."""
    if n <= p:
        raise ValueError(f"need n > p (n={n}, p={p})")
    return p * (n + 1) * (n - 1) / (n * n - n * p) * stats.f.ppf(1 - alpha, p, n - p)


def t2_null_moments(p, n):
    """Mean and SD of the Phase II T^2 under multivariate normality,
    T^2 ~ c F(p, n-p) with c = p (n+1)(n-1) / (n (n-p))."""
    d1, d2 = p, n - p
    if d2 <= 4:
        raise ValueError("need n - p > 4 for a finite variance")
    c = p * (n + 1) * (n - 1) / (n * (n - p))
    mean_f = d2 / (d2 - 2)
    var_f = 2 * d2 ** 2 * (d1 + d2 - 2) / (d1 * (d2 - 2) ** 2 * (d2 - 4))
    return c * mean_f, c * np.sqrt(var_f)


def first_index_at_or_after(flags, start):
    idx = np.flatnonzero(flags[start:])
    return int(start + idx[0]) if idx.size else None


def save_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"wrote {path}")


def add_common_args(ap):
    ap.add_argument("--data-dir", required=True, help="device folder, e.g. .../Danmini_Doorbell")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-calib", type=int, default=1000,
                    help="benign rows between training and test used only for calibration")
    ap.add_argument("--n-test-benign", type=int, default=1000)
    ap.add_argument("--n-attack-used", type=int, default=1000)
    ap.add_argument("--train-offset", type=int, default=None,
                    help="first benign row of the training block (default: start of the "
                         "first capture that holds the training, calibration, and test blocks)")
    ap.add_argument("--rank-tol", type=float, default=1e-8,
                    help="eigenvalues below rank_tol * largest are treated as zero")
    ap.add_argument("--keep-restart-rows", action="store_true",
                    help="do not drop the feature-extractor restart rows at the start of "
                         "each attack file")
    ap.add_argument("--families", nargs="+", default=["gafgyt", "mirai"],
                    choices=list(ATTACK_FILES))
    ap.add_argument("--ci-block", type=int, default=50,
                    help="block length of the moving-block bootstrap for intervals")
    ap.add_argument("--ci-replicates", type=int, default=2000)
    ap.add_argument("--ci-seed", type=int, default=0)
    return ap


def moving_block_indices(n, block, n_rep, rng):
    """Row indices for n_rep circular moving-block bootstrap resamples of a
    series of length n. Consecutive rows of N-BaIoT are dependent (the features
    are damped-window statistics), so rows are resampled in blocks."""
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(n, size=(n_rep, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return idx.reshape(n_rep, -1)[:, :n]


def percentile_interval(samples, level=0.95):
    a = (1 - level) / 2
    return [float(np.quantile(samples, a)), float(np.quantile(samples, 1 - a))]


def rate_interval(flags, args, rng):
    """95% moving-block bootstrap interval for the mean of a 0/1 series."""
    idx = moving_block_indices(len(flags), args.ci_block, args.ci_replicates, rng)
    return percentile_interval(np.asarray(flags, float)[idx].mean(axis=1))


def calibrated_rate_interval(stat_calib, stat_test, alpha, args, rng):
    """95% interval for the false-alarm rate on the test block of a limit set at
    the (1 - alpha) quantile of the calibration block, resampling both blocks."""
    ic = moving_block_indices(len(stat_calib), args.ci_block, args.ci_replicates, rng)
    it = moving_block_indices(len(stat_test), args.ci_block, args.ci_replicates, rng)
    lim = np.quantile(np.asarray(stat_calib)[ic], 1 - alpha, axis=1)
    return percentile_interval((np.asarray(stat_test)[it] > lim[:, None]).mean(axis=1))


def setup_meta(args, data, tr):
    return {"data_dir": os.path.abspath(args.data_dir),
            "benign_path": data["benign_path"], "attack_paths": data["attack_path"],
            "n_features_raw": len(data["columns"]), "n_train": args.n_train,
            "n_calib": args.n_calib, "n_test_benign": args.n_test_benign,
            "benign_row_ranges": data["row_ranges"],
            "attack_restart_rows_skipped": data["attack_restart_rows_skipped"],
            "benign_capture_restarts_in_window": data["benign_capture_restarts_in_window"],
            "benign_rows_repeating_earlier_rows": data["benign_rows_repeating_earlier_rows"], "n_attack_used": args.n_attack_used,
            "train_offset": data["train_offset"], "onset_index": args.n_test_benign,
            "preprocessing": tr.info()}
# =============================================================================
# END OF SHARED DATA SETUP
# =============================================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# CUSUM on the T^2 series (shared by Algorithms 3 and 5)
# -----------------------------------------------------------------------------
T2_FLOOR = 1e-12   # T^2 of a row exactly at the benign mean is 0; floor it before the log


def monitoring_statistic(t2, kind):
    """The series the CUSUM accumulates.

    'log' (default): log T^2. T^2 is a scale statistic with a long right tail
        (on N-BaIoT benign traffic its upper percentiles are orders of magnitude
        above its median), so a mean-and-SD standardization of raw T^2 is
        dominated by a few bursts. The log is the usual variance-stabilizing
        transform for scale statistics (as in CUSUM charts on log S^2), and it
        turns the attacker's throttle in Algorithm 5 into an additive shift:
        log T^2(delta) = log T^2 + 2 log(1 - delta).
    'raw': T^2 itself.
    """
    t2 = np.asarray(t2, dtype=float)
    if kind == "log":
        return np.log(np.maximum(t2, T2_FLOOR))
    if kind == "raw":
        return t2
    raise ValueError(kind)


def log_t2_null_moments(p, n):
    """Mean and SD of log T^2 when T^2 ~ c F(p, n-p) (multivariate normality)."""
    c = p * (n + 1) * (n - 1) / (n * (n - p))
    dist = stats.f(p, n - p)
    m1 = dist.expect(np.log)
    m2 = dist.expect(lambda x: np.log(x) ** 2)
    return float(np.log(c) + m1), float(np.sqrt(max(m2 - m1 ** 2, 0.0)))


def cusum_reference(stat_calib, stat_train, p, n, method, kind):
    """In-control mean and SD of the monitoring statistic, used to standardize it.

    'calibration' (default): mean and SD over the Phase I calibration block,
        benign rows that were not used to estimate mu and Sigma. This is the
        classical Phase I / Phase II split and needs no distributional model.
    'theoretical': moments under multivariate normality, T^2 ~ c F(p, n-p).
        On N-BaIoT this model fails badly (see Algorithm 2), so this option is
        kept only for comparison.
    'insample': mean and SD over the training rows' own statistic (optimistic,
        because those rows were used to estimate mu and Sigma).
    """
    if method == "calibration":
        m, s = float(np.mean(stat_calib)), float(np.std(stat_calib, ddof=1))
    elif method == "insample":
        m, s = float(np.mean(stat_train)), float(np.std(stat_train, ddof=1))
    elif method == "theoretical":
        m, s = t2_null_moments(p, n) if kind == "raw" else log_t2_null_moments(p, n)
    else:
        raise ValueError(method)
    return float(m), float(s)


def cusum_upper(z, k, h, reset=True):
    """Page's one-sided (upper) CUSUM on a standardized series z (Appendix A):
        S_i = max(0, S_{i-1} + z_i - k),  signal when S_i > h.
    One-sided because any shift in the mean of x raises E[T^2]; a low T^2 means
    the row is closer to the benign mean than usual, which is not evidence of an
    attack. After a signal S restarts at 0, as when an alarm is investigated and
    cleared. Returns (S, alarm indices)."""
    s = np.zeros(len(z)); alarms = []
    a = 0.0
    for i, zi in enumerate(z):
        a = max(0.0, a + zi - k)
        s[i] = a
        if a > h:
            alarms.append(i)
            if reset:
                a = 0.0
    return s, np.array(alarms, dtype=int)


def summarize_alarms(alarms, onset):
    pre = alarms[alarms < onset]; post = alarms[alarms >= onset]
    return {"false_alarm_before_onset": bool(pre.size > 0),
            "n_false_alarms_pre_onset": int(pre.size),
            "detected": bool(post.size > 0),
            "detection_delay_rows": int(post[0] - onset) if post.size else None}


def add_cusum_args(ap):
    ap.add_argument("--cusum-h-sigmas", type=float, default=5.0,
                    help="CUSUM threshold h, in in-control SDs of the monitoring statistic")
    ap.add_argument("--cusum-statistic", choices=["log", "raw"], default="log",
                    help="monitor log T^2 (default) or raw T^2")
    ap.add_argument("--cusum-reference", choices=["calibration", "theoretical", "insample"],
                    default="calibration",
                    help="where the in-control mean and SD come from")
    return ap


def standardized_stat(t2, kind, m0, s0):
    return (monitoring_statistic(t2, kind) - m0) / s0


# -----------------------------------------------------------------------------
# Choosing h by the in-control average run length (Phase I bootstrap). The ARL
# is the mean number of observations between false alarms of the chart with
# reset, estimated from stationary-bootstrap resamples of the standardized
# Phase I series.
# -----------------------------------------------------------------------------
def stationary_bootstrap(x, length, n_rep, mean_block, rng):
    """n_rep series of the given length resampled from x by the stationary
    bootstrap (Politis and Romano, 1994): blocks of geometric length with mean
    mean_block, wrapping around the end of x. Keeps the short-range dependence
    and the bursts of the Phase I series."""
    n = len(x)
    starts = rng.random((n_rep, length)) < 1.0 / mean_block
    starts[:, 0] = True
    newpos = rng.integers(n, size=(n_rep, length))
    t = np.broadcast_to(np.arange(length), (n_rep, length))
    last = np.maximum.accumulate(np.where(starts, t, 0), axis=1)
    return np.asarray(x)[(np.take_along_axis(newpos, last, axis=1) + (t - last)) % n]


def alarm_rate(boot, k, h):
    """In-control false alarms per observation of the upper CUSUM with reset,
    pooled over the bootstrap resamples (rows of boot). Its reciprocal is the
    average run length between false alarms."""
    S = np.zeros(boot.shape[0]); count = 0
    for t in range(boot.shape[1]):
        S = np.maximum(0.0, S + boot[:, t] - k)
        al = S > h
        count += int(al.sum())
        S[al] = 0.0
    return count / boot.size


def calibrate_h(boot, k, arl0, h_ref=None, h_max=80.0, tol=0.01):
    """Smallest h (to within tol) whose bootstrap in-control false-alarm rate is
    at most 1 / arl0, found by bisection (the rate is non-increasing in h). The
    rate is estimated from boot.size resampled observations, so a rate of 0
    means no false alarm in that many observations."""
    target = 1.0 / arl0
    while alarm_rate(boot, k, h_max) > target:      # small k need a large h
        h_max *= 2.0
        if h_max > 1e4:
            raise ValueError(f"no h up to 1e4 reaches ARL0 = {arl0} at k = {k}")
    lo, hi = 0.0, h_max
    if alarm_rate(boot, k, lo) <= target:
        hi = lo
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if alarm_rate(boot, k, mid) <= target:
            hi = mid
        else:
            lo = mid
    h = float(np.ceil(hi / tol) * tol)
    rate = alarm_rate(boot, k, h)
    out = {"k": float(k), "h": round(h, 4), "false_alarm_rate": rate,
           "arl0_estimate": (1.0 / rate) if rate > 0 else None,
           "n_bootstrap_observations": int(boot.size)}
    if h_ref is not None:
        r = alarm_rate(boot, k, h_ref)
        out["at_reference_h"] = {"h": float(h_ref), "false_alarm_rate": r,
                                 "arl0_estimate": (1.0 / r) if r > 0 else None}
    return out


def cusum_alarm_summary(Zm, k, h, onset):
    """Upper CUSUM with reset, run on each row of Zm at once. Returns, per row,
    the number of alarms before onset and the delay of the first alarm at or
    after onset (-1 if none). Same recursion as cusum_upper."""
    n, T = Zm.shape
    S = np.zeros(n); pre = np.zeros(n, dtype=int); first = np.full(n, -1)
    for t in range(T):
        S = np.maximum(0.0, S + Zm[:, t] - k)
        al = S > h
        if t < onset:
            pre += al
        else:
            new = al & (first < 0)
            first[new] = t - onset
        S[al] = 0.0
    return pre, first


def add_h_args(ap):
    ap.add_argument("--cusum-h-mode", choices=["arl", "fixed"], default="arl",
                    help="choose h for a target in-control ARL (default) or fix it")
    ap.add_argument("--arl0", type=float, default=1000.0,
                    help="target in-control average run length, in observations")
    ap.add_argument("--boot-block", type=float, default=50.0,
                    help="mean block length of the stationary bootstrap")
    ap.add_argument("--boot-replicates", type=int, default=400)
    ap.add_argument("--boot-length", type=int, default=20000)
    ap.add_argument("--boot-seed", type=int, default=0)
    return ap


def phase1_bootstrap(z_calib, args):
    rng = np.random.default_rng(args.boot_seed)
    return stationary_bootstrap(z_calib, args.boot_length, args.boot_replicates,
                                args.boot_block, rng)

DEFAULT_TYPES = {
    "aggressive": {"prior": 0.3, "w_eff": 0.8, "w_delay": 0.2},
    "balanced":   {"prior": 0.4, "w_eff": 0.5, "w_delay": 0.5},
    "stealthy":   {"prior": 0.3, "w_eff": 0.2, "w_delay": 0.8},
}


def throttle(X_attack, mu_benign, delta):
    return (1.0 - delta) * X_attack + delta * mu_benign


C_FA_GRID = np.logspace(-5, 1, 25)
FA_CAPS = [1.0, 5.0, 20.0]


def solve_types(types, eff, D_model, D_test, F_model, F_test, args):
    """Each type best-responds to the true k (ties: smallest delta)."""
    rows = np.arange(D_model.shape[0])
    U, U_t, dmg, dmg_t, best = 0.0, 0.0, 0.0, 0.0, {}
    for name, t in types.items():
        w = t["w_eff"] / (t["w_eff"] + t["w_delay"])
        l = np.argmax(w * eff[None, :] + (1 - w) * D_model, axis=1)
        d_m, d_t = D_model[rows, l], D_test[rows, l]
        dm = args.c_damage * eff[l] * d_m + args.c_delay * d_m
        dt = args.c_damage * eff[l] * d_t + args.c_delay * d_t
        dmg = dmg + t["prior"] * dm; dmg_t = dmg_t + t["prior"] * dt
        best[name] = {"delta_star": (1 - eff[l]).tolist(), "effectiveness_star": eff[l].tolist(),
                      "detected": (d_m < 1.0).tolist(), "normalized_delay": d_m.tolist()}
    U = -dmg - args.c_fa * np.minimum(1.0, F_model / args.fa_cap)
    U_t = -dmg_t - args.c_fa * np.minimum(1.0, F_test / args.fa_cap)
    return U, U_t, dmg, best


def best_index(U, tol):
    """Index of the best k, treating values within tol of the maximum as ties and
    breaking ties toward the smallest k. Without the tolerance, k* in a window where
    expected damage is zero at every k would be decided by floating-point rounding."""
    return int(np.flatnonzero(U >= U.max() - tol)[0])


def solve_ara(types, eff, D_model, D_test, F_model, F_test, k_grid, args):
    """Monte Carlo over the defender's uncertainty about the attacker."""
    rng = np.random.default_rng(args.ara_seed)
    names = list(types)
    prior = np.array([types[t]["prior"] for t in names])
    m = np.array([types[t]["w_eff"] / (types[t]["w_eff"] + types[t]["w_delay"]) for t in names])
    N, nk = args.ara_draws, D_model.shape[0]
    tix = rng.choice(len(names), size=N, p=prior)
    w = rng.beta(args.ara_kappa * m[tix], args.ara_kappa * (1 - m[tix]))
    pert = rng.normal(0.0, args.ara_sigma_k, size=N)
    dmg = np.empty((nk, N)); dmg_t = np.empty((nk, N))
    resp = {"p_full_strength": [], "p_no_attack": [], "p_throttled": [],
            "median_effectiveness_if_throttled": [], "p_detected": []}
    for i in range(nk):
        kp = np.clip(k_grid[i] + pert, k_grid[0], k_grid[-1])
        j = np.abs(k_grid[None, :] - kp[:, None]).argmin(axis=1)    # perceived k on the grid
        UA = w[:, None] * eff[None, :] + (1 - w[:, None]) * D_model[j]
        l = UA.argmax(axis=1)                                     # ties: smallest delta
        d_m, d_t = D_model[i, l], D_test[i, l]
        dmg[i] = args.c_damage * eff[l] * d_m + args.c_delay * d_m
        dmg_t[i] = args.c_damage * eff[l] * d_t + args.c_delay * d_t
        e_l = eff[l]
        thr = (e_l > 0) & (e_l < 1)
        resp["p_full_strength"].append(float(np.mean(e_l == 1.0)))
        resp["p_no_attack"].append(float(np.mean(e_l == 0.0)))
        resp["p_throttled"].append(float(np.mean(thr)))
        resp["median_effectiveness_if_throttled"].append(float(np.median(e_l[thr])) if thr.any() else None)
        resp["p_detected"].append(float(np.mean(d_m < 1.0)))   # detected after at least one onset
    fa = args.c_fa * np.minimum(1.0, F_model / args.fa_cap)
    U_draw = -dmg - fa[:, None]
    U = U_draw.mean(axis=1)
    U_t = -dmg_t.mean(axis=1) - args.c_fa * np.minimum(1.0, F_test / args.fa_cap)
    i_star = best_index(U, args.ara_tol)
    diff = U_draw[i_star][None, :] - U_draw
    se_diff = diff.std(axis=1, ddof=1) / np.sqrt(N)
    near = [float(k_grid[i]) for i in range(nk)
            if i == i_star or diff[i].mean() <= max(2 * se_diff[i], args.ara_tol)]
    return U, U_t, dmg.mean(axis=1), U_draw.std(axis=1, ddof=1) / np.sqrt(N), near, resp


def k_star_by_cost(dmg_mean, F_model, k_grid, tol):
    out = {}
    for cap in FA_CAPS:
        out[f"{cap:g}"] = [float(k_grid[best_index(-dmg_mean - c * np.minimum(1.0, F_model / cap), tol)])
                           for c in C_FA_GRID]
    return out


def delay_matrices(k_grid, h_k, z_cal, z_benign, za, H, n_onsets):
    """Normalized delays D[i, l] for each (k_i, h_i) and effectiveness value l, with
    the calibration block (defender's model) or the test block (realized) as the
    benign stretch before the attack. The CUSUM state at the onset depends on the
    benign traffic just before it, so the attack is started after n_onsets
    different lengths of benign prefix (evenly spaced from 100 rows to the whole
    block) and the normalized delays are averaged over these onsets."""
    D_model = np.zeros((len(k_grid), za.shape[0])); D_test = np.zeros_like(D_model)
    for pre_full, D in ((z_cal, D_model), (z_benign, D_test)):
        onsets = np.unique(np.linspace(100, len(pre_full), n_onsets).astype(int))
        for i, (k, h) in enumerate(zip(k_grid, h_k)):
            for t0 in onsets:
                pre_z = pre_full[:t0]
                Zm = np.hstack([np.broadcast_to(pre_z, (za.shape[0], t0)), za])
                _, first = cusum_alarm_summary(Zm, k, h, t0)
                D[i] += np.where(first < 0, 1.0, np.minimum(first, H) / H)
            D[i] /= len(onsets)
    return D_model, D_test


def analyze_design(name, k_grid, h_k, F_model, F_test, eff, za_by_fam, z_cal, z_benign,
                   types, H, args, extra):
    """Solve the types and ARA problems for one threshold design."""
    idx = lambda kv: int(np.argmin(np.abs(k_grid - kv)))
    out = {"k_grid": k_grid.tolist(), "h_by_k": h_k.tolist(),
           "false_alarms_per_horizon_model": F_model.tolist(),
           "n_false_alarms_test_block": F_test.astype(int).tolist(), **extra, "by_family": {}}
    for fam, za in za_by_fam.items():
        D_model, D_test = delay_matrices(k_grid, h_k, z_cal, z_benign, za, H, args.ara_onsets)
        boundary = [float(eff[np.flatnonzero(D_model[i] >= 1.0)].max()) for i in range(len(k_grid))]
        U_ty, U_ty_t, dmg_ty, best = solve_types(types, eff, D_model, D_test, F_model, F_test, args)
        U, U_t, dmg, se, near, resp = solve_ara(types, eff, D_model, D_test, F_model, F_test,
                                                k_grid, args)
        i_ty, i_ara = best_index(U_ty, args.ara_tol), best_index(U, args.ara_tol)
        fixed = lambda Uv, Ut_, dm: {
            lab: {"k": float(k_grid[i]), "exp_utility": float(Uv[i]),
                  "exp_utility_test": float(Ut_[i]), "expected_damage": float(dm[i]),
                  "n_false_alarms_test_block": int(F_test[i])}
            for lab, i in (("k_tuned", idx(args.k_tuned)), ("k_conservative", idx(args.k_conservative)))}
        out["by_family"][fam] = {
            "largest_undetected_effectiveness": boundary,
            "full_strength_normalized_delay": D_model[:, 0].tolist(),
            "full_strength_normalized_delay_test": D_test[:, 0].tolist(),
            "types": {"best_response": best,
                      "expected_defender_utility": U_ty.tolist(),
                      "expected_defender_utility_on_test": U_ty_t.tolist(),
                      "expected_damage": dmg_ty.tolist(),
                      "k_star": float(k_grid[i_ty]), "k_star_exp_utility": float(U_ty[i_ty]),
                      "k_star_exp_utility_test": float(U_ty_t[i_ty]),
                      "k_star_expected_damage": float(dmg_ty[i_ty]), **fixed(U_ty, U_ty_t, dmg_ty)},
            "ara": {"expected_defender_utility": U.tolist(), "mc_se": se.tolist(),
                    "expected_defender_utility_on_test": U_t.tolist(),
                    "expected_damage": dmg.tolist(),
                    "k_star": float(k_grid[i_ara]), "k_star_exp_utility": float(U[i_ara]),
                    "k_star_exp_utility_test": float(U_t[i_ara]),
                    "k_star_expected_damage": float(dmg[i_ara]),
                    "k_star_n_false_alarms_test_block": int(F_test[i_ara]),
                    "near_optimal_k": near,
                    "indifferent_across_k": bool(U.max() - U.min() <= args.ara_tol),
                    "attacker_response_by_k": resp, **fixed(U, U_t, dmg)},
            "cost_sensitivity": {"k_star_types": k_star_by_cost(dmg_ty, F_model, k_grid, args.ara_tol),
                                 "k_star_ara": k_star_by_cost(dmg, F_model, k_grid, args.ara_tol)}}
        a = out["by_family"][fam]["ara"]
        print(f"  [{name}] {fam}: ARA k* = {a['k_star']:g} (E[U_D] = {a['k_star_exp_utility']:.5f}, "
              f"damage {a['k_star_expected_damage']:.5f}); near-optimal k {near}; "
              f"types k* = {out['by_family'][fam]['types']['k_star']:g}")
    return out


def run(args):
    data = load_device(args.data_dir, args.n_train, args.n_test_benign,
                       args.n_attack_used, args.train_offset, args.families,
                       n_calib=args.n_calib, skip_restart=not args.keep_restart_rows)
    tr = StandardizeProject(args.rank_tol).fit(data["train"])
    Ytr = tr.transform(data["train"])
    mean, cov = Ytr.mean(axis=0), np.cov(Ytr, rowvar=False)
    n, p = Ytr.shape
    kind = args.cusum_statistic
    stat_tr = monitoring_statistic(hotelling_t2(Ytr, mean, cov), kind)
    t2_cal_raw = hotelling_t2(tr.transform(data["calib"]), mean, cov)
    stat_cal = monitoring_statistic(t2_cal_raw, kind)
    m0, s0 = cusum_reference(stat_cal, stat_tr, p, n, args.cusum_reference, kind)
    z_cal = (stat_cal - m0) / s0
    H = args.horizon or args.n_attack_used
    types = json.loads(args.attacker_types) if args.attacker_types else DEFAULT_TYPES
    types = {name: dict(t) for name, t in types.items()}
    tot = sum(t["prior"] for t in types.values())
    for t in types.values():
        t["prior"] = t["prior"] / tot
    k_grid = np.linspace(args.k_min, args.k_max, args.k_n)
    for kf in (args.k_tuned, args.k_conservative):          # make sure both fixed
        if not np.any(np.isclose(k_grid, kf)):               # choices are evaluated
            k_grid = np.sort(np.append(k_grid, kf))
    # effectiveness grid: must reach attacks no detector could see
    mu_raw = data["train"].mean(axis=0)
    t2_att = {f: hotelling_t2(tr.transform(data["attack"][f]), mean, cov) for f in args.families}
    t2_att_max = max(float(v.max()) for v in t2_att.values())
    e_min = min(args.e_min, 0.1 * np.sqrt(np.median(t2_cal_raw) / t2_att_max))
    e_min = 10.0 ** np.floor(np.log10(e_min))
    n_e = int(round(args.e_per_decade * -np.log10(e_min))) + 1
    eff = np.r_[np.logspace(0.0, np.log10(e_min), n_e), 0.0]   # descending: ties -> smallest delta
    d_grid = 1.0 - eff
    for f in args.families:   # check T^2(x(delta)) = (1 - delta)^2 T^2(x) on raw-space throttling
        for d in (d_grid[0], d_grid[len(d_grid) // 2], d_grid[-2]):
            t2_chk = hotelling_t2(tr.transform(throttle(data["attack"][f], mu_raw, d)), mean, cov)
            assert np.allclose(t2_chk, (1 - d) ** 2 * t2_att[f], rtol=1e-5, atol=1e-8), d
    za_by_fam = {f: standardized_stat(np.outer(eff ** 2, t2_att[f]), kind, m0, s0)
                 for f in args.families}
    z_benign = standardized_stat(hotelling_t2(tr.transform(data["test_benign"]), mean, cov),
                                 kind, m0, s0)
    test_alarms = lambda kg, hk: np.array(
        [cusum_alarm_summary(z_benign[None, :], k, h, len(z_benign))[0][0] for k, h in zip(kg, hk)],
        float)
    designs = {}
    # Design 1 (main): h(k) for a common in-control ARL. Only k values for which the
    # Phase I bootstrap can certify that ARL are kept: if no resampled observation
    # ever raises an alarm, the false-alarm rate lies beyond what 1,000 Phase I
    # observations can estimate. On the kept k values the false-alarm burden is
    # the design value H / ARL0, the same for every k.
    boot = phase1_bootstrap(z_cal, args)
    cal = [calibrate_h(boot, k, args.arl0) for k in k_grid]
    keep = np.array([c["false_alarm_rate"] > 0 for c in cal])
    kg_a = k_grid[keep]; h_a = np.array([c["h"] for c in cal])[keep]
    F_a = np.full(len(kg_a), H / args.arl0)
    print("== ARL-matched design")
    designs["arl"] = analyze_design(
        "arl", kg_a, h_a, F_a, test_alarms(kg_a, h_a), eff, za_by_fam, z_cal, z_benign,
        types, H, args,
        {"arl0_target": args.arl0,
         "bootstrap_false_alarm_rate_by_k": [c["false_alarm_rate"] for c in cal if c["false_alarm_rate"] > 0],
         "k_excluded_no_bootstrap_alarm": [float(k) for k, c in zip(k_grid, cal) if c["false_alarm_rate"] == 0],
         "max_standardized_phase1_value": float(z_cal.max())})
    # Design 2 (comparison): h fixed for every k; false alarms counted on the
    # calibration block, so the false-alarm burden varies with k.
    h_f = np.full(len(k_grid), args.cusum_h_sigmas)
    F_f = np.array([cusum_upper(z_cal, k, h)[1].size for k, h in zip(k_grid, h_f)], float)
    print("== fixed-h design")
    designs["fixed_h"] = analyze_design(
        "fixed_h", k_grid, h_f, F_f, test_alarms(k_grid, h_f), eff, za_by_fam, z_cal, z_benign,
        types, H, args, {"h": args.cusum_h_sigmas})
    res = {"meta": {**setup_meta(args, data, tr),
                    "k_grid_spec": {"n": int(len(k_grid)), "min": float(k_grid.min()),
                                    "max": float(k_grid.max())},
                    "delta_grid_spec": {"n": int(len(d_grid)), "spacing": "log in 1 - delta",
                                        "min_effectiveness_nonzero": float(e_min),
                                        "values_per_decade": args.e_per_decade,
                                        "includes_delta_1": True},
                    "cusum_reference": {"method": args.cusum_reference, "statistic": kind,
                                        "mean": m0, "sd": s0, "one_sided": True},
                    "horizon_rows": H,
                    "defender_costs": {"c_delay": args.c_delay, "c_fa": args.c_fa,
                                       "c_damage": args.c_damage, "fa_cap": args.fa_cap},
                    "ara_settings": {"kappa": args.ara_kappa, "sigma_k": args.ara_sigma_k,
                                     "draws": args.ara_draws, "seed": args.ara_seed,
                                     "onsets": args.ara_onsets},
                    "bootstrap": {"method": "stationary", "mean_block": args.boot_block,
                                  "replicates": args.boot_replicates,
                                  "length": args.boot_length, "seed": args.boot_seed},
                    "attacker_types": types,
                    "cost_grid": {"c_fa": C_FA_GRID.tolist(), "fa_caps": FA_CAPS}},
           "designs": designs}
    # figures: ARA under the ARL-matched design (section4_ara_tradeoff.png), and
    # k* against the false-alarm cost under the fixed-h design (section4_ara_cost.png)
    fig, axes = plt.subplots(2, len(args.families), figsize=(3.5 * len(args.families), 5.2),
                             squeeze=False)
    da, df = designs["arl"], designs["fixed_h"]
    kga = np.array(da["k_grid"])
    for j, fam in enumerate(args.families):
        a = da["by_family"][fam]["ara"]; t = da["by_family"][fam]["types"]
        bf = da["by_family"][fam]
        ax = axes[0, j]
        floor = 1e-5
        fs = np.array(bf["full_strength_normalized_delay"])
        ax.semilogy(kga, np.maximum(fs, floor), "o-", ms=2.5, lw=1,
                    label="full-strength attack: delay / H")
        ax.semilogy(kga, bf["largest_undetected_effectiveness"], "s--", ms=2.5, lw=1,
                    label="throttled attack: largest undetected e")
        ax.axhline(floor, color="0.7", lw=0.5, ls=":")
        ax.set_ylim(floor / 2, 0.05); ax.set_ylabel("damage per attacker")
        ax.set_title(f"{fam.capitalize()}: common ARL$_0$ (dotted: caught at once)",
                     fontsize=7.5, loc="left")
        ax.legend(frameon=False, fontsize=6, loc="upper center")
        ax = axes[1, j]
        dm, se = np.array(a["expected_damage"]), np.array(a["mc_se"])
        ax.plot(kga, dm, "k-", lw=1, label="ARA")
        ax.fill_between(kga, dm - 2 * se, dm + 2 * se, color="k", alpha=0.15, lw=0)
        ax.plot(kga, t["expected_damage"], color="C2", ls=":", lw=1, label="types (known weights)")
        ax.axvline(a["k_star"], color="g", lw=0.8, label=f"ARA $k^*$ = {a['k_star']:g}")
        ax.set_yscale("log"); ax.set_ylabel("expected damage")
        ax.legend(fontsize=6, loc="upper center", framealpha=0.95, edgecolor="none")
    for ax in axes.ravel():
        ax.set_xlabel("CUSUM reference value k")
    fig.tight_layout(); os.makedirs(args.out_dir, exist_ok=True)
    fig.savefig(os.path.join(args.out_dir, "section4_ara_tradeoff.png"), dpi=200); plt.close(fig)
    fig, axes = plt.subplots(1, len(args.families), figsize=(3.5 * len(args.families), 2.6),
                             squeeze=False)
    for j, fam in enumerate(args.families):
        ax = axes[0, j]
        cs = df["by_family"][fam]["cost_sensitivity"]["k_star_ara"]
        for cap, ls in zip(FA_CAPS, [":", "-", "--"]):
            ax.semilogx(C_FA_GRID, cs[f"{cap:g}"], ls=ls, drawstyle="steps-mid", lw=1,
                        label=f"fa_cap = {cap:g}")
        ax.axvline(args.c_fa, color="0.6", lw=0.6)
        ax.set_title(f"{fam.capitalize()}: fixed h = {args.cusum_h_sigmas:g}", fontsize=8, loc="left")
        ax.set_xlabel("false-alarm cost $c_{fa}$"); ax.set_ylabel("ARA $k^*$")
        ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "section4_ara_cost.png"), dpi=200); plt.close(fig)
    return res


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__,
                         formatter_class=argparse.RawDescriptionHelpFormatter))
    add_cusum_args(ap)
    add_h_args(ap)
    ap.add_argument("--k-tuned", type=float, default=1.0)
    ap.add_argument("--k-conservative", type=float, default=3.0)
    ap.add_argument("--k-min", type=float, default=0.0)
    ap.add_argument("--k-max", type=float, default=6.0)
    ap.add_argument("--k-n", type=int, default=25)
    ap.add_argument("--e-per-decade", type=int, default=10,
                    help="log-spaced effectiveness values 1 - delta per decade")
    ap.add_argument("--e-min", type=float, default=1e-6,
                    help="smallest nonzero effectiveness 1 - delta on the grid")
    ap.add_argument("--horizon", type=int, default=None, help="default: n_attack_used")
    ap.add_argument("--ara-kappa", type=float, default=20.0,
                    help="concentration of the Beta distribution of each type's weight")
    ap.add_argument("--ara-sigma-k", type=float, default=0.5,
                    help="SD of the attacker's error in perceiving k")
    ap.add_argument("--ara-draws", type=int, default=4000)
    ap.add_argument("--ara-onsets", type=int, default=20,
                    help="benign-prefix lengths over which attack delays are averaged")
    ap.add_argument("--ara-seed", type=int, default=0)
    ap.add_argument("--ara-tol", type=float, default=1e-9,
                    help="differences in expected utility below this are treated as ties "
                         "(broken toward the smallest k)")
    ap.add_argument("--c-delay", type=float, default=0.0,
                    help="cost per unit of normalized time undetected, regardless of effectiveness")
    ap.add_argument("--c-fa", type=float, default=1.0)
    ap.add_argument("--c-damage", type=float, default=1.0,
                    help="cost of damage: effectiveness times normalized time undetected")
    ap.add_argument("--fa-cap", type=float, default=5.0,
                    help="pre-onset false alarms at which the false-alarm burden reaches 1")
    ap.add_argument("--attacker-types", default=None,
                    help='JSON, e.g. \'{"aggressive": {"prior": 0.3, "w_eff": 0.8, "w_delay": 0.2}, ...}\'')
    args = ap.parse_args()
    save_json(run(args), os.path.join(args.out_dir, "alg5_ara.json"))


if __name__ == "__main__":
    main()
