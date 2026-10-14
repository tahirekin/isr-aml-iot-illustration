"""Algorithm 3 (Section 4.2.2): CUSUM on the naive T^2 series.

Applies Page's one-sided (upper) CUSUM to the standardized naive monitoring
statistic (log T^2 by default) of each test sequence, at a sensitive ("tuned")
and a conservative reference value k. The in-control mean and SD come from the
Phase I calibration block of benign rows, which is used neither to estimate mu
and Sigma nor to measure false alarms.

The threshold h is chosen, for each k, by the classical criterion: the smallest
h whose in-control average run length (ARL0) is at least a target (default
1,000 observations). ARL0 is the mean number of observations between false
alarms, estimated from stationary-bootstrap resamples of the standardized
Phase I series, which keep its bursts and short-range dependence. With --cusum-h-mode fixed, h is instead fixed at
--cusum-h-sigmas; the fixed-h results (h = 5) are always reported as well, for
comparison. In units of the in-control SD of log T^2, k is half the shift the
chart is tuned to detect, so k corresponds to multiplying T^2 by
exp(2 k sd_log): the JSON reports this factor.

Reports, per attack family and per k, the number of alarms before the known
onset and the delay to the first alarm at or after onset, for the recorded
attack and for weakened versions x(e) = e x + (1 - e) mu at a few effectiveness
levels e (the throttle Algorithm 5 lets the attacker choose). Delays are in
observations; each N-BaIoT row describes one packet.

Usage:
    python alg3_cusum_t2.py --data-dir /path/to/Danmini_Doorbell --out-dir results
Outputs:
    results/alg3_cusum_t2.json, results/section4_cusum.png
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

EFFECTIVENESS_PROBE = [1.0, 1e-1, 1e-2, 1e-3, 3e-4, 1e-4]


def cusum_results(z_benign, t2_att, t2_att_e, kind, m0, s0, k, h, onset):
    """Alarms before onset and delays (recorded and weakened attacks) at (k, h)."""
    S, alarms = cusum_upper(np.r_[z_benign, standardized_stat(t2_att, kind, m0, s0)], k, h)
    out = {"k": k, "h": h, **summarize_alarms(alarms, onset), "delay_by_effectiveness": {}}
    for e, t2_e in t2_att_e.items():
        _, al = cusum_upper(np.r_[z_benign, standardized_stat(t2_e, kind, m0, s0)], k, h)
        out["delay_by_effectiveness"][e] = summarize_alarms(al, onset)["detection_delay_rows"]
    return out, S


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
    stat_cal = monitoring_statistic(hotelling_t2(tr.transform(data["calib"]), mean, cov), kind)
    m0, s0 = cusum_reference(stat_cal, stat_tr, p, n, args.cusum_reference, kind)
    z_cal = (stat_cal - m0) / s0
    onset = args.n_test_benign
    ks = {"tuned": args.cusum_k_tuned, "conservative": args.cusum_k_conservative}
    h_fixed = args.cusum_h_sigmas
    boot = phase1_bootstrap(z_cal, args)
    design = {}
    for name, k in ks.items():
        cal = calibrate_h(boot, k, args.arl0, h_ref=h_fixed)
        h_used = cal["h"] if args.cusum_h_mode == "arl" else h_fixed
        design[name] = {**cal, "h_used": h_used,
                        "targeted_t2_factor": float(np.exp(2 * k * s0)) if kind == "log" else None}
    res = {"meta": setup_meta(args, data, tr),
           "CUSUM_K_CHOICES": ks, "CUSUM_H_IN_SIGMAS": h_fixed,
           "cusum_h_design": {"mode": args.cusum_h_mode, "arl0_target": args.arl0,
                              "bootstrap": {"method": "stationary", "mean_block": args.boot_block,
                                            "replicates": args.boot_replicates,
                                            "length": args.boot_length, "seed": args.boot_seed},
                              "by_k": design},
           "cusum_reference": {"method": args.cusum_reference, "statistic": kind,
                               "mean": m0, "sd": s0, "one_sided": True},
           "per_attack": {}}
    mu_raw = data["train"].mean(axis=0)
    t2_benign = hotelling_t2(tr.transform(data["test_benign"]), mean, cov)
    z_benign = standardized_stat(t2_benign, kind, m0, s0)
    # Diagnostic: the textbook alternative (two-sided CUSUM on raw T^2 standardized
    # by its Gaussian moments) on the same benign test rows, alarms by arm, at h = 5.
    mt, st = t2_null_moments(p, n)
    zt = (t2_benign - mt) / st
    diag = {"statistic": "raw T^2", "reference": "theoretical (Gaussian)",
            "mean": float(mt), "sd": float(st), "h": h_fixed}
    for name, k in ks.items():
        _, up = cusum_upper(zt, k, h_fixed)
        _, lo = cusum_upper(-zt, k, h_fixed)
        diag[name] = {"k": k, "benign_alarms_upper_arm": int(up.size),
                      "benign_alarms_lower_arm": int(lo.size)}
    res["diagnostic_two_sided_gaussian_reference"] = diag
    fig, axes = plt.subplots(len(args.families), 2, figsize=(7.5, 2.4 * len(args.families)),
                             gridspec_kw={"width_ratios": [2.2, 1]}, squeeze=False)
    cols = {"tuned": "C0", "conservative": "C1"}
    for row, fam in enumerate(args.families):
        Xa = data["attack"][fam]
        t2_att = hotelling_t2(tr.transform(Xa), mean, cov)
        t2_att_e = {f"{e:g}": hotelling_t2(tr.transform(e * Xa + (1 - e) * mu_raw), mean, cov)
                    for e in EFFECTIVENESS_PROBE}
        out, out_fixed = {}, {}
        ax = axes[row, 0]
        top_s = 0.0
        for name, k in ks.items():
            h = design[name]["h_used"]
            out[name], S = cusum_results(z_benign, t2_att, t2_att_e, kind, m0, s0, k, h, onset)
            out_fixed[name], _ = cusum_results(z_benign, t2_att, t2_att_e, kind, m0, s0, k,
                                               h_fixed, onset)
            ax.plot(S, lw=0.7, color=cols[name], label=f"$S^+$, k = {k} ({name})")
            ax.axhline(h, color=cols[name], lw=0.8, ls="--", label=f"h = {h:g} (k = {k})")
            top_s = max(top_s, h)
            print(f"{fam} k={k} h={h:g}: false alarms before onset={out[name]['n_false_alarms_pre_onset']}, "
                  f"delay={out[name]['detection_delay_rows']}, "
                  f"delay by effectiveness={out[name]['delay_by_effectiveness']}")
        res["per_attack"][fam] = {"n_attack_used": int(len(Xa)), "onset_index": onset,
                                  "cusum_on_naive_t2": out,
                                  "cusum_on_naive_t2_fixed_h": out_fixed}
        ax.axvline(onset, color="b", lw=0.8, label="attack onset")
        ax.set_ylim(0, 2.0 * top_s); ax.set_ylabel("$S^+_i$")
        ax.set_title(fam.capitalize(), fontsize=9, loc="left")
        ax2 = axes[row, 1]
        top = max([v for nm in ks for v in out[nm]["delay_by_effectiveness"].values()
                   if v is not None] + [10]) * 1.15
        for (name, k), mk in zip(ks.items(), ["o", "s"]):
            d = out[name]["delay_by_effectiveness"]
            es = np.array([float(e) for e in d]); ds = np.array([np.nan if v is None else v for v in d.values()])
            ax2.semilogx(es, ds, marker=mk, ms=3, lw=0.8, color=cols[name], label=f"k = {k}")
            miss = np.isnan(ds)
            if miss.any():   # not detected within the attack segment
                ax2.semilogx(es[miss], np.full(miss.sum(), top), ls="none", marker="x", ms=5,
                             color=cols[name])
        ax2.axhline(top, color="0.6", lw=0.5, ls=":")
        ax2.set_ylim(-0.05 * top, 1.1 * top)
        ax2.set_ylabel("delay (observations)")
        ax2.set_title("weakened attack (x: not detected)", fontsize=8, loc="left")
        ax2.legend(frameon=False, fontsize=6, loc="center right")
    h_, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper center", ncol=5, frameon=False, fontsize=6.5)
    axes[-1, 0].set_xlabel("observation in test sequence")
    axes[-1, 1].set_xlabel("attack effectiveness e")
    fig.tight_layout(rect=(0, 0, 1, 0.92)); os.makedirs(args.out_dir, exist_ok=True)
    fig.savefig(os.path.join(args.out_dir, "section4_cusum.png"), dpi=200); plt.close(fig)
    return res


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__,
                         formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--cusum-k-tuned", type=float, default=1.0)
    ap.add_argument("--cusum-k-conservative", type=float, default=3.0)
    add_cusum_args(ap)
    add_h_args(ap)
    args = ap.parse_args()
    save_json(run(args), os.path.join(args.out_dir, "alg3_cusum_t2.json"))


if __name__ == "__main__":
    main()
