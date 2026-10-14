"""Algorithm 4 (Section 4.3): training-set contamination and the MCD estimator.

A defender who trained the baseline partway through an undetected intrusion:
a fraction of the clean training rows is replaced by attack rows (drawn from
attack rows that are NOT part of the test sequence). The standardization,
projection, and both estimators are then fitted to the contaminated block, as
the defender would. The false-alarm rate on the same clean held-out benign
block is compared for
  * the naive T^2 (sample mean and covariance), and
  * the robust T^2 based on the Minimum Covariance Determinant (Rousseeuw and
    Van Driessen, 1999; scikit-learn's FastMCD with reweighting),
both against the Phase II F limit for their own dimension. Both estimators are
also fitted to the clean training block, so that the effect of contamination on
each is measured against that estimator's own clean baseline. The share of
attack rows each detector flags is reported so that a lower false-alarm rate is
not mistaken for a better detector.

Robust distances use the same rank rule as the naive T^2: eigen-directions of
the MCD scatter with eigenvalue below rank_tol times the largest are dropped,
and the F limit uses the retained rank q. This matters on N-BaIoT: when more
than half of the rows lie on a lower-dimensional affine subspace (a recurring
state in which the host-to-host channel statistics are exactly constant), MCD's
exact-fit property makes its scatter matrix singular. The script reports the
rank q, the raw features that are exactly constant within the MCD support, the
share of training and test rows in that state, the false-alarm rates of both
detectors inside and outside it, and the number of numerical warnings raised
by the MCD fit, so that this case is visible in the output rather than hidden
in the distances.

A false-alarm rate against the F limit mixes two things: the estimator, and
the Gaussian model behind the limit. To separate them, each of the four
detectors (naive or MCD, clean or contaminated training) is also calibrated
empirically: its limit is the (1 - alpha) quantile of its statistic on the
Phase I calibration block. At these calibrated limits the script reports the
false-alarm rate on the test block and the detection rate against weakened
attacks x(e) = e * x_attack + (1 - e) * mu_benign over a grid of effectiveness
values e, which is where contamination does its damage (masking).

FastMCD is a randomized search, and on this data its result depends noticeably
on the random start (the scatter of the selected half-sample is singular, and
scikit-learn warns that the determinant increased during the concentration
steps). MCD is therefore fitted from --mcd-starts random starts; MCD rates are
reported as the median over starts, with the range, and the figure shows the
range as a band.

Usage:
    python alg4_contamination_mcd.py --data-dir /path/to/Danmini_Doorbell \
        --out-dir results --contamination-frac 0.05 --seed 0
Outputs:
    results/alg4_contamination_mcd.json, results/section4_contamination.png
Requires: numpy, pandas, scipy, scikit-learn, matplotlib
Note: the MCD fit with n = 2000 and p_eff near 80 can take a minute or more.
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
import warnings

from sklearn.covariance import MinCovDet


def contaminate(train, spare, frac, rng):
    m = int(round(frac * len(train)))
    if m > len(spare):
        raise ValueError(f"need {m} spare attack rows, have {len(spare)}")
    rows = rng.choice(len(train), size=m, replace=False)
    src = rng.choice(len(spare), size=m, replace=False)
    out = train.copy(); out[rows] = spare[src]
    return out, m


def fit_mcd(Y, support_fraction, seed, rank_tol):
    """FastMCD with reweighting, then an eigen-decomposition of the robust scatter
    truncated by the same rule as StandardizeProject. Returns a dict with the
    location, retained eigenvectors and eigenvalues, the support mask, and the
    number of warnings sklearn raised during the fit."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mcd = MinCovDet(support_fraction=support_fraction, random_state=seed).fit(Y)
    evals, evecs = np.linalg.eigh(mcd.covariance_)
    order = np.argsort(evals)[::-1]; evals, evecs = evals[order], evecs[:, order]
    keep = evals > rank_tol * evals[0]
    return {"location": mcd.location_, "V": evecs[:, keep], "evals": evals[keep],
            "support": mcd.support_, "rank": int(keep.sum()),
            "n_fit_warnings": sum(issubclass(x.category, RuntimeWarning) for x in w)}


def robust_t2(Y, fit):
    Z = (Y - fit["location"]) @ fit["V"]
    return np.sum(Z ** 2 / fit["evals"], axis=1)


def constant_within(X, mask, tol=1e-9):
    """Raw features that take a single value (to relative precision tol) on the
    rows in `mask` but vary over all rows; returns the boolean mask and values."""
    med = np.median(X[mask], axis=0)
    const = (np.ptp(X[mask], axis=0) <= tol * np.maximum(1.0, np.abs(med))) & (X.std(axis=0) > 0)
    return const, med


def in_state(X, const, med, tol=1e-9):
    """Rows whose `const` features all equal the values `med` (the exact-fit state)."""
    if not const.any():
        return np.zeros(len(X), dtype=bool)
    return np.all(np.abs(X[:, const] - med[const]) <= tol * np.maximum(1.0, np.abs(med[const])), axis=1)


EFFECTIVENESS_GRID = np.logspace(0.0, -5.0, 21)


def _summary(values):
    v = np.asarray(values, dtype=float)
    return {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())}


def estimator_results(Xtrain, Xcalib, Xtest, Xatt, mu_benign, cols, args, label):
    """Naive and MCD T^2 fitted to Xtrain (preprocessing refitted too).

    FastMCD is a randomized search, and on this data its result is numerically
    fragile (the scatter of the best h-subset is singular), so MCD is fitted from
    args.mcd_starts random starts. Each MCD quantity is reported per start and as
    median, min, and max over starts; the diagnostics of the exact-fit state use
    the first start."""
    a = args.alpha
    tr = StandardizeProject(args.rank_tol).fit(Xtrain)
    Y = tr.transform(Xtrain); n, p = Y.shape
    Yc, Yt, Ya = tr.transform(Xcalib), tr.transform(Xtest), tr.transform(Xatt)
    Ye = {f"{e:.3g}": tr.transform(e * Xatt + (1 - e) * mu_benign) for e in EFFECTIVENESS_GRID}
    mu, S = Y.mean(0), np.cov(Y, rowvar=False)
    lim = t2_control_limit(p, n, a)

    def calibrated(f):
        lim_e = float(np.quantile(f(Yc), 1 - a))
        return {"limit": lim_e, "false_alarm_rate": float(np.mean(f(Yt) > lim_e)),
                "detection_rate_by_effectiveness": {k: float(np.mean(f(Z) > lim_e))
                                                    for k, Z in Ye.items()}}

    naive = lambda Z: hotelling_t2(Z, mu, S)
    fa_naive = naive(Yt) > lim
    print(f"  {label}: fitting MCD (n={n}, p_eff={p}) from {args.mcd_starts} random starts ...",
          flush=True)
    starts, fits = [], []
    for j in range(args.mcd_starts):
        fit = fit_mcd(Y, args.mcd_support_fraction, args.mcd_random_state + j, args.rank_tol)
        lim_r = t2_control_limit(fit["rank"], n, a)
        f = lambda Z, fit=fit: robust_t2(Z, fit)
        starts.append({"random_state": args.mcd_random_state + j, "rank": fit["rank"],
                       "control_limit": float(lim_r), "n_support": int(fit["support"].sum()),
                       "false_alarm_rate": float(np.mean(f(Yt) > lim_r)),
                       "detection_rate_on_attack": float(np.mean(f(Ya) > lim_r)),
                       "fit_warnings": fit["n_fit_warnings"], "calibrated": calibrated(f)})
        fits.append(fit)
    fit0 = fits[0]
    fa_mcd0 = robust_t2(Yt, fit0) > starts[0]["control_limit"]
    mcd0 = lambda Z: robust_t2(Z, fit0)
    # row-level series used for bootstrap intervals (not saved)
    rows = {"naive_test_flags": fa_naive, "mcd0_test_flags": fa_mcd0,
            "naive_calib_stat": naive(Yc), "naive_test_stat": naive(Yt),
            "naive_attack_stat": naive(Ya),
            "mcd0_calib_stat": mcd0(Yc), "mcd0_test_stat": mcd0(Yt), "mcd0_attack_stat": mcd0(Ya)}
    const, med = constant_within(Xtrain, fit0["support"])
    st_tr, st_te = in_state(Xtrain, const, med), in_state(Xtest, const, med)
    rate = lambda fl, m: float(np.mean(fl[m])) if m.any() else None
    keys = list(Ye)
    mcd_cal = {
        "limit": _summary([s_["calibrated"]["limit"] for s_ in starts]),
        "false_alarm_rate": _summary([s_["calibrated"]["false_alarm_rate"] for s_ in starts]),
        "detection_rate_by_effectiveness": {
            k: float(np.median([s_["calibrated"]["detection_rate_by_effectiveness"][k] for s_ in starts]))
            for k in keys},
        "detection_rate_by_effectiveness_min": {
            k: float(min(s_["calibrated"]["detection_rate_by_effectiveness"][k] for s_ in starts))
            for k in keys},
        "detection_rate_by_effectiveness_max": {
            k: float(max(s_["calibrated"]["detection_rate_by_effectiveness"][k] for s_ in starts))
            for k in keys}}
    far = _summary([s_["false_alarm_rate"] for s_ in starts])
    return {
        "p_eff": p, "naive_control_limit": float(lim),
        "naive_false_alarm_rate": float(np.mean(fa_naive)),
        "naive_detection_rate_on_attack": float(np.mean(naive(Ya) > lim)),
        "mcd_n_starts": args.mcd_starts,
        "mcd_false_alarm_rate": far["median"],
        "mcd_false_alarm_rate_range": [far["min"], far["max"]],
        "mcd_detection_rate_on_attack": _summary([s_["detection_rate_on_attack"] for s_ in starts]),
        "mcd_scatter_rank": _summary([s_["rank"] for s_ in starts]),
        "mcd_n_support": _summary([s_["n_support"] for s_ in starts]),
        "mcd_fit_warnings_total": int(sum(s_["fit_warnings"] for s_ in starts)),
        "mcd_per_start": starts,
        "exact_fit_state_first_start": {
            "n_raw_features_constant_within_mcd_support": int(const.sum()),
            "raw_features_constant_within_mcd_support": [c for c, k in zip(cols, const) if k],
            "share_of_training_rows": float(np.mean(st_tr)),
            "share_of_mcd_support_in_state": float(np.mean(st_tr[fit0["support"]])),
            "share_of_test_rows": float(np.mean(st_te)),
            "naive_false_alarm_rate_in_state": rate(fa_naive, st_te),
            "naive_false_alarm_rate_outside_state": rate(fa_naive, ~st_te),
            "mcd_false_alarm_rate_in_state": rate(fa_mcd0, st_te),
            "mcd_false_alarm_rate_outside_state": rate(fa_mcd0, ~st_te)},
        "calibrated_limits": {"naive": calibrated(naive), "mcd": mcd_cal},
    }, fits, rows


def contamination_intervals(r0, rc, args):
    """95% moving-block bootstrap intervals (rows resampled in blocks, fitted
    estimates and limits held fixed except where stated)."""
    rng = np.random.default_rng(args.ci_seed)
    n = len(r0["naive_test_flags"])
    it = moving_block_indices(n, args.ci_block, args.ci_replicates, rng)
    f0, fc = r0["naive_test_flags"].astype(float), rc["naive_test_flags"].astype(float)
    out = {"method": f"moving-block bootstrap, block {args.ci_block}, {args.ci_replicates} replicates",
           "F_limit_false_alarm_rate": {
               "naive_clean": percentile_interval(f0[it].mean(axis=1)),
               "naive_contaminated": percentile_interval(fc[it].mean(axis=1)),
               "naive_clean_minus_contaminated": percentile_interval((f0[it] - fc[it]).mean(axis=1)),
               "share_of_replicates_with_lower_contaminated_rate": float(np.mean(f0[it].mean(1) > fc[it].mean(1))),
               "mcd_clean_first_start": percentile_interval(r0["mcd0_test_flags"].astype(float)[it].mean(axis=1)),
               "mcd_contaminated_first_start": percentile_interval(rc["mcd0_test_flags"].astype(float)[it].mean(axis=1))},
           "calibrated_false_alarm_rate": {
               lab: calibrated_rate_interval(r[f"{est}_calib_stat"], r[f"{est}_test_stat"], args.alpha, args, rng)
               for lab, r, est in (("naive_clean", r0, "naive"), ("naive_contaminated", rc, "naive"),
                                   ("mcd_clean_first_start", r0, "mcd0"),
                                   ("mcd_contaminated_first_start", rc, "mcd0"))}}
    # detection of the full-strength attack at the calibrated limit, resampling
    # calibration rows (limit) and attack rows together
    out["calibrated_detection_rate_full_strength"] = {
        lab: calibrated_rate_interval(r[f"{est}_calib_stat"], r[f"{est}_attack_stat"], args.alpha, args, rng)
        for lab, r, est in (("naive_clean", r0, "naive"), ("naive_contaminated", rc, "naive"),
                            ("mcd_clean_first_start", r0, "mcd0"),
                            ("mcd_contaminated_first_start", rc, "mcd0"))}
    return out


def estimand_shift(X_clean, X_cont, X_att, fits_clean, args):
    """How far each estimator moves under contamination, measured in one fixed
    coordinate system: the standardization and projection fitted to the clean
    training block, whitened by the naive clean covariance. Reports the shift of
    the location estimate (Mahalanobis units) and the estimated variance along
    the attack direction v (the whitened direction from the clean mean to the
    mean of the attack rows), for the clean and the contaminated fit. For the
    naive estimator the clean variance along v is 1 by construction."""
    tr0 = StandardizeProject(args.rank_tol).fit(X_clean)
    Y0, Yc, Ya = tr0.transform(X_clean), tr0.transform(X_cont), tr0.transform(X_att)
    mu0, S0 = Y0.mean(0), np.cov(Y0, rowvar=False)
    ev, U = np.linalg.eigh(S0)
    W = U @ np.diag(ev ** -0.5) @ U.T
    v = W @ (Ya.mean(0) - mu0); v = v / np.linalg.norm(v)
    var_v = lambda S: float(v @ W @ S @ W @ v)
    shift = lambda a_, b_: float(np.linalg.norm(W @ (a_ - b_)))
    cov_of = lambda f: f["V"] @ np.diag(f["evals"]) @ f["V"].T
    out = {"coordinates": "clean-training standardization and projection, whitened by the naive clean covariance",
           "naive": {"location_shift": shift(Yc.mean(0), mu0),
                     "variance_along_attack_clean": var_v(S0),
                     "variance_along_attack_contaminated": var_v(np.cov(Yc, rowvar=False))}}
    loc, vc, vk = [], [], []
    for j, f0 in enumerate(fits_clean):
        fc = fit_mcd(Yc, args.mcd_support_fraction, args.mcd_random_state + j, args.rank_tol)
        loc.append(shift(fc["location"], f0["location"]))
        vc.append(var_v(cov_of(f0))); vk.append(var_v(cov_of(fc)))
    out["mcd"] = {"location_shift": _summary(loc), "variance_along_attack_clean": _summary(vc),
                  "variance_along_attack_contaminated": _summary(vk)}
    return out


def run(args):
    data = load_device(args.data_dir, args.n_train, args.n_test_benign,
                       args.n_attack_used, args.train_offset, args.families,
                       n_calib=args.n_calib, skip_restart=not args.keep_restart_rows)
    rng = np.random.default_rng(args.seed)
    mu_benign = data["train"].mean(axis=0)     # attacker throttles toward clean benign mean
    tr0 = StandardizeProject(args.rank_tol).fit(data["train"])
    res = {"meta": setup_meta(args, data, tr0),
           "settings": {"alpha": args.alpha, "contamination_frac": args.contamination_frac,
                        "seed": args.seed, "mcd_support_fraction": args.mcd_support_fraction,
                        "mcd_starts": args.mcd_starts, "mcd_random_state": args.mcd_random_state,
                        "rank_tol": args.rank_tol},
           "contamination_demo": {}}
    for fam in args.families:
        print(f"{fam}:", flush=True)
        clean, fits_0, rows_0 = estimator_results(data["train"], data["calib"], data["test_benign"],
                                     data["attack"][fam], mu_benign, data["columns"], args,
                                     "clean training block")
        Xc, m = contaminate(data["train"], data["attack_spare"][fam], args.contamination_frac, rng)
        injected = np.flatnonzero(np.any(Xc != data["train"], axis=1))
        cont, fits_c, rows_c = estimator_results(Xc, data["calib"], data["test_benign"],
                                         data["attack"][fam], mu_benign, data["columns"], args,
                                         "contaminated training block")
        outside = [float(np.mean(~fc["support"][injected])) for fc in fits_c] if injected.size else None
        intervals = contamination_intervals(rows_0, rows_c, args)
        estimand = estimand_shift(data["train"], Xc, data["attack"][fam], fits_0, args)
        res["contamination_demo"][fam] = {
            "n_contaminating_rows": m,
            # keys used by the manuscript (MCD rates: median over random starts)
            "naive_false_alarm_rate_uncontaminated_baseline": clean["naive_false_alarm_rate"],
            "naive_false_alarm_rate_on_clean": cont["naive_false_alarm_rate"],
            "mcd_false_alarm_rate_uncontaminated_baseline": clean["mcd_false_alarm_rate"],
            "mcd_false_alarm_rate_on_clean": cont["mcd_false_alarm_rate"],
            "naive_detection_rate_on_attack": cont["naive_detection_rate_on_attack"],
            "mcd_detection_rate_on_attack": cont["mcd_detection_rate_on_attack"]["median"],
            "mcd_share_of_contaminating_rows_outside_support":
                _summary(outside) if outside is not None else None,
            "intervals": intervals, "estimand_shift": estimand,
            "clean_fit": clean, "contaminated_fit": cont}
        d = res["contamination_demo"][fam]
        print(f"{fam}: FAR naive clean {d['naive_false_alarm_rate_uncontaminated_baseline']:.3f} -> "
              f"contaminated {d['naive_false_alarm_rate_on_clean']:.3f}; MCD clean "
              f"{d['mcd_false_alarm_rate_uncontaminated_baseline']:.3f} "
              f"(range {clean['mcd_false_alarm_rate_range'][0]:.3f}-{clean['mcd_false_alarm_rate_range'][1]:.3f})"
              f" -> contaminated {d['mcd_false_alarm_rate_on_clean']:.3f}; MCD rank "
              f"{clean['mcd_scatter_rank']['min']:.0f}-{clean['mcd_scatter_rank']['max']:.0f}, "
              f"{clean['exact_fit_state_first_start']['n_raw_features_constant_within_mcd_support']}"
              f" features constant in MCD support")
    fams = list(res["contamination_demo"])
    fig, axes = plt.subplots(1, 1 + len(fams), figsize=(3.0 + 2.9 * len(fams), 2.9),
                             gridspec_kw={"width_ratios": [1.25] + [1] * len(fams)})
    ax = axes[0]; w = 0.2; x = np.arange(len(fams))
    bars = [("naive_false_alarm_rate_uncontaminated_baseline", "naive, clean", "0.85", "", None),
            ("naive_false_alarm_rate_on_clean", "naive, contaminated", "0.55", "", None),
            ("mcd_false_alarm_rate_uncontaminated_baseline", "MCD, clean", "0.85", "///", "clean_fit"),
            ("mcd_false_alarm_rate_on_clean", "MCD, contaminated", "0.55", "///", "contaminated_fit")]
    for j, (key, lab, col, hatch, fk) in enumerate(bars):
        hv = [res["contamination_demo"][f][key] for f in fams]
        err = None
        if fk is not None:   # range over MCD random starts
            rg = [res["contamination_demo"][f][fk]["mcd_false_alarm_rate_range"] for f in fams]
            err = np.array([[v - r[0] for v, r in zip(hv, rg)], [r[1] - v for v, r in zip(hv, rg)]])
        ax.bar(x + (j - 1.5) * w, hv, w, yerr=err, capsize=2, error_kw={"lw": 0.6},
               label=lab, color=col, hatch=hatch, edgecolor="k", lw=0.5)
    ax.axhline(args.alpha, color="r", lw=0.8, ls="--", label=f"nominal {args.alpha}")
    ax.set_xticks(x); ax.set_xticklabels([f.capitalize() for f in fams])
    ax.set_ylabel("false-alarm rate, test block")
    ax.set_title("(a) F limit", fontsize=8, loc="left")
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    ax.set_ylim(0, 1.0)
    styles = [("clean_fit", "naive", "naive, clean", "C0", "-"),
              ("contaminated_fit", "naive", "naive, contaminated", "C0", "--"),
              ("clean_fit", "mcd", "MCD, clean", "C3", "-"),
              ("contaminated_fit", "mcd", "MCD, contaminated", "C3", "--")]
    for ax, f in zip(axes[1:], fams):
        for fit_key, est, lab, col, ls in styles:
            dr = res["contamination_demo"][f][fit_key]["calibrated_limits"][est]
            e = np.array([float(k) for k in dr["detection_rate_by_effectiveness"]])
            ax.semilogx(e, list(dr["detection_rate_by_effectiveness"].values()),
                        color=col, ls=ls, lw=1, label=lab)
            if est == "mcd":     # band: min to max over random starts
                ax.fill_between(e, list(dr["detection_rate_by_effectiveness_min"].values()),
                                list(dr["detection_rate_by_effectiveness_max"].values()),
                                color=col, alpha=0.12 if ls == "-" else 0.22, lw=0)
        ax.set_ylim(-0.03, 1.03); ax.set_xlabel("attack effectiveness e")
        ax.set_title(f"({'bcd'[fams.index(f)]}) {f.capitalize()}, calibrated limits", fontsize=8, loc="left")
    axes[1].set_ylabel("detection rate")
    h_, l_ = axes[1].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=4, frameon=False, fontsize=6,
               bbox_to_anchor=(0.62, 0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 1)); os.makedirs(args.out_dir, exist_ok=True)
    fig.savefig(os.path.join(args.out_dir, "section4_contamination.png"), dpi=200); plt.close(fig)
    return res


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__,
                         formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--contamination-frac", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--mcd-support-fraction", type=float, default=None,
                    help="default: scikit-learn's (n + p + 1) / 2n")
    ap.add_argument("--seed", type=int, default=0, help="which spare attack rows are injected")
    ap.add_argument("--mcd-starts", type=int, default=5,
                    help="number of FastMCD random starts (results reported as median and range)")
    ap.add_argument("--mcd-random-state", type=int, default=0,
                    help="random_state of the first FastMCD start")
    args = ap.parse_args()
    save_json(run(args), os.path.join(args.out_dir, "alg4_contamination_mcd.json"))


if __name__ == "__main__":
    main()
