"""Algorithm 2 (Section 4.2.1): naive Hotelling's T^2 as a fixed-sample detector.

Estimates the mean and covariance from the clean training block (sample
moments, in the projected subspace), computes T^2 for each row of the test
sequence (held-out benign rows followed by attack rows), and flags rows above
the Phase II control limit p(n+1)(n-1)/(n^2 - np) F_{p,n-p,alpha} (Appendix A)
at alpha = 0.01 and 0.001. Reports, per attack family, the number and rate of
false alarms before the known onset, the detection delay (rows after onset
until the first flag), and the share of attack rows flagged.

The F limit assumes Gaussian rows. As a check on that assumption, the same
quantities are also reported for an empirical (Phase I) limit: the (1 - alpha)
quantile of T^2 over the calibration block of benign rows, which were used
neither to estimate mu and Sigma nor to count false alarms.

Usage:
    python alg2_hotelling_t2.py --data-dir /path/to/Danmini_Doorbell --out-dir results
Outputs:
    results/alg2_hotelling_t2.json, results/section4_tsq_traces.png
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


def evaluate_flags(t2, limit, onset):
    flags = t2 > limit
    pre = flags[:onset]
    first = first_index_at_or_after(flags, onset)
    return {"false_alarms_pre_onset": int(pre.sum()),
            "false_alarm_rate_pre_onset": float(pre.mean()),
            "detected": first is not None,
            "detection_delay_rows": None if first is None else first - onset,
            "detection_rate_post_onset": float(flags[onset:].mean())}


def run(args):
    data = load_device(args.data_dir, args.n_train, args.n_test_benign,
                       args.n_attack_used, args.train_offset, args.families,
                       n_calib=args.n_calib, skip_restart=not args.keep_restart_rows)
    tr = StandardizeProject(args.rank_tol).fit(data["train"])
    Ytr = tr.transform(data["train"])
    mean, cov = Ytr.mean(axis=0), np.cov(Ytr, rowvar=False)
    n, p = Ytr.shape
    alphas = args.alphas
    limits = {str(a): float(t2_control_limit(p, n, a)) for a in alphas}
    t2_cal = hotelling_t2(tr.transform(data["calib"]), mean, cov)
    emp_limits = {str(a): float(np.quantile(t2_cal, 1 - a)) for a in alphas}
    onset = args.n_test_benign
    res = {"meta": setup_meta(args, data, tr),
           "t2_control_limits": {"naive": limits, "naive_empirical_calibration": emp_limits},
           "benign_t2_quantiles": {
               "calibration": {q: float(np.quantile(t2_cal, float(q))) for q in ["0.5", "0.9", "0.99"]},
               "training_insample": {q: float(np.quantile(hotelling_t2(Ytr, mean, cov), float(q)))
                                     for q in ["0.5", "0.9", "0.99"]},
               "F_reference_median": float(t2_control_limit(p, n, 0.5)),
               "test_block_median": float(np.median(hotelling_t2(tr.transform(data["test_benign"]),
                                                                 mean, cov)))},
           "projected_training_excess_kurtosis": {
               "min": float(stats.kurtosis(Ytr, axis=0).min()),
               "max": float(stats.kurtosis(Ytr, axis=0).max())},
           "per_attack": {}}
    # 95% moving-block bootstrap intervals for the false-alarm rates on the test
    # block: conditional on the fitted training estimates for the F limit, and
    # resampling both the calibration and the test block for the calibrated limit.
    rng = np.random.default_rng(args.ci_seed)
    t2_test = hotelling_t2(tr.transform(data["test_benign"]), mean, cov)
    res["false_alarm_rate_intervals"] = {
        "F_limit": {str(a): rate_interval(t2_test > limits[str(a)], args, rng) for a in alphas},
        "calibrated_limit": {str(alphas[0]): calibrated_rate_interval(t2_cal, t2_test, alphas[0],
                                                                       args, rng)},
        "method": f"moving-block bootstrap, block {args.ci_block}, {args.ci_replicates} replicates"}
    traces = {}
    for fam in args.families:
        seq = np.vstack([data["test_benign"], data["attack"][fam]])
        t2 = hotelling_t2(tr.transform(seq), mean, cov)
        traces[fam] = t2
        t2a = t2[onset:]
        res["per_attack"][fam] = {
            "n_attack_used": int(len(data["attack"][fam])), "onset_index": onset,
            "naive": {str(a): evaluate_flags(t2, limits[str(a)], onset) for a in alphas},
            "naive_empirical_limit": {str(a): evaluate_flags(t2, emp_limits[str(a)], onset)
                                      for a in alphas},
            "attack_t2_quantiles": {"min": float(t2a.min()), "0.5": float(np.median(t2a)),
                                    "max": float(t2a.max())}}
        r = res["per_attack"][fam]["naive"][str(alphas[0])]
        re_ = res["per_attack"][fam]["naive_empirical_limit"][str(alphas[0])]
        print(f"{fam}: alpha={alphas[0]} F limit: false alarms {r['false_alarms_pre_onset']}/{onset} "
              f"({r['false_alarm_rate_pre_onset']:.3f}), delay {r['detection_delay_rows']}; "
              f"empirical limit: false alarms {re_['false_alarms_pre_onset']}, delay {re_['detection_delay_rows']}")
    fig, axes = plt.subplots(len(traces), 1, figsize=(6.5, 2.2 * len(traces)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (fam, t2) in zip(axes, traces.items()):
        ax.semilogy(np.arange(len(t2)), t2, color="0.25", lw=0.6)
        a0 = str(alphas[0])   # the alpha = 0.001 F limit sits too close to be told apart
        ax.axhline(limits[a0], color="r", lw=0.8, ls="--",
                   label=f"F limit, alpha = {alphas[0]}")
        ax.axhline(emp_limits[a0], color="g", lw=0.8, ls="-.",
                   label=f"calibrated limit, alpha = {alphas[0]}")
        ax.axvline(onset, color="b", lw=0.8, label="attack onset")
        ax.set_ylabel("$T^2$ (log scale)"); ax.set_title(fam.capitalize(), fontsize=9, loc="left")
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper center", ncol=3, frameon=False, fontsize=7)
    axes[-1].set_xlabel("row of test sequence")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); os.makedirs(args.out_dir, exist_ok=True)
    fig.savefig(os.path.join(args.out_dir, "section4_tsq_traces.png"), dpi=200); plt.close(fig)
    return res


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__,
                         formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.001])
    args = ap.parse_args()
    save_json(run(args), os.path.join(args.out_dir, "alg2_hotelling_t2.json"))


if __name__ == "__main__":
    main()
