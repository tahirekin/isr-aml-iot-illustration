"""Algorithm 1 (Section 4.1, Data and Setup): load, split, standardize, project.

Loads the Danmini Doorbell device of N-BaIoT (Meidan et al., 2018), forms the
training, held-out benign, and attack blocks used throughout Section 4, and
reports the two preprocessing facts every later algorithm depends on:
  * the raw features span many orders of magnitude, so each is z-scored with
    training moments;
  * the standardized training covariance is rank-deficient, so T^2 is computed
    in the non-degenerate principal subspace (effective dimension p_eff).

Usage:
    python alg1_data_setup.py --data-dir /path/to/Danmini_Doorbell --out-dir results
Outputs:
    results/alg1_data_setup.json, results/section4_eigenvalues.png
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


def run(args):
    data = load_device(args.data_dir, args.n_train, args.n_test_benign,
                       args.n_attack_used, args.train_offset, args.families,
                       n_calib=args.n_calib, skip_restart=not args.keep_restart_rows)
    tr = StandardizeProject(args.rank_tol).fit(data["train"])
    sd = data["train"].std(axis=0, ddof=1)
    pos = sd[sd > 0]
    Z = (data["train"][:, tr.keep_cols] - tr.mu[tr.keep_cols]) / tr.sd
    evals = np.sort(np.linalg.eigvalsh(np.cov(Z, rowvar=False)))[::-1]
    res = {"meta": setup_meta(args, data, tr),
           "raw_scale": {"log10_max_over_min_feature_sd": float(np.log10(pos.max() / pos.min())),
                         "min_feature_sd": float(pos.min()), "max_feature_sd": float(pos.max())},
           "eigenvalues_standardized_cov": evals.tolist(),
           "n_rows_available": {"benign_used": args.n_train + args.n_test_benign,
                                **{f"{f}_spare_for_contamination": int(len(data["attack_spare"][f]))
                                   for f in args.families}}}
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.semilogy(np.arange(1, len(evals) + 1), np.maximum(evals, 1e-18), "k.", ms=3)
    ax.axhline(args.rank_tol * evals[0], color="r", lw=0.8, ls="--",
               label=f"retention cut (p_eff = {tr.p_eff})")
    ax.set_xlabel("component"); ax.set_ylabel("eigenvalue")
    ax.set_title("standardized training covariance", fontsize=8, loc="left")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout()
    os.makedirs(args.out_dir, exist_ok=True)
    fig.savefig(os.path.join(args.out_dir, "section4_eigenvalues.png"), dpi=200); plt.close(fig)
    print(f"p_eff = {tr.p_eff} of {len(data['columns'])} features "
          f"({tr.n_constant_cols} constant columns dropped)")
    return res


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__,
                         formatter_class=argparse.RawDescriptionHelpFormatter))
    args = ap.parse_args()
    save_json(run(args), os.path.join(args.out_dir, "alg1_data_setup.json"))


if __name__ == "__main__":
    main()
