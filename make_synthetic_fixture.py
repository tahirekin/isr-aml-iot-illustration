"""TEST FIXTURE ONLY. Writes synthetic data shaped like one N-BaIoT device
(UCI layout, the 115 Kitsune column names) so that the Section 4 scripts can be
checked end to end without the real data. Nothing produced from this fixture
belongs in the paper.

    python tests/make_synthetic_fixture.py --out /tmp/Fake_Doorbell
"""
import argparse
import os

import numpy as np
import pandas as pd

WINDOWS = ["L5", "L3", "L1", "L0.1", "L0.01"]


def kitsune_columns():
    cols = []
    for w in WINDOWS:
        cols += [f"MI_dir_{w}_{s}" for s in ("weight", "mean", "variance")]
    for w in WINDOWS:
        cols += [f"H_{w}_{s}" for s in ("weight", "mean", "variance")]
    for w in WINDOWS:
        cols += [f"HH_{w}_{s}" for s in ("weight", "mean", "std", "magnitude", "radius", "covariance", "pcc")]
    for w in WINDOWS:
        cols += [f"HH_jit_{w}_{s}" for s in ("weight", "mean", "variance")]
    for w in WINDOWS:
        cols += [f"HpHp_{w}_{s}" for s in ("weight", "mean", "std", "magnitude", "radius", "covariance", "pcc")]
    assert len(cols) == 115
    return cols


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1); a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    p, r = 115, 80                                   # rank-deficient by construction
    A = rng.normal(size=(r, p))
    scale = 10 ** rng.uniform(-6, 9, size=p)        # raw scales over ~15 orders of magnitude
    center = 10 ** rng.uniform(-3, 6, size=p)

    def latent(n, phi=0.6):
        z = np.zeros((n, r)); e = rng.normal(size=(n, r))
        for i in range(1, n):
            z[i] = phi * z[i - 1] + np.sqrt(1 - phi ** 2) * e[i]
        return z

    benign = latent(6000) @ A
    benign[3500:3600] += 2.5 * rng.normal(size=(1, p))       # a burst of unusual benign traffic
    shift_g = rng.normal(size=r) @ A; shift_m = rng.normal(size=r) @ A
    gafgyt = latent(3000) @ A * 1.5 + 1.2 * shift_g
    mirai = latent(3000) @ A * 1.2 + 0.6 * shift_m
    cols = kitsune_columns()
    def to_df(Z): return pd.DataFrame(center + scale * Z / 10.0, columns=cols)
    os.makedirs(os.path.join(a.out, "gafgyt_attacks"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "mirai_attacks"), exist_ok=True)
    to_df(benign).to_csv(os.path.join(a.out, "benign_traffic.csv"), index=False)
    to_df(gafgyt).to_csv(os.path.join(a.out, "gafgyt_attacks", "combo.csv"), index=False)
    to_df(mirai).to_csv(os.path.join(a.out, "mirai_attacks", "syn.csv"), index=False)
    print("wrote synthetic device folder to", a.out)


if __name__ == "__main__":
    main()
