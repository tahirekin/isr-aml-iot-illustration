"""Sensitivity of the Section 4 results to choices the main run fixes.

Re-runs Algorithms 2-5 with one setting changed at a time and writes a table of
the headline numbers:
  * where the benign blocks sit in the recording (--train-offset), which moves
    the training, calibration, and test blocks together. The first such window is
    the main run's default. The Doorbell's benign
    file is several captures joined end to end; a capture boundary is a row at
    which every long-window weight (*_L0.01_weight) restarts at 1. By default
    the windows are laid end to end inside each capture, so that no window
    crosses a boundary. Windows that do cross one can be added with
    --crossing-offsets; they are labelled as such, because a restart of the
    feature extractor is a change in the benign data, not an attack;
  * the random draw of contaminating rows (--seed) and the contamination
    fraction (Algorithm 4 only);
  * MCD's support fraction (Algorithm 4 only);
  * the eigenvalue cut-off rank_tol (Algorithms 2, 3, 5);
  * the design of the CUSUM threshold: the target in-control ARL, the mean block
    length of the Phase I bootstrap, and h fixed at 5 instead (Algorithms 3, 5);
  * the defender's uncertainty about the attacker in the ARA: the concentration
    kappa of the attacker's weights and the SD sigma_k of its perception of k
    (Algorithm 5).
Each detection run also reports the ARA k* under the fixed-h design.

Usage:
    python sensitivity.py --data-dir /path/to/Danmini_Doorbell --out-dir results/sensitivity
Outputs:
    sensitivity.json and sensitivity.csv in --out-dir
Takes about two hours on one core, most of it in the MCD fits (five random
starts each). To use two cores, run
    python sensitivity.py ... --part 0 --of 2 &  python sensitivity.py ... --part 1 --of 2
and then python sensitivity.py ... --merge.
"""
import argparse
import copy
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alg2_hotelling_t2 as a2        # noqa: E402
import alg3_cusum_t2 as a3            # noqa: E402
import alg4_contamination_mcd as a4   # noqa: E402
import alg5_ara_choice_of_k as a5     # noqa: E402

DEFAULTS = dict(n_train=2000, n_calib=1000, n_test_benign=1000, n_attack_used=1000,
                train_offset=None, rank_tol=1e-8, families=["gafgyt", "mirai"],
                keep_restart_rows=False, alphas=[0.01, 0.001],
                cusum_k_tuned=1.0, cusum_k_conservative=3.0, cusum_h_sigmas=5.0,
                cusum_statistic="log", cusum_reference="calibration",
                contamination_frac=0.05, alpha=0.01, mcd_support_fraction=None, seed=0,
                mcd_starts=5, mcd_random_state=0,
                k_tuned=1.0, k_conservative=3.0, k_min=0.0, k_max=6.0, k_n=25,
                e_per_decade=10, e_min=1e-6, horizon=None, c_delay=0.0, c_fa=1.0, c_damage=1.0,
                fa_cap=5.0, attacker_types=None,
                cusum_h_mode="arl", arl0=1000.0, boot_block=50.0, boot_replicates=400,
                boot_length=20000, boot_seed=0,
                ara_kappa=20.0, ara_sigma_k=0.5, ara_draws=4000, ara_seed=0, ara_onsets=20, ara_tol=1e-9,
                ci_block=50, ci_replicates=2000, ci_seed=0)


def capture_boundaries(data_dir):
    """Rows of the benign file at which the feature extractor restarts, the number
    of rows, and the mask of rows that repeat an earlier row."""
    X, cols = a2.load_matrix(a2.find_csv(data_dir, None, None))
    return a2.capture_restarts(X, cols), X.shape[0], a2.repeated_rows(X)


def within_capture_offsets(bounds, n_rows, width, repeated, max_repeat=0.5):
    """Window starts laid end to end inside each capture segment, skipping windows
    in which more than max_repeat of the rows repeat earlier rows of the file
    (several Doorbell captures begin with the same packets as the first one)."""
    edges = list(bounds) + [n_rows]
    out, skipped = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        o = int(a)
        while o + width <= b:
            (skipped if repeated[o:o + width].mean() > max_repeat else out).append(o)
            o += width
    return out, skipped


def e50(curve):
    """Smallest attack effectiveness on the grid at which the detection rate reaches
    0.5 (None if it never does; '<=' + floor if it is reached at the grid floor)."""
    es = [float(e) for e in curve]
    hit = [float(e) for e, r in curve.items() if r >= 0.5]
    if not hit:
        return None
    return f"<={min(es):g}" if min(hit) == min(es) else min(hit)


def headline(args, which):
    """Run the requested algorithms and return one flat dict per attack family."""
    rows = {f: {} for f in args.families}
    if "detect" in which:
        r2 = a2.run(args); r3 = a3.run(args); r5 = a5.run(args)
        hd = r3["cusum_h_design"]["by_k"]
        for f in args.families:
            n = r2["per_attack"][f]["naive"]["0.01"]
            e = r2["per_attack"][f]["naive_empirical_limit"]["0.01"]
            c = r3["per_attack"][f]["cusum_on_naive_t2"]
            za = r5["designs"]["arl"]["by_family"][f]["ara"]
            zf = r5["designs"]["fixed_h"]["by_family"][f]["ara"]
            rows[f].update({
                "t2_far_F_limit": n["false_alarm_rate_pre_onset"],
                "t2_far_empirical_limit": e["false_alarm_rate_pre_onset"],
                "t2_delay": n["detection_delay_rows"],
                "cusum_h_k_tuned": hd["tuned"]["h_used"],
                "cusum_h_k_conservative": hd["conservative"]["h_used"],
                "cusum_fa_k_tuned": c["tuned"]["n_false_alarms_pre_onset"],
                "cusum_fa_k_conservative": c["conservative"]["n_false_alarms_pre_onset"],
                "cusum_delay_k_tuned": c["tuned"]["detection_delay_rows"],
                "cusum_delay_k_conservative": c["conservative"]["detection_delay_rows"],
                "cusum_delay_e1e-3_k_tuned": c["tuned"]["delay_by_effectiveness"]["0.001"],
                "cusum_delay_e1e-3_k_conservative": c["conservative"]["delay_by_effectiveness"]["0.001"],
                "ara_k_star": za["k_star"], "ara_indifferent_across_k": za["indifferent_across_k"],
                "ara_fixed_h_indifferent_across_k": zf["indifferent_across_k"],
                "ara_near_optimal_k": " ".join(f"{k:g}" for k in za["near_optimal_k"]),
                "ara_damage_k_star": za["k_star_expected_damage"],
                "ara_damage_k_tuned": za["k_tuned"]["expected_damage"],
                "ara_damage_k_conservative": za["k_conservative"]["expected_damage"],
                "ara_k_star_test_false_alarms": za["k_star_n_false_alarms_test_block"],
                "ara_fixed_h_k_star": zf["k_star"]})
    if "contam" in which:
        r4 = a4.run(args)
        for f in args.families:
            d = r4["contamination_demo"][f]
            iv = d["intervals"]["F_limit_false_alarm_rate"]
            es = d["estimand_shift"]
            rows[f].update({
                "naive_far_clean": d["naive_false_alarm_rate_uncontaminated_baseline"],
                "naive_far_contaminated": d["naive_false_alarm_rate_on_clean"],
                "naive_far_diff_ci_low": iv["naive_clean_minus_contaminated"][0],
                "naive_far_diff_ci_high": iv["naive_clean_minus_contaminated"][1],
                "mcd_far_clean": d["mcd_false_alarm_rate_uncontaminated_baseline"],
                "mcd_far_contaminated": d["mcd_false_alarm_rate_on_clean"],
                "mcd_far_clean_min": d["clean_fit"]["mcd_false_alarm_rate_range"][0],
                "mcd_far_clean_max": d["clean_fit"]["mcd_false_alarm_rate_range"][1],
                "mcd_rank_clean_median": d["clean_fit"]["mcd_scatter_rank"]["median"],
                "mcd_injected_outside_support_min": d["mcd_share_of_contaminating_rows_outside_support"]["min"],
                "exact_fit_state_share_test":
                    d["clean_fit"]["exact_fit_state_first_start"]["share_of_test_rows"],
                "naive_location_shift": es["naive"]["location_shift"],
                "naive_variance_along_attack_contaminated": es["naive"]["variance_along_attack_contaminated"],
                "mcd_location_shift_median": es["mcd"]["location_shift"]["median"],
                "mcd_variance_ratio_along_attack": (es["mcd"]["variance_along_attack_contaminated"]["median"]
                                                   / es["mcd"]["variance_along_attack_clean"]["median"])})
            for fit_key, lab in (("clean_fit", "clean"), ("contaminated_fit", "contaminated")):
                for est in ("naive", "mcd"):
                    c = d[fit_key]["calibrated_limits"][est]
                    fa = c["false_alarm_rate"]
                    rows[f][f"{est}_{lab}_far_calibrated"] = fa["median"] if isinstance(fa, dict) else fa
                    rows[f][f"{est}_{lab}_e50_calibrated"] = e50(c["detection_rate_by_effectiveness"])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default="results/sensitivity")
    ap.add_argument("--offsets", type=int, nargs="+", default=None,
                    help="training-block starts; default: windows inside each capture")
    ap.add_argument("--crossing-offsets", type=int, nargs="*", default=[10000],
                    help="extra starts whose window crosses a capture boundary")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.02, 0.05, 0.10])
    ap.add_argument("--support-fractions", type=float, nargs="+", default=[0.75, 0.9])
    ap.add_argument("--rank-tols", type=float, nargs="+", default=[1e-10, 1e-6, 1e-4])
    ap.add_argument("--arl0s", type=float, nargs="+", default=[500.0, 2000.0])
    ap.add_argument("--boot-blocks", type=float, nargs="+", default=[20.0, 100.0])
    ap.add_argument("--kappas", type=float, nargs="+", default=[5.0, 100.0])
    ap.add_argument("--sigma-ks", type=float, nargs="+", default=[0.0, 1.0])
    ap.add_argument("--part", type=int, default=0, help="run every --of-th setting, from this one")
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--only", type=int, nargs="*", default=None,
                    help="run only these positions of the plan")
    ap.add_argument("--tag", default=None, help="name of the part file for --only")
    ap.add_argument("--merge", action="store_true",
                    help="merge sensitivity_part*.json in --out-dir into sensitivity.json/.csv")
    cli = ap.parse_args()
    base = argparse.Namespace(**copy.deepcopy(DEFAULTS), data_dir=cli.data_dir,
                              out_dir=os.path.join(cli.out_dir, "scratch"))
    bounds, n_rows, repeated = capture_boundaries(cli.data_dir)
    width = DEFAULTS["n_train"] + DEFAULTS["n_calib"] + DEFAULTS["n_test_benign"]
    skipped = []
    if cli.offsets is None:
        offsets, skipped = within_capture_offsets(bounds, n_rows, width, repeated)
    else:
        offsets = cli.offsets
    print(f"capture boundaries: {bounds.tolist()}; within-capture offsets: {offsets}; "
          f"skipped as mostly repeated rows: {skipped}")
    crossing = lambda o: [int(b) for b in bounds if o < b < o + width]
    plan = [("train_offset", v, ("detect", "contam")) for v in offsets]
    plan += [("train_offset_crossing", v, ("detect", "contam")) for v in cli.crossing_offsets]
    plan += [("seed", v, ("contam",)) for v in cli.seeds if v != 0]
    plan += [("contamination_frac", v, ("contam",)) for v in cli.fracs if v != 0.05]
    plan += [("mcd_support_fraction", v, ("contam",)) for v in cli.support_fractions]
    plan += [("rank_tol", v, ("detect",)) for v in cli.rank_tols]
    plan += [("arl0", v, ("detect",)) for v in cli.arl0s]
    plan += [("boot_block", v, ("detect",)) for v in cli.boot_blocks]
    plan += [("ara_kappa", v, ("detect",)) for v in cli.kappas]
    plan += [("ara_sigma_k", v, ("detect",)) for v in cli.sigma_ks]
    plan += [("cusum_h_mode", "fixed", ("detect",))]
    if cli.merge:
        parts = sorted(f for f in os.listdir(cli.out_dir) if f.startswith("sensitivity_part") and f.endswith(".json"))
        rows = {(r["order"], r["family"]): r for f in parts
                for r in json.load(open(os.path.join(cli.out_dir, f)))["rows"]}
        table = [rows[key] for key in sorted(rows)]
        for row in table:   # rows from parts run before the flag existed: a single
            if "ara_k_star" in row and "ara_indifferent_across_k" not in row \
                    and len(str(row["ara_near_optimal_k"]).split()) == 1:   # near-optimal k
                row["ara_indifferent_across_k"] = False                  # means not indifferent
        write_tables(cli.out_dir, "sensitivity", skipped, table)
        return
    if cli.only:
        plan = [(i, *x) for i, x in enumerate(plan) if i in cli.only]
    else:
        plan = [(i, *x) for i, x in enumerate(plan) if i % cli.of == cli.part]
    stem = "sensitivity" if (cli.of == 1 and not cli.only) else f"sensitivity_part{cli.tag or cli.part}"
    table = []
    for order, name, value, which in plan:
        args = copy.deepcopy(base)
        setattr(args, "train_offset" if name.startswith("train_offset") else name, value)
        print(f"== {name} = {value}", flush=True)
        extra = {}
        if name.startswith("train_offset"):
            extra = {"benign_rows": f"{value}-{value + width - 1}",
                     "capture_boundaries_inside": " ".join(map(str, crossing(value))),
                     "share_rows_repeating_earlier": float(repeated[value:value + width].mean())}
        for fam, vals in headline(args, which).items():
            table.append({"order": order, "setting": name, "value": value, "family": fam,
                          **extra, **vals})
        write_tables(cli.out_dir, stem, skipped, table)    # after every setting


def write_tables(out_dir, stem, skipped, table):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{stem}.json"), "w") as fh:
        json.dump({"skipped_offsets_mostly_repeated_rows": skipped, "rows": table}, fh, indent=2)
    first = ("order", "setting", "value", "family", "benign_rows", "capture_boundaries_inside",
             "share_rows_repeating_earlier")
    keys = sorted({k for r in table for k in r},
                  key=lambda k: (k not in first, first.index(k) if k in first else 0, k))
    with open(os.path.join(out_dir, f"{stem}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, restval=""); w.writeheader(); w.writerows(table)
    print(f"wrote {out_dir}/{stem}.json and .csv")


if __name__ == "__main__":
    main()
