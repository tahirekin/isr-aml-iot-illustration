"""Run Algorithms 1-5 of Section 4 in sequence on one device and collect the
numbers the manuscript reports.

Each alg*.py script also runs on its own; this driver only calls their run()
functions with one shared set of options so that every algorithm uses the same
data split, preprocessing, and CUSUM threshold.

Usage:
    python run_all.py --data-dir /path/to/Danmini_Doorbell --out-dir results
Outputs (in --out-dir):
    section4_summary.json   naive T^2, CUSUM, and contamination results, keyed as in
                            the [PLACEHOLDER: ...] markers of Section 4
    ara_summary.json        ARA companion results (Section 4.5.1)
    alg1..alg5 JSON files and the figures written by each script
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alg1_data_setup as a1          # noqa: E402
import alg2_hotelling_t2 as a2        # noqa: E402
import alg3_cusum_t2 as a3            # noqa: E402
import alg4_contamination_mcd as a4   # noqa: E402
import alg5_ara_choice_of_k as a5     # noqa: E402


def main():
    ap = a1.add_common_args(argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.001])
    ap.add_argument("--cusum-k-tuned", type=float, default=1.0)
    ap.add_argument("--cusum-k-conservative", type=float, default=3.0)
    a3.add_cusum_args(ap)
    a3.add_h_args(ap)
    ap.add_argument("--contamination-frac", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=0.01, help="level for the contamination demo")
    ap.add_argument("--mcd-support-fraction", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mcd-starts", type=int, default=5)
    ap.add_argument("--mcd-random-state", type=int, default=0)
    ap.add_argument("--k-min", type=float, default=0.0)
    ap.add_argument("--k-max", type=float, default=6.0)
    ap.add_argument("--k-n", type=int, default=25)
    ap.add_argument("--e-per-decade", type=int, default=10)
    ap.add_argument("--e-min", type=float, default=1e-6)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--c-delay", type=float, default=0.0)
    ap.add_argument("--c-fa", type=float, default=1.0)
    ap.add_argument("--c-damage", type=float, default=1.0)
    ap.add_argument("--fa-cap", type=float, default=5.0)
    ap.add_argument("--attacker-types", default=None)
    ap.add_argument("--ara-kappa", type=float, default=20.0)
    ap.add_argument("--ara-sigma-k", type=float, default=0.5)
    ap.add_argument("--ara-draws", type=int, default=4000)
    ap.add_argument("--ara-onsets", type=int, default=20)
    ap.add_argument("--ara-seed", type=int, default=0)
    ap.add_argument("--ara-tol", type=float, default=1e-9)
    ap.add_argument("--skip-ara", action="store_true")
    args = ap.parse_args()
    # Algorithm 5 uses the same k choices as Algorithm 3
    args.k_tuned, args.k_conservative = args.cusum_k_tuned, args.cusum_k_conservative
    out = args.out_dir
    print("== Algorithm 1: data setup"); r1 = a1.run(args); a1.save_json(r1, os.path.join(out, "alg1_data_setup.json"))
    print("== Algorithm 2: naive Hotelling T^2"); r2 = a2.run(args); a2.save_json(r2, os.path.join(out, "alg2_hotelling_t2.json"))
    print("== Algorithm 3: CUSUM on T^2"); r3 = a3.run(args); a3.save_json(r3, os.path.join(out, "alg3_cusum_t2.json"))
    print("== Algorithm 4: contamination and MCD"); r4 = a4.run(args); a4.save_json(r4, os.path.join(out, "alg4_contamination_mcd.json"))
    summary = {"meta": {**r1["meta"], "raw_scale": r1["raw_scale"]},
               "t2_control_limits": r2["t2_control_limits"],
               "benign_t2_quantiles": r2["benign_t2_quantiles"],
               "projected_training_excess_kurtosis": r2["projected_training_excess_kurtosis"],
               "CUSUM_K_CHOICES": r3["CUSUM_K_CHOICES"], "CUSUM_H_IN_SIGMAS": r3["CUSUM_H_IN_SIGMAS"],
               "cusum_reference": r3["cusum_reference"],
               "cusum_h_design": r3["cusum_h_design"],
               "diagnostic_two_sided_gaussian_reference":
                   r3["diagnostic_two_sided_gaussian_reference"],
               "per_attack": {f: {**r2["per_attack"][f],
                                  "cusum_on_naive_t2": r3["per_attack"][f]["cusum_on_naive_t2"],
                                  "cusum_on_naive_t2_fixed_h":
                                      r3["per_attack"][f]["cusum_on_naive_t2_fixed_h"]}
                              for f in args.families},
               "contamination_demo": r4["contamination_demo"],
               "contamination_settings": r4["settings"]}
    a1.save_json(summary, os.path.join(out, "section4_summary.json"))
    if not args.skip_ara:
        print("== Algorithm 5: ARA choice of k"); r5 = a5.run(args)
        a5.save_json(r5, os.path.join(out, "alg5_ara.json"))
        a5.save_json(r5, os.path.join(out, "ara_summary.json"))


if __name__ == "__main__":
    main()
