"""Write the LaTeX tables of the sensitivity appendix and of the supplement
from sensitivity.csv (written by sensitivity.py), so that no number in those
tables is typed by hand.

Usage:
    python sensitivity_tables.py --csv results/sensitivity/sensitivity.csv \
        --out-dir results/sensitivity
Outputs:
    table_windows.tex      supplement (Table S2): one row per benign window
    table_contam.tex       supplement (Table S3): contamination and MCD settings, main window
    table_detect.tex       supplement (Table S4): threshold and ARA settings, main window
    table_full.tex         every setting and botnet, all columns (not printed in the
                           supplement; sensitivity.csv holds the same values)
"""
import argparse
import csv
import math
import os


def num(x):
    return None if x in ("", None) else float(x)


def f3(x):
    v = num(x)
    return "--" if v is None else f"{v:.3f}"


def f2(x):
    v = num(x)
    return "--" if v is None else f"{v:.2f}"


def fint(x):
    v = num(x)
    return "--" if v is None else f"{int(round(v))}"


def kfmt(x):
    v = num(x)
    return "--" if v is None else f"{v:g}"


def sci(v, digits=2):
    """LaTeX scientific notation, e.g. 1.8e-3 -> $1.8\\times10^{-3}$."""
    if v is None:
        return "--"
    if v == 0:
        return "0"
    e = int(math.floor(math.log10(abs(v))))
    m_s = f"{v / 10 ** e:.{digits - 1}f}"
    if m_s == "1":
        return f"$10^{{{e}}}$"
    if m_s.startswith("10"):            # rounding pushed the mantissa to 10
        e += 1
        m_s = f"{v / 10 ** e:.{digits - 1}f}"
    return f"${m_s}\\times10^{{{e}}}$"


def small(v):
    """Two significant digits at or above 0.1, scientific below."""
    if v is None:
        return "--"
    return f"{v:.2g}" if v >= 0.1 else sci(v)


def shift(x):
    """Location shift in Mahalanobis units: two decimals below 10, thousands
    separators up to 10^5, scientific above."""
    v = num(x)
    if v is None:
        return "--"
    if v < 10:
        return f"{v:.2f}"
    if v < 1e5:
        return f"{v:,.0f}".replace(",", "{,}")
    return sci(v, 1)


def e50(x):
    """Effectiveness at which the calibrated detector flags half of the attack
    observations: '<=1e-05' is the floor of the grid, '' means fewer than half
    even at full strength."""
    if x == "<=1e-05":
        return "$\\le10^{-5}$"
    if x in ("", None):
        return "$>1$"
    v = float(x)
    return "1" if v == 1.0 else small(v)


def masking(clean, cont):
    """Direction of the change in the calibrated naive detector's e50."""
    def val(s):
        return 1e-5 if s == "<=1e-05" else (2.0 if s in ("", None) else float(s))
    if clean == cont == "<=1e-05":
        return "floor"
    return "lost" if val(cont) > val(clean) else ("gained" if val(cont) < val(clean) else "same")


def pair(rows, col, fmt):
    g = [r for r in rows if r["family"] == "gafgyt"][0]
    m = [r for r in rows if r["family"] == "mirai"][0]
    return f"{fmt(g[col])} / {fmt(m[col])}"


def kstar(r):
    s = kfmt(r["ara_k_star"])
    if str(r.get("ara_indifferent_across_k", "")).lower() == "true":
        return s + "$^{\\ddagger}$"
    if len(str(r.get("ara_near_optimal_k", "")).split()) > 1:
        return s + "$^{\\dagger}$"
    return s


def by_setting(rows):
    out = {}
    for r in rows:
        out.setdefault((int(r["order"]), r["setting"], r["value"]), []).append(r)
    return dict(sorted(out.items()))


def label(setting, value):
    v = float(value) if setting not in ("cusum_h_mode",) else value
    return {
        "seed": lambda: f"Contamination draw {int(v)}",
        "contamination_frac": lambda: f"Contamination {v * 100:g}\\%",
        "mcd_support_fraction": lambda: f"MCD support {v:g}",
        "rank_tol": lambda: "Cut-off " + sci(v, 1),
        "arl0": lambda: f"ARL$_0$ target {int(v):,}".replace(",", "{,}"),
        "boot_block": lambda: f"Bootstrap block {int(v)}",
        "ara_kappa": lambda: f"Concentration {v:g}",
        "ara_sigma_k": lambda: f"Perception SD {v:g}",
        "cusum_h_mode": lambda: "Fixed $h=5$",
    }[setting]()


def table_windows(groups):
    lines = []
    for (order, setting, value), rows in groups.items():
        if setting not in ("train_offset", "train_offset_crossing"):
            continue
        r = rows[0]
        lo, hi = r["benign_rows"].split("-")
        rng = f"{int(lo):,}--{int(hi):,}".replace(",", "{,}")
        if setting == "train_offset_crossing":
            rng += "$^{*}$"
        g = [x for x in rows if x["family"] == "gafgyt"][0]
        m = [x for x in rows if x["family"] == "mirai"][0]
        mask = f"{masking(g['naive_clean_e50_calibrated'], g['naive_contaminated_e50_calibrated'])} / " \
               f"{masking(m['naive_clean_e50_calibrated'], m['naive_contaminated_e50_calibrated'])}"
        lines.append(" & ".join([
            rng, f3(r["t2_far_F_limit"]), f3(r["t2_far_empirical_limit"]),
            f"{fint(r['cusum_fa_k_tuned'])} / {fint(r['cusum_fa_k_conservative'])}",
            f2(r["mcd_far_clean"]), pair(rows, "mcd_location_shift_median", shift),
            mask, f"{kstar(g)} / {kstar(m)}"]) + r" \\")
    head = (r"Benign rows & \multicolumn{2}{c}{Naive $T^2$ FAR} & CUSUM FA & MCD & MCD shift & "
            r"Masking & $k^*$ \\" "\n"
            r"\cmidrule(lr){2-3}" "\n"
            r" & $F$ limit & Calibrated & $k=1$ / 3 & FAR & G / M & G / M & G / M \\")
    return head, lines


def table_contam(groups, base):
    lines = []
    for (order, setting, value), rows in [(k, v) for k, v in groups.items()
                                          if k[1] in ("seed", "contamination_frac", "mcd_support_fraction")]:
        lines.append(" & ".join([
            label(setting, value), pair(rows, "naive_far_contaminated", f3),
            f2(rows[0]["mcd_far_clean"]), pair(rows, "mcd_far_contaminated", f2),
            pair(rows, "mcd_location_shift_median", shift),
            pair(rows, "naive_contaminated_e50_calibrated", e50),
            pair(rows, "mcd_contaminated_e50_calibrated", e50)]) + r" \\")
    b = base
    first = " & ".join(["Main analysis (draw 0, 5\\%)", pair(b, "naive_far_contaminated", f3),
                        f2(b[0]["mcd_far_clean"]), pair(b, "mcd_far_contaminated", f2),
                        pair(b, "mcd_location_shift_median", shift),
                        pair(b, "naive_contaminated_e50_calibrated", e50),
                        pair(b, "mcd_contaminated_e50_calibrated", e50)]) + r" \\"
    head = (r"Setting & Naive FAR & \multicolumn{2}{c}{MCD FAR} & MCD shift & "
            r"\multicolumn{2}{c}{$e_{50}$, contaminated} \\" "\n"
            r"\cmidrule(lr){3-4}\cmidrule(lr){6-7}" "\n"
            r" & cont., G / M & clean & cont., G / M & G / M & Naive, G / M & MCD, G / M \\")
    return head, [first, r"\addlinespace"] + lines


def table_detect(groups, base):
    lines = []
    sel = [(k, v) for k, v in groups.items()
           if k[1] in ("rank_tol", "arl0", "boot_block", "ara_kappa", "ara_sigma_k", "cusum_h_mode")]
    for (order, setting, value), rows in [((0, "main", "main"), base)] + sel:
        r = rows[0]
        g = [x for x in rows if x["family"] == "gafgyt"][0]
        m = [x for x in rows if x["family"] == "mirai"][0]
        name = "Main analysis" if setting == "main" else label(setting, value)
        lines.append(" & ".join([
            name,
            f"{f2(r['cusum_h_k_tuned'])} / {f2(r['cusum_h_k_conservative'])}",
            f"{fint(r['cusum_fa_k_tuned'])} / {fint(r['cusum_fa_k_conservative'])}",
            f"{fint(g['cusum_delay_e1e-3_k_tuned'])} / {fint(g['cusum_delay_e1e-3_k_conservative'])}",
            f"{fint(m['cusum_delay_e1e-3_k_tuned'])} / {fint(m['cusum_delay_e1e-3_k_conservative'])}",
            (f"{kfmt(g['ara_fixed_h_k_star'])}$^{{\\S}}$ / {kfmt(m['ara_fixed_h_k_star'])}$^{{\\S}}$"
             if setting == "cusum_h_mode" else f"{kstar(g)} / {kstar(m)}"),
            ("--" if setting == "cusum_h_mode" else
             f"{sci(num(g['ara_damage_k_star']))} / {sci(num(m['ara_damage_k_star']))}")]) + r" \\")
        if setting == "main":
            lines.append(r"\addlinespace")
    head = (r"Setting & $h$ & CUSUM FA & \multicolumn{2}{c}{Delay at $e=10^{-3}$} & "
            r"$k^*$ & Damage at $k^*$ \\" "\n"
            r"\cmidrule(lr){4-5}" "\n"
            r" & $k=1$ / 3 & $k=1$ / 3 & G, $k=1$ / 3 & M, $k=1$ / 3 & G / M & G / M \\")
    return head, lines


FULL_COLS = [
    ("t2_far_F_limit", "FAR, $F$", f3), ("t2_far_empirical_limit", "FAR, cal.", f3),
    ("cusum_h_k_tuned", "$h_1$", f2), ("cusum_h_k_conservative", "$h_3$", f2),
    ("cusum_fa_k_tuned", "FA$_1$", fint), ("cusum_fa_k_conservative", "FA$_3$", fint),
    ("cusum_delay_k_tuned", "D$_1$", fint), ("cusum_delay_k_conservative", "D$_3$", fint),
    ("ara_k_star", "$k^*$", kfmt), ("ara_k_star_test_false_alarms", "FA$_{k^*}$", fint),
    ("ara_fixed_h_k_star", "$k^*_{h=5}$", kfmt),
    ("exact_fit_state_share_test", "Exact fit", f2), ("mcd_rank_clean_median", "Rank", fint),
    ("mcd_far_clean", "MCD", f2), ("naive_far_contaminated", "Naive c.", f3),
    ("mcd_far_contaminated", "MCD c.", f2),
    ("naive_clean_e50_calibrated", "$e_{50}$ n.", e50),
    ("naive_contaminated_e50_calibrated", "$e_{50}$ n.c.", e50),
    ("mcd_location_shift_median", "MCD shift", shift),
]


def table_full(groups):
    lines = []
    for (order, setting, value), rows in groups.items():
        for r in rows:
            name = (f"Rows {r['benign_rows']}" if setting.startswith("train_offset")
                    else label(setting, value))
            contam_run = r.get("naive_far_contaminated", "") != ""
            cells = [name, "G" if r["family"] == "gafgyt" else "M"] + [
                "--" if (fmt is e50 and not contam_run) else fmt(r.get(col, ""))
                for col, _, fmt in FULL_COLS]
            lines.append(" & ".join(cells) + r" \\")
    head = "Setting & & " + " & ".join(h for _, h, _ in FULL_COLS) + r" \\"
    return head, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv)))
    groups = by_setting(rows)
    base = [r for r in rows if r["setting"] == "train_offset" and r["value"] == "3"]
    for name, (head, lines) in {"table_windows": table_windows(groups),
                                "table_contam": table_contam(groups, base),
                                "table_detect": table_detect(groups, base),
                                "table_full": table_full(groups)}.items():
        with open(os.path.join(a.out_dir, name + ".tex"), "w") as f:
            f.write(head + "\n\\midrule\n" + "\n".join(lines) + "\n")
        print("wrote", name)


if __name__ == "__main__":
    main()
