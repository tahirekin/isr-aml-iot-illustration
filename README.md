# Section 4 code: statistical anomaly detection on N-BaIoT

Code for Section 4 ("Illustrative Application") of *A Statistical Perspective on
Adversarial Machine Learning in the Internet of Things* (Ekin and Shaw).
There is one script per algorithm. Each script is self-contained: the
data-loading and preprocessing block is copied verbatim into every file, so any
one of them can be run, read, or shared on its own.

| Script | Manuscript | What it does | Main outputs |
|---|---|---|---|
| `alg1_data_setup.py` | 4.1 | Loads Danmini Doorbell, splits benign/attack blocks, z-scores with training moments, projects onto the non-degenerate principal subspace | `alg1_data_setup.json`, `section4_eigenvalues.png` |
| `alg2_hotelling_t2.py` | 4.2.1 | Naive Hotelling T² with the Phase II F limit at α = 0.01, 0.001: false alarms before onset (with moving-block bootstrap intervals), detection delay, and the same rates under a limit calibrated at the Phase I 99% quantile | `alg2_hotelling_t2.json`, `section4_tsq_traces.png` |
| `alg3_cusum_t2.py` | 4.2.2 | Page's one-sided (upper) CUSUM on standardized log T² at a tuned and a conservative k, each with the threshold h set for a common in-control ARL (Phase I stationary bootstrap); delays for weakened attacks; the fixed-h results and the textbook two-sided raw-T² CUSUM (466 benign alarms at k = 1) as comparisons | `alg3_cusum_t2.json`, `section4_cusum.png` |
| `alg4_contamination_mcd.py` | 4.3 | Replaces a fraction of training rows with attack rows; compares naive T² and MCD-based T², each against its own clean-training baseline: false alarms on the clean test block at the F limit, and, at limits calibrated on Phase I data, false alarms and detection of weakened attacks; block-bootstrap intervals; how far each estimate moves (location, and variance along the attack direction) in fixed coordinates; MCD rates are medians over five random starts with their range, and the exact-fit diagnostics (rank, the 56 constant raw features, rates inside and outside the exact-fit state) are in the JSON | `alg4_contamination_mcd.json`, `section4_contamination.png` |
| `alg5_ara_choice_of_k.py` | 4.4 | The defender chooses k, from Phase I data, against an attacker who chooses its effectiveness e (a throttle; the code parameterizes it as δ = 1 − e): ARA over three attacker types (aggressive, balanced, stealthy; weights on effectiveness 0.8, 0.5, 0.2) with uncertainty about the attacker's weights and its perception of k (Monte Carlo, 4,000 draws), and the known-types game as a limiting case; thresholds for a common ARL (main) or h fixed (comparison); k* against the false-alarm cost; evaluation on the test sequence | `alg5_ara.json`, `section4_ara_tradeoff.png` (Figure 4), `section4_ara_cost.png` (Supplementary Figure S3) |
| `run_all.py` | all | Runs 1–5 with one shared set of options and writes the two summary files the manuscript reads from | `section4_summary.json`, `ara_summary.json` |
| `sensitivity.py` | robustness | Re-runs 2–5 with the benign blocks moved through the recording (one window per stretch of a capture), other contamination draws and fractions, other MCD support fractions, other eigenvalue cut-offs, other ARL targets and bootstrap block lengths, other ARA uncertainty settings, and fixed h = 5 | `sensitivity.json`, `sensitivity.csv` |
| `sensitivity_tables.py` | tables | Writes the LaTeX tables of the supplementary material (Table S2 for the windows, Table S3 for contamination and MCD, Table S4 for the threshold and ARA settings); `sensitivity.csv` holds every setting and output from `sensitivity.csv`, so that no number in them is typed by hand | `table_windows.tex`, `table_contam.tex`, `table_detect.tex`, `table_full.tex` |

## Requirements

Python ≥ 3.9 with `numpy`, `pandas`, `scipy`, `scikit-learn` (Algorithm 4 only)
and `matplotlib`. Tested with numpy 2.5, pandas 3.0, scipy 1.18,
scikit-learn 1.9 and matplotlib 3.11.

```
pip install -r requirements.txt
```

## Data

N-BaIoT (Meidan et al., 2018), UCI Machine Learning Repository. Only the
Danmini Doorbell device is used, with one attack file per botnet family:
`gafgyt_attacks/combo.csv` and `mirai_attacks/syn.csv`. Point `--data-dir` at the
device folder. Both common layouts are recognized:

```
UCI layout                               Kaggle layout
Danmini_Doorbell/                        nbaiot/
  benign_traffic.csv                       1.benign.csv
  gafgyt_attacks/combo.csv                 1.gafgyt.combo.csv
  mirai_attacks/syn.csv                    1.mirai.syn.csv
```

With the Kaggle layout, put only the Doorbell (device 1) files in the folder,
because the finder matches on `*benign*.csv` and `*gafgyt*combo*.csv`.

## Running

The usual route, which produces every number in Section 4:

```
python run_all.py --data-dir /path/to/Danmini_Doorbell --out-dir results
```

Each algorithm can also be run alone with the same arguments:

```
python alg2_hotelling_t2.py --data-dir /path/to/Danmini_Doorbell --out-dir results
```

`python <script> --help` lists every option. The defaults are the settings the
manuscript describes:

| Option | Default | Meaning |
|---|---|---|
| `--n-train` | 2000 | benign rows used to fit (μ, Σ) |
| `--n-calib` | 1000 | next benign rows, used only for calibration (empirical limit, CUSUM reference) |
| `--n-test-benign` | 1000 | next benign rows, held out; they precede the attack onset |
| `--n-attack-used` | 1000 | attack rows after onset (per family) |
| `--keep-restart-rows` | off | keep the feature-extractor restart row(s) at the start of each attack file (dropped by default) |
| `--train-offset` | auto | first benign row used; by default the start of the first capture long enough for all three benign blocks (row 3 on the Doorbell) |
| `--rank-tol` | 1e-8 | eigenvalues below this times the largest are treated as zero |
| `--families` | gafgyt mirai | attack families to run |
| `--alphas` | 0.01 0.001 | levels for the T² limit (Alg. 2) |
| `--cusum-k-tuned`, `--cusum-k-conservative` | 1.0, 3.0 | CUSUM reference values, in in-control SDs (`--k-tuned`, `--k-conservative` when running `alg5` alone) |
| `--cusum-h-mode` | arl | `arl`: h for each k is the smallest value whose in-control ARL reaches `--arl0`; `fixed`: h = `--cusum-h-sigmas` |
| `--arl0` | 1000 | target in-control average run length, in observations |
| `--boot-block`, `--boot-replicates`, `--boot-length`, `--boot-seed` | 50, 400, 20000, 0 | stationary bootstrap of the Phase I series used to estimate the ARL |
| `--cusum-h-sigmas` | 5.0 | CUSUM threshold h for the fixed design, in in-control SDs |
| `--cusum-statistic` | log | monitor log T² (`log`) or T² (`raw`) |
| `--cusum-reference` | calibration | in-control mean/SD from the calibration block; `theoretical` (Gaussian F model) and `insample` (training rows) for comparison |
| `--contamination-frac` | 0.05 | share of training rows replaced (Alg. 4) |
| `--alpha` | 0.01 | level for the contamination comparison |
| `--seed` | 0 | which spare attack rows are injected (Alg. 4) |
| `--mcd-starts`, `--mcd-random-state` | 5, 0 | FastMCD random starts (Alg. 4); MCD results are the median over starts, with the range |
| `--k-min`, `--k-max`, `--k-n` | 0, 6, 25 | k grid for ARA (Alg. 5); the tuned and conservative k are added |
| `--e-per-decade`, `--e-min` | 10, 1e-6 | attacker's grid: 1 − δ log-spaced on [e_min, 1], plus δ = 1; e_min is lowered automatically when the attacks are so far from benign data that 1e-6 would still be detected |
| `--c-damage`, `--c-fa`, `--c-delay`, `--fa-cap` | 1, 1, 0, 5 | defender utility weights (see design note 9) |
| `--attacker-types` | built-in three types | JSON string `'{"name": {"prior": .., "w_eff": .., "w_delay": ..}, ...}'` |
| `--ara-kappa`, `--ara-sigma-k`, `--ara-draws`, `--ara-seed`, `--ara-onsets` | 20, 0.5, 4000, 0, 20 | ARA: Beta concentration of each type's weight, SD of the attacker's error in perceiving k, Monte Carlo draws, seed, and onset times over which delays are averaged |
| `--ara-tol` | 1e-9 | differences in expected utility below this are ties, broken toward the smallest k |
| `--ci-block`, `--ci-replicates`, `--ci-seed` | 50, 2000, 0 | moving-block bootstrap for the intervals (Algs. 2 and 4) |

On one core, `run_all.py` takes about five minutes and `sensitivity.py` about
an hour and a half. The sensitivity run can be split across processes with
`--part i --of n` (or `--only` with a list of setting indices and `--tag`); each
part writes its table after every setting, and `--merge` combines the parts
into `sensitivity.json` and `.csv`. Then
`python sensitivity_tables.py --csv results/sensitivity/sensitivity.csv --out-dir results/sensitivity`
writes the appendix and supplement tables.

## Where the manuscript's numbers come from

Every number in Section 4 is read from `results/section4_summary.json`,
`results/ara_summary.json`, or `results/alg4_contamination_mcd.json` (the
calibrated-limit and exact-fit diagnostics). The figures map as follows:
`section4_contamination.png` is Figure 3, `section4_ara_tradeoff.png` is
Figure 4, and in the supplementary material `section4_tsq_traces.png` is Figure
S1, `section4_cusum.png` is Figure S2 and `section4_ara_cost.png` is Figure S3
(`section4_eigenvalues.png` is a diagnostic and is not in the manuscript). The robustness statements in Section 4 come from
`results/sensitivity/sensitivity.csv`.

The `results/` folder holds the outputs of the run on the real Danmini Doorbell
data that the manuscript reports (`results/*.json`, `results/*.png`) and of the
sensitivity run (`results/sensitivity/`: the merged `sensitivity.csv` and `.json`
and the four LaTeX tables). Rerunning the pipeline on the same data regenerates
them.

## Tests

```
python tests/test_math.py
```

These checks cover the following: the size of the T² limit under simulated
Gaussian data (empirical 0.0099 against nominal 0.01); the F-based null mean and
variance of T² and of log T²; that the projected T² equals the pseudo-inverse
T²; the calibration reference; the one-sided CUSUM recursion and alarm
bookkeeping on a hand-worked series; the restart-row rule; the robust-distance
rank rule, including an exact-fit case; the throttle identity
T²(δ) = (1 − δ)² T²; that the vectorized CUSUM used by Algorithm 5 matches the
scalar one; that the stationary bootstrap keeps blocks of the right mean length,
that the alarm rate matches a direct count, and that on i.i.d. normal data the
estimated ARL at k = 0.5, h = 4 is close to the known value (about 336); that
`calibrate_h` returns the smallest h meeting the target; and the block-bootstrap
intervals.

`tests/make_synthetic_fixture.py` writes a fake device folder (115 Kitsune
column names, rank 80, scales spanning 15 orders of magnitude, AR(1) rows) so
the pipeline can be exercised end to end without the data. **Numbers from the
fixture are for testing the code only and must not appear in the paper.**

## Design notes

These are the choices a reader of Section 4 should know about. Where an
alternative is reasonable, it is a command-line option. Several of them were
made after a first run on the real data showed that the textbook default failed;
those failures are described so that the choices can be checked.

1. **Blocks in time order, inside one capture.** N-BaIoT's benign file joins
   several captures end to end; at each boundary the feature extractor restarts
   and every long-window weight (`*_L0.01_weight`) returns to 1. The Doorbell's
   first capture is three rows long, so by default the benign blocks start at
   row 3: training (rows 3–2002) to estimate (μ, Σ), a calibration block
   (2003–3002), then the held-out test block (3003–4002), followed by the
   attack rows. The calibration block is the Phase I sample of classical SPC:
   it sets empirical limits and the CUSUM reference and is never used to
   estimate (μ, Σ) or to count false alarms. A window that crosses a capture
   boundary prints a warning; in the sensitivity run such a window raises the
   F-limit false-alarm rate to about 0.7, which is a change in benign data, not
   an attack.
2. **Attack-file restart rows.** Each attack CSV is also a separate capture, so
   its first row is a restart row (window weight 1; T² ≈ 3,800 for Gafgyt and
   4,600 for Mirai, the same order as benign restart rows). A detector that
   flags it is reacting to the restart, not to the attack, so leading rows with
   `MI_dir_L0.01_weight` ≤ 1 are dropped (one row per file);
   `--keep-restart-rows` keeps them.
3. **Standardize, then project.** The raw features span about 18 orders of
   magnitude (the jitter variances reach 10^16), and the standardized training
   covariance has no clear eigen-gap: its spectrum decays smoothly over about
   ten orders of magnitude before dropping to numerical zero. T² is computed in the eigen-subspace
   whose eigenvalues exceed `--rank-tol` (10^-8) times the largest, which gives
   p_eff = 81 on the Doorbell. The Phase II limit uses p_eff. `sensitivity.py`
   repeats the detection results at 10^-10, 10^-6, and 10^-4.
4. **F limit vs calibrated limit.** The F limit assumes i.i.d. Gaussian rows.
   Benign Doorbell traffic is far from that: the projected coordinates have
   excess kurtosis between 11 and 342, and the median T² of new benign rows
   (10 to 12) is far below the median of the Gaussian reference (84), while a
   few bursts reach T² in the thousands. Algorithm 2 therefore reports both the
   F-limit false-alarm rate and the rate under a limit calibrated on the Phase I
   block (its 99% quantile). The 99.9% quantile of 1,000 calibration rows is
   essentially the sample maximum and should not be reported as a limit.
5. **CUSUM: one-sided, on log T², calibrated.** Three choices, each made
   because the alternative failed on the real data in a first run:
   * one-sided: a two-sided CUSUM with Gaussian reference moments raised 466
     alarms on 1,000 benign rows at k = 1, 410 of them from the lower arm,
     because benign T² is usually far *below* its Gaussian mean. A low T² is not
     evidence of an attack;
   * calibration reference: the Gaussian moments put the in-control mean at 85
     when the benign median is 12;
   * log T²: on the raw scale the calibration SD is dominated by a few bursts
     (SD ≈ 1,580 against a median of 12), so k and h "in SDs" mean little.
     The log is the usual variance-stabilizing transform for a scale statistic,
     and it makes the attacker's throttle an additive shift.
6. **MCD and the exact-fit property.** About 58% of the Doorbell's benign
   training rows (and 62% of the test rows) sit in one recurring state in which
   56 of the 115 raw features, all of them host-to-host channel statistics
   (`HH_*`, `HpHp_*`, and six `HH_jit_*`), take exactly the same values. FastMCD's
   h-subset (default h ≈ n/2) lies entirely inside that state, so the robust
   scatter is singular (rank 19 or 20 of 81). Robust distances use the same rank
   rule as the naive T², and the F limit uses that rank. On the clean fit MCD
   flags every benign test row outside the state and 21% of the rows inside it,
   for an overall false-alarm rate of about 0.5; the JSON reports these splits
   (`exact_fit_state_first_start`), the constant features, and the number of
   numerical warnings from the fits. The high rate is a property of the data and
   the estimator, not a coding error; `sensitivity.py` repeats it at support
   fractions 0.75 and 0.9.
   Because the selected half-sample is singular, FastMCD's result is numerically
   fragile: scikit-learn warns that the determinant increased during the
   concentration steps, and an orthogonal rotation of the input or a different
   random start changes which half-sample is found. The false-alarm rate barely
   moves (0.47–0.51 over five starts), but detection of moderately weakened
   attacks does (for Gafgyt at e = 0.01, from 0.27 to 0.84). MCD is therefore
   fitted from five random starts; every MCD rate is reported as the median
   over starts with its range, and the figure shows the range as a band.
   The fit is sensitive even to the floating-point order of operations: with
   the same random_state, a run with several BLAS threads finds different
   half-samples than a single-threaded run. Every script therefore fixes the
   BLAS thread count to one (environment variables plus `threadpoolctl`, set in
   the shared data-setup block), so results are identical across machines.
7. **Contamination: false alarms and masking.** Injected rows come from attack
   rows that are never part of the test sequence. The standardization,
   projection, and both estimators are refit on the contaminated block, as a
   defender who did not know about the contamination would do, and each
   estimator is also fitted to the clean block, so contamination is judged
   against that estimator's own baseline. A false-alarm rate alone can mislead
   here: contamination inflates the covariance in the attack direction, which
   *lowers* the false-alarm rate while blinding the detector. So each of the
   four detectors is also calibrated on the Phase I block (its own 99%
   quantile), and its detection rate is reported against weakened attacks
   e·x_attack + (1 − e)·μ_benign over a log grid of e.
8. **CUSUM threshold for a common in-control ARL.** Comparing k values at one
   fixed h compares charts held to very different false-alarm standards: at
   h = 5 the in-control ARL is about 255 observations for k = 1 and about 2.7
   million for k = 3. By default h is therefore chosen, for each k, as the
   smallest value whose in-control ARL (mean number of observations between
   false alarms, chart with reset) is at least 1,000 observations, estimated
   from 400 stationary-bootstrap resamples (Politis and Romano, 1994; mean block
   length 50) of 20,000 observations each, drawn from the standardized Phase I
   series so that its bursts and short-range dependence are kept. This gives
   h = 7.47 at k = 1 and h = 2.51 at k = 3. The Phase I sample can only certify
   an ARL for k below its largest standardized value (5.36 here): for larger k
   no resampled observation ever raises an alarm, so those k are left out of
   the ARA. In units of the SD of log T², k is half the shift the chart is
   tuned to, so k = 1 targets T² multiplied by about 19 and k = 3 by about
   6,600.
9. **ARA grid.** The recorded attacks sit far from benign data (median T²
   around 10^9), so evading the CUSUM needs 1 − δ of order 10^-4 to 10^-2. The
   attacker's grid is therefore log-spaced in 1 − δ (10 values per decade down
   to 10^-6, lowered automatically if needed, plus δ = 1). On a uniform 51-point
   δ grid every δ < 1 is detected, and the best responses are degenerate.
10. **ARA utilities.** The type weights, prior, and defender costs are
    illustrative inputs, not estimates. The defender's utility is
    U_D = −c_damage (1 − δ) D − c_delay D − c_fa F(k), where D is the share of
    the horizon before detection (1 if never detected) and F(k) the false-alarm
    burden, min(1, alarms per horizon / 5). The damage term is effectiveness
    times time undetected, so detection limits damage and a nearly fully
    throttled attack does little harm even if it is never caught. An earlier
    version charged c_damage (1 − δ) whether or not the attack was detected; with
    it the defender's utility was the same for every k without false alarms, so
    k* was set by false alarms alone. c_delay is kept, at 0.
11. **ARA with uncertainty about the attacker.** In the main analysis the
    defender does not know the attacker's weights or what the attacker believes
    k to be: a draw picks a type from the prior, a weight
    w ~ Beta(κ m, κ (1 − m)) around the type's weight m (κ = 20), and a perceived
    k~ = k + N(0, 0.5²); the attacker best-responds to k~. Monte Carlo over
    4,000 draws gives p_D(δ | k), the expected damage, and its standard error
    (common random numbers across k). The known-types game (each type
    best-responds to the true k) is reported as the limiting case.
12. **ARA uses Phase I data for the decision.** In the defender's model the
    benign stretch before the attack is the calibration block, and delays are
    averaged over 20 onset times (benign prefixes of 100 to 1,000 rows), because
    the CUSUM state at the onset depends on the traffic just before it. Under the
    common-ARL design the false-alarm burden is the design value, the same for
    every k, so k* is decided by detection alone; under the fixed-h design the
    burden is the number of alarms on the calibration block, and the script
    reports k* across a grid of false-alarm costs. k*, k = 1, and k = 3 are then
    evaluated on the held-out test sequence with the alarms actually raised
    there.
13. **What decides k\*.** Under the common-ARL design, k* is the smallest k at
    which the full-strength attack is flagged on its first observation (k + h(k)
    falls below that observation's standardized log T²); above it, raising k
    only lets throttled attackers keep more effectiveness. Because this turns on
    a single observation, k* should be read as the location of that switch, not
    as a finely estimated optimum, and it moves when the thresholds move (bootstrap
    block length, ARL target, eigenvalue cut-off; see `sensitivity.py`). When the
    switch moves to a k at which throttled attackers keep more, the minimum can
    instead sit at the smallest k that catches the attack after one observation.
    Expected utilities within `--ara-tol` (default 1e-9) are treated as ties and
    broken toward the smallest k. This matters in benign windows where the attacks
    are so far from benign data that no throttle escapes detection at any k: the
    expected damage is then zero to rounding error at every k, and without the
    tolerance k* would be decided by rounding. The JSON reports
    `indifferent_across_k` and the full near-optimal set.
14. **Intervals.** 95% intervals are moving-block bootstrap percentile
    intervals (Künsch, 1989; block 50, 2,000 replicates). For F-limit rates the
    fitted estimates and limits are held fixed and test rows are resampled; for
    calibrated limits the calibration and test (or attack) rows are resampled
    together, so the interval includes the uncertainty of the limit.
15. **How far the estimates move.** Algorithm 4 also measures, in one fixed
    coordinate system (the clean-training standardization and projection,
    whitened by the naive clean covariance), how far each estimator moves under
    contamination: the shift of its location estimate in Mahalanobis units and
    its estimated variance along the attack direction. This separates the
    estimator from the preprocessing, which a defender refits to the
    contaminated data.

## References

Meidan, Y., Bohadana, M., Mathov, Y., Mirsky, Y., Shabtai, A., Breitenbacher,
D. and Elovici, Y. (2018). N-BaIoT: Network-based detection of IoT botnet attacks
using deep autoencoders. *IEEE Pervasive Computing* 17(3), 12–22.

Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary
observations. *The Annals of Statistics* 17(3), 1217–1241.

Politis, D. N. and Romano, J. P. (1994). The stationary bootstrap. *Journal of
the American Statistical Association* 89(428), 1303–1313.
