# Changelog

## Unreleased — re-review hardening pass (no result changes)

A verification re-review after the previous pass surfaced nine residual issues (six
in the manuscript, three in the code). All fixed here; the two code changes are
hardening of currently-unreachable paths, and both result CSVs were re-verified
**bit-exact identical** before/after this pass.

Code:
- `knn_impute()` no-donor fallback: the fallback's Monte Carlo resampling weights
  previously pointed at donor slots that are all masked at that row — a dormant
  leak of the same kind as the one fixed in the previous pass, inert on the reported
  12x3 grid (verified: the branch fires zero times) but live if rates/seeds are
  extended. The fallback row's donor values are now overwritten with the masked-aware
  fallback estimate, making resampling degenerate and leak-free by construction.
- `repair_one()` (both scripts) and the KNN fallback: the fully-masked-row last
  resort could evaluate an all-NaN mean; now guarded (row mean of observed cells,
  0.0 if none). Unreachable at tested rates.

Manuscript (text only, no numbers changed):
- Fig. 3 caption updated to cover all three patterns (was stale: "random-loss and
  supplier-block" after the sweep was extended).
- §4.2 "in most pattern-rate combinations it fell again beyond its peak" corrected
  to the actual count in the full 12-condition sweep: 5 of 12 decline past their
  peak; the other 7 (incl. all FLOW rates) rise toward a plateau far below nominal.
  The 45.2% "peak" is now explicitly scoped to the BLOCK pattern.
- Abstract's "restores near-nominal calibration" softened (KNN's range is 66-88%
  against a 95% target).
- §4.5 "vary between runs" claim reworded — it dated from before the seed fix, when
  reruns genuinely were non-deterministic; under fixed seeds it contradicted the
  bit-exact reproducibility claim.
- Materials and Methods now discloses the column-sum stabilization rescale in
  `impacts()` (fires only on repaired matrices, mostly MCAR@20/40%; the true matrix
  never approaches the bound) and the per-method Monte Carlo sample counts
  (80/60/40; 150 in the sweep).
- Table 1: GNN row now carries the same † instability annotation as VAE/SoftImpute
  (its FLOW@40% spike is 120%).

## Unreleased — data-leakage fix, GSD-sweep FLOW bug fix, manuscript corrections

This pass followed an independent 5-reviewer methodological audit (Editor-in-Chief +
Methodology + Domain + Cross-disciplinary + Devil's Advocate) of the manuscript and
this repository, which found issues beyond what the previous pass's numeric-fidelity
check (quoted numbers vs. CSVs) had caught. All are fixed here.

### 1. Data leakage in `repair_baseline()` / `knn_impute()`

Both functions computed sibling/donor averages by reading `A0`/`B0` — the TRUE,
unmasked matrices — directly. Above the lowest missingness rates, more than one cell
in a row is typically masked at once, so a "sibling average" for one masked cell could
be computed partly from other cells that were themselves masked in the same run:
information that would not actually be observable in a real missing-data scenario.

**Fix**: both functions now average only over sibling/donor entries that are not
themselves masked at that row, with a widened fallback (then a row mean as a last
resort) when nothing is observed. `knn_impute()`'s donor-distance calculation and
donor-resampling weights were similarly made mask-aware. Applied identically in
`experiment_final_eu.py` and `gsd_sensitivity.py`.

**Effect**: PEDIGREE and KNN numbers moved under MCAR and FLOW (both patterns where a
row can have some cells masked and others not); BLOCK was unaffected (BLOCK masks
whole columns, so the pre-fix "sibling" pool for a masked column already excluded
other fully-masked columns in the common case). SoftImpute, VAE, and GNN were already
leakage-free (confirmed bit-identical before/after) — none of their training or
imputation logic reads the unmasked matrix for masked cells.

Condition-level before -> after (mean of 3 seeds, percentage points):

| method | pattern | rate | MdAPE before | MdAPE after | Coverage before | Coverage after |
|---|---|---|---|---|---|---|
| PEDIGREE | MCAR | 5% | 3.1% | 3.2% | 10.1% | 9.6% |
| PEDIGREE | MCAR | 10% | 5.3% | 5.8% | 30.7% | 28.1% |
| PEDIGREE | MCAR | 20% | 18.7% | 20.9% | 10.1% | 9.2% |
| PEDIGREE | MCAR | 40% | 25.1% | 33.6% | 15.8% | 11.0% |
| PEDIGREE | FLOW | 5% | 0.5% | 0.9% | 80.3% | 78.1% |
| PEDIGREE | FLOW | 10% | 3.2% | 6.8% | 84.6% | 78.9% |
| PEDIGREE | FLOW | 20% | 6.3% | 9.1% | 78.9% | 73.7% |
| PEDIGREE | FLOW | 40% | 9.2% | 11.8% | 54.4% | 50.0% |
| KNN | MCAR | 5-40% | 3.4-32.8% | 2.7-40.9% | 72.4-97.8% | 66.2-88.2% |
| KNN | FLOW | 5-40% | 1.0-6.1% | 1.0-10.8% | 82.9-95.2% | 78.9-84.6% |
| KNN, PEDIGREE | BLOCK, all rates | — | unchanged | unchanged | unchanged | unchanged |

No qualitative finding in the manuscript changed as a result: the pedigree baseline
still never approaches nominal coverage anywhere, KNN is still better-calibrated than
pedigree in nearly every entrywise condition (now 7 of 8, with one exact tie rather
than 8 of 8), and BLOCK's ordering of methods by accuracy is unchanged.

### 2. Latent FLOW mask-generation bug in `gsd_sensitivity.py`

Discovered while fixing issue #3 below. `gsd_sensitivity.py`'s `make_masks()` only ever
distinguished `pattern == 'MCAR'` from an `else` branch — there was no `elif pattern ==
'FLOW'`. This was harmless as long as the sweep never included FLOW (the previous
sweep list was `[('MCAR',0.20), ('BLOCK',0.20), ('MCAR',0.40), ('BLOCK',0.40)]`), but
the moment FLOW was added to the sweep (issue #3), it silently fell through to the
BLOCK-style column-masking branch, producing masks nothing like the FLOW pattern used
everywhere else in the paper (84 cells masked via column-blocking vs. the ~32-84 cells
a true row-structured FLOW mask produces) and coverage numbers around 15-20% where the
correct FLOW branch gives 75-85%. Fixed to mirror `experiment_final_eu.py`'s FLOW
branch exactly; re-verified bit-exact reproducible after the fix.

### 3. GSD sweep excluded FLOW entirely

The sweep previously covered only MCAR and BLOCK at the 20% and 40% rates — 4 of the
12 pattern-rate conditions in the main grid — and specifically excluded FLOW@10%, the
single condition where pedigree's coverage comes closest to nominal anywhere in the
main grid (84.6% -> 78.9% after the issue-#1 fix). The manuscript's "not a GSD
artifact" claim was drawn from this partial grid. Extended to the full 3-pattern x
4-rate (12-condition) grid. Result: even at FLOW@10%, coverage only reaches 84.6% at
the widest GSD tested (3.0), and no condition anywhere in the resulting 60
pattern-rate-GSD combinations reaches nominal 95% coverage — the claim is now
evidenced across the full grid rather than 4 of 12 conditions, and holds up.

### 4. Table 1's Coverage column didn't represent its own caption

The table was captioned "Summary at 40% missingness rate under supplier
non-disclosure (BLOCK)"; the MdAPE column genuinely was BLOCK@40% for every method,
but the Coverage column mixed in other conditions (KNN's "72-98%" was footnoted as
"range across MCAR/FLOW patterns, all rates"; GNN's "0.9-3.5%" was footnoted as "the
random-loss condition specifically"). Fixed: every column now reports the actual
BLOCK@40% value per method, pulled from `results/results_final_eu_summary.csv`.

### 5. Two manuscript-only disclosures added (no code or data change)

- KNN's donor search under BLOCK routinely has no jointly-observed candidate and falls
  back to the same sibling-averaging rule as the pedigree baseline (already documented
  in this README's Limitations, but not previously stated in the manuscript body,
  where the BLOCK method comparison is presented as five methods on identical masks).
- The manuscript's "following the missing-data taxonomy of Rubin [19]" framing
  overstated what the code implements: all three missingness patterns are
  mechanistically uniform-random selection within a differently-scoped candidate pool
  (entries / rows / columns), not differing in the MAR/MNAR sense. Reworded in
  Materials and Methods and Related Work. The BLOCK candidate pool's exclusion of the
  19 manufacturing target industries (previously undisclosed) is now stated explicitly.

### Reproducibility re-verification

All of the above were re-verified bit-exact reproducible (`numpy.array_equal`) across
independent fresh Python processes after being applied — both `experiment_final_eu.py`
(all three patterns) and `gsd_sensitivity.py` (full grid) independently. Reproducibility
survives all four fixes.

## Unreleased — seed-derivation fix, real-data reproducibility verification

### The bug

`code/experiment_final_eu.py` and `code/gsd_sensitivity.py` derived part of each
run's random seed from Python's built-in `hash(pattern)` (`pattern` being the string
`'MCAR'`, `'FLOW'`, or `'BLOCK'`):

```python
rng = np.random.default_rng(seed*1000 + int(rate*100) + hash(pattern)%97)
```

Python randomizes string hashing per process by design (`PYTHONHASHSEED`, PEP 456,
a security feature against hash-flooding attacks). This means `hash('MCAR')` returns
a *different* value every time the interpreter starts — so even with `seed` fixed
at 0/1/2, every fresh run of the script used a different effective seed for the
missingness mask. Three separate interpreter processes returned `hash('MCAR')%97`
= 67, 13, and 36 in a direct test — three different values for the identical
expression. Results were consequently not reproducible run to run, only self-consistent
within a single process.

### The fix

Replaced the hash-derived offset with a fixed, explicit mapping:

```python
PATTERN_OFFSET = {'MCAR': 11, 'FLOW': 43, 'BLOCK': 79}
rng = np.random.default_rng(seed*1000 + int(rate*100) + PATTERN_OFFSET[pattern])
```

Nothing about the method, the data, or the masking logic changed — only the seed
derivation. No other use of Python's `hash()` exists anywhere in the codebase (checked
by grep across both scripts).

### Verification performed in this pass

With the real `data_build/` matrices now available (they were not present in earlier
review passes, which could only cross-check the already-computed result CSVs against
the paper text and figures, not regenerate them):

1. Ran the corrected `experiment_final_eu.py` (all three patterns) and
   `gsd_sensitivity.py` in **four independent, freshly-started Python processes**
   (across two separate review sessions). All four produced **bit-exact identical**
   `mdape`, `coverage`, and `flips` columns — confirmed with `numpy.array_equal`
   (not `np.isclose`), i.e. exact floating-point equality, not approximate agreement.
2. Cross-checked the regenerated numbers against every quantitative claim in the
   manuscript (Abstract, Results 4.1-4.6, Table 1) and against all 8 result figures
   embedded in the manuscript. All matched.
3. Confirmed `torch.set_num_threads(1)` is set before any model training, and that
   `torch.manual_seed(seed)` is called per-condition for the VAE and GNN — both
   necessary for the bit-exact reproducibility observed in (1); CPU-only, so no
   GPU non-determinism applies.

This is a stronger reproducibility check than "the fix looks correct" — it is an
empirical demonstration, on the actual EU-27 data, that the same code with the same
seeds now produces the same numbers every time.

### Two prose corrections carried over from manuscript review

Independently verifying the manuscript's Results section against these CSVs (and
the manuscript's own embedded figures) surfaced two places where the *prose*
description of the corrected data was inaccurate, now fixed in the manuscript
(not a code or data issue — the underlying numbers were already correct):

- The GSD sensitivity sweep (`results/gsd_sensitivity_eu.csv`, `figures/fig_gsd_sensitivity.png`)
  peaks at GSD 2.5 (45.2% coverage, BLOCK @ 40%) and *declines* to 41.2% by GSD 3.0
  in that same condition — a non-monotonic, peak-then-decline pattern in three of the
  four pattern-rate combinations swept. An earlier manuscript draft described this as
  peaking "even at the widest setting tested," implying the opposite (monotonic rise).
- Under sensor failure at the 40% rate (`results/results_final_eu.csv`, pattern=FLOW,
  rate=0.4), the manuscript's "Model complexity" discussion originally attributed the
  instability spike solely to SoftImpute (116% median error, "an order of magnitude
  above every other method"). The data show the VAE reaches 289% and the GNN reaches
  120% at the same condition — both above SoftImpute, which is a close third rather
  than an outlier.

### Prior state

Before this pass, the repository as originally assembled bundled the *pre-fix* (buggy)
code together with results and a README that had already been generated from a
still-earlier, differently-seeded run — i.e., the code, results, and documentation
were three-way inconsistent. `results/`, `figures/`, and `README.md` are now all
regenerated from a single, verified, bit-exact-reproducible run of the fixed code.
