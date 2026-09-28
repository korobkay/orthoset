# Replication package — Sections 3--5 (bracketed-wage example, empirical results, verification)

Semenova and Korobka, *Orthogonal Moment Inequalities*. This package reproduces every number in Section 4 (and the propensity diagnostics of Section 5.3): Figure 1,
Table 1, and each figure quoted in the text.

## What it reproduces

| Output | Produced by | Used in the paper as |
|---|---|---|
| `output/results.json` | `code/01_estimate.py` | source of all numbers |
| `output/panel_D{1,2,3}.npz` | `code/01_estimate.py` | grids for Figure 1 |
| `output/cps_identified_sets.pdf` | `code/02_figure.py` | Figure 1 (`\includegraphics`; the $\Delta=2$ panel; set `FIG_WIDTHS = BRACKET_WIDTHS` for all three) |
| `output/table_cps.tex` | `code/03_table.py` | Table 1, the whole `tabular` (`\input` in `main.tex`) |
| `output/numbers_in_text.txt` | `code/03_table.py` | every number quoted in the prose of Sections 4 and 5 |
| `output/validate_scalar.json` | `code/00_validate.py` | not in the paper; sanity check, see below |

## How to run

    pip install -r requirements.txt
    bash run_all.sh

About 3 minutes on a 2024 laptop (the first stage is 2 folds × 8 lasso fits on
32,523 × 259; `LassoCV` uses all cores). Python 3.11; package versions pinned
in `requirements.txt` are the ones the paper's numbers were produced with.

The pipeline is fully deterministic given the three seeds in `code/common.py`
(`SEED_FOLDS`, `SEED_LASSO`, `SEED_BLOCKS`). Changing any constant there changes
the results; none should be changed to reproduce the paper.

## Data

`data/cps2015.csv` (4.6 MB, md5 `0d74fd60e2d8679b1c8e2d20d7780897`) is the
analysis file from the replication package of Semenova (2023, *Journal of
Econometrics* 235(2), 1725–1746), itself built from the March Supplement of the
2015 Current Population Survey: white non-Hispanic individuals aged 25–64 working
more than 35 hours per week for at least 50 weeks of the year; N = 32,523. The
scripts check the md5 before running. Variables used: `lnw`, `female`, `cg`,
`exp1`–`exp4`, `educ`, `occ2`, `ind2`, `ms`, `reg`.

## Design, and where it departs from Semenova (2023)

The design matrix is `-1 + (exp1+exp2+exp3+exp4)*(educ+occ2+ind2+ms+reg) + (educ+occ2+ind2+ms+reg)`
in R syntax, standardised. It has 259 columns here against 260 in that paper; the
difference is one collinear factor level dropped by pandas' `drop_first`.

Three deliberate departures, all stated in the paper (Section 4):

1. **Penalty selection.** `LassoCV` with 3 folds for regressions and an
   ℓ1-logistic with `C = 0.1` (liblinear) for propensities, instead of the
   plug-in penalty of `hdm::rlasso` / `rlassologit`.
2. **Bracketing.** `common.bracket()` covers the full support of `lnw` with edges
   1, 1+Δ, 1+2Δ, …; the `bracket()` in the 2023 R code leaves the upper tail
   unbracketed at some widths.
3. **Moment.** The paper's main results use the orthogonality-corrected moment
   (paper eq. emp-moment) with π_q(X) in closed form (eq. pi-closed). The
   uncorrected "SYM" moment of the 2023 paper is computed alongside and reported
   in the row "uncorrected moment" of Table 1.

## Sanity check against the 2023 paper

`code/00_validate.py` runs the *scalar* Semenova (2023) estimator (treatment
`female` alone, SYM moment) on this sample. Expected output, and what the 2023
paper reports:

    observed-wage gender gap = -0.2000        (2023: 18–20%)
    Delta=2: midpoint -0.2649                 (2023: 25–28%)
    width/Delta = 2.000 at every Delta        (exact identity for a binary treatment;
                                               paper, Section 3, "coordinate widths")

If these do not reproduce, the design matrix or the data file is wrong; stop there.

## How the paper consumes this

`main.tex` does not transcribe numbers. It `\input`s `table_cps.tex` and
`\includegraphics` `cps_identified_sets.pdf`, both copied from `output/` to the
paper's directory after a run. The numbers quoted in the prose of Sections 4 and 5 are
listed in `output/numbers_in_text.txt` so they can be checked against the text
by eye; that is the one manual step.

## Files

    run_all.sh             entry point
    requirements.txt       pinned versions
    code/common.py         data loading, design matrix, first stage, constants and seeds
    code/00_validate.py    scalar reproduction of Semenova (2023)     [optional]
    code/01_estimate.py    identified set, confidence region, subsampling
    code/02_figure.py      Figure 1
    code/03_table.py       Table 1 body and numbers quoted in the text
    data/cps2015.csv       analysis file (see Data)
    output/                everything the scripts write; safe to delete and regenerate

## Contact

Vira Semenova, semenovavira@gmail.com
