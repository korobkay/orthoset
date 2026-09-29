# Replication package — Section 5 (bracketed-wage example)

Korobka and Semenova (2026), *Debiased Inference for Bounding Wage Inequality with Many Controls*. This package reproduces Table 1 and Figure 1 in Section 5.

## Outputs

| Output | Produced by | Used in the paper as |
|---|---|---|
| `output/results.json` | `code/estimate.py` | estimates the sets and constructs confidence regions |
| `output/panel_D{1,2,3}.npz` | `code/estimate.py` | grids for Figure 1 |
| `output/cps_identified_sets.pdf` | `code/figure.py` | Figure 1 |

## How to run
In the Terminal, run as follows:

    python3 -m venv .venv
    source .venv/bin/activate

    python -m pip install --upgrade pip
    python -m pip install numpy pandas scipy scikit-learn matplotlib

    mkdir -p ../output

    python estimate.py
    python figure.py

The code is fully automatic given the three seeds in `code/common.py`
(`SEED_FOLDS`, `SEED_LASSO`, `SEED_BLOCKS`). Values of the seeds reproduce the paper's results.

## Data

`data/cps2015.csv` is the analysis file from the replication package of Semenova (2023, *Journal of
Econometrics* 235(2), 1725–1746), itself built from the March Supplement of the
2015 Current Population Survey: white non-Hispanic individuals aged 25–64 working
more than 35 hours per week for at least 50 weeks of the year; N = 32,523. Variables used: `lnw`, `female`, `cg`,
`exp1`–`exp4`, `educ`, `occ2`, `ind2`, `ms`, `reg`.

## Design

The design matrix is `-1 + (exp1+exp2+exp3+exp4)*(educ+occ2+ind2+ms+reg) + (educ+occ2+ind2+ms+reg)` standardised, with 259 columns.

Design details:

1. **Penalty selection.** `LassoCV` with 3 folds for regressions and an
   ℓ1-logistic with `C = 0.1` (liblinear) for propensities.
2. **Bracketing.** `common.bracket()` covers the full support of `lnw` with edges
   1, 1+Δ, 1+2Δ, ….
3. **Moment.** The paper's main results use the orthogonality-corrected moment with π_q(X) in closed form. The
   uncorrected "SYM" moment is computed alongside and reported in the row "uncorrected moment" of Table 1.

## Files

    code/common.py         data loading, design matrix, first stage, constants and seeds
    code/estimate.py       identified set, confidence region, subsampling
    code/figure.py         Figure 1
    data/cps2015.csv       analysis file (see Data)
    output/                scripts' output

## Contact

Yaroslav Korobka is responsible for the repository maintenance, yaroslav.korobka@cerge-ei.cz.
