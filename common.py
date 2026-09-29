"""
Common functions for the replication package (Section 5).

Data and design: CPS March Supplement 2015, N = 32,523 white non-Hispanic 
full-time full-year workers aged 25-64. 
Outcome: log hourly wage. 
Controls: exp1-exp4 and the factors educ, occ2, ind2, ms, reg, with all exp x factor interactions.
"""

import warnings; warnings.filterwarnings("ignore")
import os, json, hashlib
import numpy as np, pandas as pd
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.model_selection import KFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "cps2015.csv")
OUT  = os.path.join(ROOT, "output")
DATA_MD5 = "0d74fd60e2d8679b1c8e2d20d7780897"

# ---- fixed constants (current seeds reproduce the paper) ----
SEED_FOLDS = 1          # KFold shuffle seed (first stage)
SEED_LASSO = 0          # LassoCV internal CV seed
SEED_BLOCKS = 7         # random permutation defining the disjoint blocks
K_FOLDS = 2             # cross-fitting folds
L_DIRECTIONS = 180      # supporting directions on the unit circle
TAU = 0.05              # nominal non-coverage
BLOCK_SIZE = 160        # b; B_N = floor(N/b) = 203
LOGIT_C = 0.1           # l1-logistic inverse penalty (liblinear)
GRID = 401              # grid points per axis for the contour sets
BRACKET_WIDTHS = (1, 2, 3)

def check_data():
    h = hashlib.md5(open(DATA, "rb").read()).hexdigest()
    if h != DATA_MD5:
        raise SystemExit(f"data/cps2015.csv md5 {h} != expected {DATA_MD5}; wrong or modified file")

def load():
    """Returns Z (standardised design, N x 259), Y (lnw), female, cg."""
    check_data()
    d = pd.read_csv(
        DATA, 
        index_col = 0
    )
    fac = ["educ", "occ2", "ind2", "ms", "reg"]
    Dm = pd.get_dummies(
        d[fac].astype(str), 
        drop_first = True
    ).astype(float)
    
    exp = d[["exp1", "exp2", "exp3", "exp4"]].astype(float)
    inter = pd.concat(
        {f"{e}:{c}": exp[e] * Dm[c] for e in exp.columns for c in Dm.columns}, 
        axis = 1
    )
    Z = pd.concat([exp, Dm, inter], axis = 1).to_numpy()
    Z = Z - Z.mean(0)
    Z = Z / np.where(Z.std(0) > 0, Z.std(0), 1.0)
    return Z, d["lnw"].to_numpy(), d["female"].to_numpy().astype(float), d["cg"].to_numpy().astype(float)

def folds(N):
    return list(
        KFold(
            n_splits = K_FOLDS, 
            shuffle = True, 
            random_state = SEED_FOLDS
        ).split(np.arange(N))
    )

def crossfit_regression(Z, target, fl):
    """Cross-fitted E[target|X] by lasso with 3-fold CV penalty selection."""
    out = np.empty(len(target))
    for tr, te in fl:
        m = LassoCV(
            cv = 3, 
            max_iter = 3000, 
            n_jobs = -1, 
            random_state = SEED_LASSO
        ).fit(Z[tr], target[tr])
        out[te] = m.predict(Z[te])
    return out

def crossfit_propensity(Z, target, fl):
    """Cross-fitted P(target=1|X) by l1-penalised logistic regression."""
    out = np.empty(len(target))
    for tr, te in fl:
        m = LogisticRegression(
            penalty = "l1", 
            solver = "liblinear", 
            C = LOGIT_C,
            max_iter = 2000
        ).fit(Z[tr], target[tr])
        out[te] = m.predict_proba(Z[te])[:, 1]
    return out

def bracket(y, width):
    """Full-support bracketing with edges 1, 1 + width, ... covering [min y, max y]."""
    edges = np.arange(
        np.floor(y.min()), 
        np.ceil(y.max()) + width, 
        width
    )   # = 1, 1 + w, ... on this sample
    idx = np.clip(
        np.searchsorted(edges, y, side="right") - 1, 
        0, 
        len(edges) - 2
    )
    return edges[idx], edges[idx + 1]

def directions(L = L_DIRECTIONS):
    a = np.linspace(
        0, 
        2 * np.pi, 
        L, 
        endpoint = False
    )
    return np.column_stack([np.cos(a), np.sin(a)])

def save_json(obj, name):
    os.makedirs(OUT, exist_ok = True)
    json.dump(
        obj, 
        open(os.path.join(OUT, name), "w"), 
        indent = 2
    )
