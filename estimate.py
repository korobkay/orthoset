"""
Estimated identified set and 95% confidence region for
theta = (gender gap, female x college) with a bracketed log wage.
Produces output/results.json and output/panel_D{1,2,3}.npz.
"""

from common import *
from scipy.optimize import linprog

Z, Y, fem, cg = load(); N = len(Y); fl = folds(N)
D = np.column_stack([fem, fem * cg])
print(
    f"N={N} p_X={Z.shape[1]}", 
    flush = True
)

ETA1 = crossfit_propensity(Z, D[:, 0], fl)
TRIM = 1e-4
res_trim = dict(
    eta1_min = float(ETA1.min()), 
    eta1_max = float(ETA1.max()), 
    n_trimmed = int(((ETA1 < TRIM) | (ETA1 > 1 - TRIM)).sum())
)
ETA1 = np.clip(
    ETA1, 
    TRIM, 
    1 - TRIM
)

# treatment residual
ETA = np.column_stack([ETA1, ETA1 * cg])
Dt = D - ETA
M = Dt.T @ Dt / N
corr = M[0, 1] / np.sqrt(M[0, 0] * M[1, 1])

# lower bracket residual
Yt = Y - crossfit_regression(Z, Y, fl)
th_gt = np.linalg.solve(M, Dt.T @ Yt / N)
r = Yt - Dt @ th_gt

# V matrix
V = np.linalg.solve(
    M, 
    (Dt * r[:, None]).T @ (Dt * r[:, None]) / N
) @ np.linalg.inv(M)
se = np.sqrt(np.diag(V) / N)

print(
    f"observed-wage estimate ({th_gt[0]:+.4f},{th_gt[1]:+.4f}) se ({se[0]:.4f},{se[1]:.4f})  corr(Dt)={corr:.3f}", 
    flush = True
)

QD = directions()
rng = np.random.default_rng(SEED_BLOCKS)
res = {"N": int(N), "p_X": int(Z.shape[1]), "theta_gt": th_gt.tolist(), "se_gt": se.tolist(),
       "corr_Dt": float(corr), "M": M.tolist(), "L": L_DIRECTIONS, "K": K_FOLDS, "tau": TAU,
       "b": BLOCK_SIZE, "B_N": int(N // BLOCK_SIZE), "panels": {}}

e1 = ETA[:, 0]; e2 = ETA[:, 1]
res.update(res_trim)

# some propensity diagnostics
res["eta1_deciles"] = [float(np.quantile(e1, 0.1)), float(np.quantile(e1, 0.9))]
res["share_eta1_near_half"] = float(np.mean(np.abs(e1 - 0.5) <= 0.05))

A0 = np.column_stack([-e1, -e2]); A1 = np.column_stack([1 - e1, -e2]); A2 = np.column_stack([1 - e1, 1 - e2])
PI = (1 - e1)[:, None] * (A0 @ QD.T > 0) + (e1 - e2)[:, None] * (A1 @ QD.T > 0) + e2[:, None] * (A2 @ QD.T > 0)

# some margin diagnostics
near_q = np.zeros(QD.shape[0])
for A, w in ((A0, np.ones(N)), (A1, 1 - cg), (A2, cg)):
    V = np.abs(A @ QD.T)
    near_q = np.maximum(
        near_q, 
        np.mean(
            ((V > 1e-12) & (V <= 0.05)) * w[:, None], 
            axis = 0
        )
    )
    
res["max_share_margin_005_over_q"] = float(near_q.max())
res["share_margin_005"] = float(np.median(near_q))

print(f"eta1 deciles {res['eta1_deciles']}, share near 1/2 {res['share_eta1_near_half']:.3f}, "
      f"margin share (median over q) {res['share_margin_005']:.3f}, max over q {res['max_share_margin_005_over_q']:.3f}", flush=True)

def cr_projection(B, ab, c, j, sgn, lo, hi):
    """Exact coordinate projection of the convex set {theta: N * sum_l (B_l theta - a_l)_+^2 <= c}."""
    from scipy.optimize import minimize_scalar
    k = 1 - j
    
    def inside(t):
        f = lambda u: N * np.sum(np.maximum((B[:, j] * t + B[:, k] * u) - ab, 0.0) ** 2)
        r = minimize_scalar(
            f, 
            bounds = (lo[k] - 10, hi[k] + 10), 
            method = "bounded", 
            options = {"xatol": 1e-10}
        )
        return r.fun <= c
    
    a_in = hi[j] if sgn > 0 else lo[j]          # inside: the estimated set is contained in the region
    a_out = a_in + sgn * 5.0 * (hi[j] - lo[j] + 1.0)
    assert inside(a_in) and not inside(a_out)
    
    for _ in range(60):
        mid = (a_in + a_out) / 2
        if inside(mid): a_in = mid
        else: a_out = mid
    return a_in

def polygon_area(B, ab):
    """Exact area of the polytope {theta in R^2: B theta <= a} via half-space intersection."""
    from scipy.spatial import HalfspaceIntersection
    
    # Chebyshev center as an interior point
    norms = np.linalg.norm(B, axis = 1)
    lp = linprog(
        np.array([0, 0, -1.0]), 
        A_ub = np.column_stack([B, norms]), 
        b_ub = ab, 
        bounds = [(None, None), (None, None), (0, None)], 
        method = "highs"
    )
    centre = lp.x[:2]
    hs = HalfspaceIntersection(
        np.column_stack([B, -ab]), 
        centre
    )
    V = hs.intersections; ang = np.arctan2(V[:, 1] - centre[1], V[:, 0] - centre[0]); V = V[np.argsort(ang)]
    x, y = V[:, 0], V[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def projections(B, ab):
    out = []
    for j in (0, 1):
        for sgn in (1, -1):
            c = np.zeros(2); c[j] = sgn
            lp = linprog(
                c, 
                A_ub = B, 
                b_ub = ab, 
                bounds = [(None, None)] * 2, 
                method = "highs"
            )
            out.append(sgn * lp.fun)
    return out  # [lo1, hi1, lo2, hi2]

for Delta in BRACKET_WIDTHS:
    YL, YU = bracket(Y, Delta)
    YtL = YL - crossfit_regression(Z, YL, fl)
    YtM = (YL + YU) / 2 - crossfit_regression(Z, (YL + YU) / 2, fl)
    th_mid = np.linalg.solve(M, Dt.T @ YtM / N)
    qd = Dt @ QD.T
    
    a = (qd * YtL[:, None] + Delta * (np.maximum(qd, 0) - qd * PI)).mean(0)   # corrected moment
    a_sym = (qd * YtL[:, None] + Delta * (np.maximum(qd, 0) - qd / 2)).mean(0)    # uncorrected moment
    
    B = (qd[:, :, None] * Dt[:, None, :]).mean(0)
    NQ = lambda th: N * np.sum(np.maximum(th @ B.T - a, 0.0) ** 2, axis = 1)

    lo1, hi1, lo2, hi2 = projections(B, a); pr_sym = projections(B, a_sym)
    m1, m2 = 0.35 * (hi1 - lo1), 0.35 * (hi2 - lo2)
    g1 = np.linspace(lo1 - m1, hi1 + m1, GRID); g2 = np.linspace(lo2 - m2, hi2 + m2, GRID)
    G1, G2 = np.meshgrid(g1, g2); TH = np.column_stack([G1.ravel(), G2.ravel()])
    nq = NQ(TH).reshape(G1.shape)
    cN = np.log(N); prelim = TH[nq.ravel() <= cN]

    # disjoint blocks balanced across folds
    BN = N // BLOCK_SIZE
    per = BLOCK_SIZE // K_FOLDS
    perm = np.column_stack([rng.permutation(te)[:BN * per].reshape(BN, per) for _, te in fl])
    assert perm.shape == (BN, BLOCK_SIZE)
    Tj = np.empty(BN)
    
    for j in range(BN):
        ix = perm[j]
        Bj = (qd[ix][:, :, None] * Dt[ix][:, None, :]).mean(0)
        aj = (qd[ix] * YtL[ix, None] + Delta * (np.maximum(qd[ix], 0) - qd[ix] * PI[ix])).mean(0)
        Tj[j] = np.max(BLOCK_SIZE * np.sum(np.maximum(prelim @ Bj.T - aj, 0.0) ** 2, axis = 1))
    c_tau = float(np.quantile(Tj, 1 - TAU)); c_wrong = float(np.quantile(Tj, TAU))

    cell = (g1[1] - g1[0]) * (g2[1] - g2[0])
    idset = nq <= 1e-9; cr = nq <= c_tau
    lo, hi = (lo1, lo2), (hi1, hi2)
    cr_proj = [cr_projection(B, a, c_tau, 0, -1, lo, hi), cr_projection(B, a, c_tau, 0, +1, lo, hi),
               cr_projection(B, a, c_tau, 1, -1, lo, hi), cr_projection(B, a, c_tau, 1, +1, lo, hi)]
    area_exact = polygon_area(B, a)
    
    panel = {"theta_mid": th_mid.tolist(),
             "proj_set": [lo1, hi1, lo2, hi2], "proj_set_sym": pr_sym, "proj_cr": cr_proj,
             "width_over_Delta": [(hi1 - lo1) / Delta, (hi2 - lo2) / Delta],
             "area_ratio": float(area_exact / ((hi1 - lo1) * (hi2 - lo2))),
             "area_ratio_grid": float(idset.sum() * cell / ((hi1 - lo1) * (hi2 - lo2))),
             "c_tau": c_tau, "c_tau_lower_quantile": c_wrong,
             "max_shift_sym_vs_corrected": float(max(abs(np.array(pr_sym) - np.array([lo1, hi1, lo2, hi2])))),
             "midpoint_bias_in_se": float(abs(th_mid[0] - th_gt[0]) / se[0]),
             "gt_in_set": bool(NQ(th_gt[None, :])[0] <= 1e-9), "gt_in_cr": bool(NQ(th_gt[None, :])[0] <= c_tau),
             "mid_in_set": bool(NQ(th_mid[None, :])[0] <= 1e-9)}
    res["panels"][str(Delta)] = panel
    
    np.savez_compressed(
        os.path.join(OUT, f"panel_D{Delta}.npz"), 
        g1 = g1, 
        g2 = g2, 
        nq = nq, 
        c_tau = c_tau,
        proj = np.array([lo1, hi1, lo2, hi2]), 
        theta_gt = th_gt, 
        theta_mid = th_mid, 
        Tj = Tj
    )
    print(f"Delta={Delta}: set th1 [{lo1:+.3f},{hi1:+.3f}] th2 [{lo2:+.3f},{hi2:+.3f}]  area/box={panel['area_ratio']:.3f}  "
          f"c_tau={c_tau:.2f}  midpoint bias {panel['midpoint_bias_in_se']:.1f} se", flush = True)

save_json(res, "results.json"); print("wrote output/results.json")
