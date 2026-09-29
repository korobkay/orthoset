"""Figure 1: output/cps_identified_sets.{pdf,png}."""

from common import *
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update(
    {
        "font.family": "serif", 
        "font.size": 9, 
        "axes.linewidth": .6, 
        "xtick.direction": "in", 
        "ytick.direction": "in", 
        "figure.dpi": 160
    }
)
R = json.load(
    open(
        os.path.join(OUT, "results.json")
    )
)
gt = np.array(R["theta_gt"]); se = np.array(R["se_gt"])

C_SET, C_CR, C_GT, C_MID, C_BOX = "#1b6ca8", "#9ecae1", "#111111", "#c0392b", "#6b6b6b"
FIG_WIDTHS = BRACKET_WIDTHS

x_lows = []
x_highs = []
y_lows = []
y_highs = []

for D in FIG_WIDTHS:
    z = np.load(os.path.join(OUT, f"panel_D{D}.npz"))
    lo1, hi1, lo2, hi2 = z["proj"]

    x_lows.append(lo1)
    x_highs.append(hi1)
    y_lows.append(lo2)
    y_highs.append(hi2)

x_min = min(x_lows)
x_max = max(x_highs)
x_margin = 0.06 * (x_max - x_min)
X_LIM = (x_min - x_margin, x_max + x_margin)

y_min = min(y_lows)
y_max = max(y_highs)
y_margin = 0.06 * (y_max - y_min)
Y_LIM = (y_min - y_margin, y_max + y_margin)

fig, axes = plt.subplots(
    1,
    len(FIG_WIDTHS),
    figsize = (5.4 if len(FIG_WIDTHS) == 1 else 9.2,
               3.6 if len(FIG_WIDTHS) == 1 else 3.25
            ),
    squeeze = False,
    sharex = True,
    sharey = True
)

for ax, D in zip(axes[0], FIG_WIDTHS):
    z = np.load(
        os.path.join(OUT, f"panel_D{D}.npz")
    )
    g1, g2, nq = z["g1"], z["g2"], z["nq"]
    ct = float(z["c_tau"])
    lo1, hi1, lo2, hi2 = z["proj"]
    mid = z["theta_mid"]
    
    ax.contourf(
        g1, 
        g2, 
        nq, 
        levels = [0, ct], 
        colors = [C_CR], 
        alpha = .55
    )
    ax.contour(
        g1, 
        g2, 
        nq, 
        levels = [ct], 
        colors = [C_SET], 
        linewidths = .7, 
        linestyles = "--"
    )
    ax.contourf(
        g1, 
        g2, 
        nq, 
        levels = [0, 1e-9], 
        colors = [C_SET], 
        alpha = .55
    )
    ax.add_patch(
        Rectangle(
            (lo1, lo2), 
            hi1 - lo1, 
            hi2 - lo2, 
            fill = False, 
            ec = C_BOX, 
            lw = .9, 
            ls = (0, (2, 2))
        )
    )
    ax.errorbar(
        gt[0], 
        gt[1], 
        xerr = 1.96 * se[0], 
        yerr = 1.96 * se[1], 
        fmt = "o", 
        ms = 3.4, 
        color = C_GT, 
        lw = .9, 
        capsize = 1.6, 
        zorder = 6
    )
    ax.plot(
        mid[0], 
        mid[1], 
        marker = "x", 
        ms = 5.5, 
        mew = 1.4, 
        color = C_MID, 
        zorder = 6, 
        ls = "none"
    )
    ax.axhline(
        0, 
        color = "#cccccc", 
        lw = .5, 
        zorder = 0) 
    ax.axvline(
        0, 
        color = "#cccccc", 
        lw = .5, 
        zorder = 0
    )
    r = R["panels"][str(D)]
    ax.set_title(
        rf"$\Delta={D}$", 
        fontsize = 9.5, 
        pad = 4
    )
    ax.text(
        .5, 
        .07, 
        f"area of $\\widehat{{\\Theta}}_I$ = {r['area_ratio']:.2f}" + r"$\times$ box",
        transform = ax.transAxes, 
        ha = "center", 
        va = "bottom", 
        fontsize = 7.2, 
        color = "#444444",
        bbox = dict(
            fc = "white", 
            ec = "none", 
            alpha = .85, 
            pad = 1.5
        )
    )
    ax.set_xlabel(r"$\theta_1$: gender gap")
    
    if D == FIG_WIDTHS[0]: ax.set_ylabel(r"$\theta_2$: female $\times$ college")
    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)
    for s in ax.spines.values(): s.set_color("#666666")

h = [plt.Line2D([], [], color = C_SET, lw = 6, alpha = .55), plt.Line2D([], [], color = C_CR, lw = 6, alpha = .55),
     plt.Line2D([], [], color = C_BOX, lw = .9, ls = (0, (2, 2))), plt.Line2D([], [], color = C_GT, marker = "o", ms=3.4, lw=.9),
     plt.Line2D([], [], color = C_MID, marker = "x", ms = 5.5, mew = 1.4, ls = "none")]
l = [r"$\widehat{\Theta}_I$", r"95% confidence region", "box of coordinate projections", "observed-wage estimate (95% CI)", "midpoint regression"]
fig.legend(
    h, 
    l, 
    loc = "lower center", 
    ncol = 3 if len(FIG_WIDTHS) == 1 else 5, 
    frameon = False, 
    fontsize = 7.6, 
    bbox_to_anchor = (.5, -.06)
)
fig.tight_layout(
    rect = [0, .10 if len(FIG_WIDTHS) == 1 else .055, 1, 1]
)
fig.savefig(
    os.path.join(
        OUT, 
        "cps_identified_sets.pdf"
    ), 
    bbox_inches = "tight"
)
fig.savefig(
    os.path.join(
        OUT, 
        "cps_identified_sets.png"
    ), 
    bbox_inches = "tight", 
    dpi = 200
)
print("wrote output/cps_identified_sets.{pdf,png}")
