from pathlib import Path
import csv
import json
import zipfile

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
RESULTS = PROJECT / "results"
RESULTS.mkdir(exist_ok=True)

# Source data transcribed from manuscript Tables V and VI.
table_v = list(csv.DictReader((DATA / "table_v_floor_sweep.csv").open()))
table_vi = list(csv.DictReader((DATA / "table_vi_mismatch_regret.csv").open()))

g_db = np.array([float(r["gamma_f_db"]) for r in table_v])
floor_fraction = np.array([float(r["nf_over_n"]) for r in table_v])
teq = np.array([float(r["teq_mbps"]) for r in table_v])
lhs_table = np.array([float(r["eq24_lhs"]) for r in table_v])
rhs_table = np.array([float(r["eq24_rhs"]) for r in table_v])

delta = np.array([float(r["delta"]) for r in table_vi])
regret_random = np.array([float(r["random_draws"]) for r in table_vi])
regret_corner = np.array([float(r["adversarial_corner"]) for r in table_vi])
regret_bound = np.array([float(r["bound"]) for r in table_vi])

# ------------------------------------------------------------
# Panel (a): Table V floor/refinement split.
# Panel (b1): Eq. (24) balance. The RHS is interpolated only
#             between the discrete dual values reported in Table V.
# ------------------------------------------------------------
# Exact analytical left side, using gamma in linear units.
def eq24_lhs_db(gamma_db):
    gamma = 10.0 ** (np.asarray(gamma_db) / 10.0)
    return (1.0 + gamma) * np.log1p(gamma) - gamma

rhs_interp = PchipInterpolator(g_db, rhs_table, extrapolate=False)
lhs_interp_exact = eq24_lhs_db

root_db = brentq(
    lambda x: float(rhs_interp(x) - lhs_interp_exact(x)),
    9.0, 9.5,
)

# The Table V exhaustive grid optimum.
idx_opt = int(np.argmax(teq))
exhaustive_opt_db = float(g_db[idx_opt])
teq_opt = float(teq[idx_opt])

# The manuscript reports the 9.1 dB stationary approximation to be
# 0.6% below the throughput optimum and 0.7% above minimum distortion.
# Keep those reported relative differences as annotations; do not
# invent a distortion trajectory at the stationary point.
throughput_gap_pct = 0.6
distortion_gap_pct = 0.7

# Dense grid for plotting, using exact LHS and shape-preserving
# interpolation of the *tabulated* RHS.
g_dense = np.linspace(g_db.min(), g_db.max(), 500)
lhs_dense = eq24_lhs_db(g_dense)
rhs_dense = rhs_interp(g_dense)

# ------------------------------------------------------------
# Panel (b2): local-linearized controller trace from Proposition 10.
# The manuscript reports s ~= -0.52 for the smaller synthetic model
# and fixed point ~= 9.41 dB. This is explicitly an illustration of
# the reported local linearization, not a full nonlinear recomputation.
# ------------------------------------------------------------
gamma_star = 9.41
slope = -0.52
iterations = np.arange(0, 8)
starts = [6.0, 14.0]
taus = [1.0, 0.66]
controller = {}
for tau in taus:
    factor = 1.0 - tau + tau * slope
    for start in starts:
        values = gamma_star + (start - gamma_star) * factor ** iterations
        controller[f"tau={tau:g}, start={start:g} dB"] = values.tolist()

# ------------------------------------------------------------
# Source-table checks and scientific assertions.
# ------------------------------------------------------------
if exhaustive_opt_db != 10.5 or abs(teq_opt - 9.781) > 1e-12:
    raise RuntimeError("Table V exhaustive optimum failed reproduction.")
if not (9.0 <= root_db <= 9.5):
    raise RuntimeError("Eq. (24) stationary crossing is outside reported interval.")
if not np.all(np.diff(floor_fraction) < 0):
    raise RuntimeError("Table V floor-use fraction should decrease over sweep.")
if not (np.all(regret_random >= 1.0) and np.all(regret_corner >= 1.0)):
    raise RuntimeError("Mismatch regret ratios must be >= 1.")
if not np.all(regret_random <= regret_bound) or not np.all(regret_corner <= regret_bound):
    raise RuntimeError("Observed regrets exceed Proposition 12 bound.")
if not np.allclose(regret_bound, (1.0 + delta) ** 2, atol=1e-12):
    raise RuntimeError("Proposition 12 bound does not equal (1+delta)^2.")

# Export panel datasets in a reproducible dense-grid form.
with (DATA / "figure5_eq24_dense.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["gamma_f_db", "lhs_exact_eq24", "rhs_pchip_from_table_v"])
    w.writerows(zip(g_dense, lhs_dense, rhs_dense))

with (DATA / "figure5_controller_local_linearization.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["iteration", *controller.keys()])
    for j, it in enumerate(iterations):
        w.writerow([int(it), *[controller[k][j] for k in controller]])

validation = {
    "source_data": {
        "table_v_rows": len(table_v),
        "table_vi_rows": len(table_vi),
        "table_v_exhaustive_grid_optimum_db": exhaustive_opt_db,
        "table_v_max_teq_mbps": teq_opt,
        "eq24_crossing_db_pchip_rhs": float(root_db),
        "eq24_crossing_lhs_rhs_value": float(lhs_interp_exact(root_db)),
        "stationary_throughput_gap_pct_reported": throughput_gap_pct,
        "stationary_distortion_gap_pct_reported": distortion_gap_pct,
    },
    "controller_local_linearization": {
        "fixed_point_db": gamma_star,
        "reported_slope_approx": slope,
        "interpretation": "Local affine approximation using the manuscript-reported slope; not a recomputation of the full nonlinear Gamma map.",
        "trajectories_db": controller,
    },
    "mismatch": {
        "all_random_and_corner_regrets_below_bound": True,
        "bound_matches_1_plus_delta_squared": True,
        "random_instances_per_delta": 300,
        "components_per_instance": 60,
    },
    "scene_cut_inset": "Omitted: manuscript gives aggregate factors but not a complete reproducible time-series dataset; plan says not to draw a time series unless explicitly simulated.",
}
(RESULTS / "figure5_validation.json").write_text(json.dumps(validation, indent=2))

# ------------------------------------------------------------
# Publication style and 3-panel plot.
# ------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.5,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "legend.fontsize": 6.3,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.75,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

blue = "#245B78"
orange = "#D18B35"
teal = "#32877D"
purple = "#8064A2"
grey = "#777777"

fig = plt.figure(figsize=(7.16, 5.0))
gs = fig.add_gridspec(
    2, 2,
    height_ratios=[1.0, 1.22],
    width_ratios=[1.08, 0.92],
    hspace=0.43,
    wspace=0.32,
)

# Panel (a): full width.
ax_a = fig.add_subplot(gs[0, :])
ax_a_r = ax_a.twinx()

line_teq, = ax_a.plot(
    g_db, teq, color=blue, marker="o", markersize=3.4,
    linewidth=1.65, label=r"Equivalent throughput $T_{eq}$",
)
line_floor, = ax_a_r.plot(
    g_db, 100.0 * floor_fraction, color=teal, marker="s",
    markersize=3.0, linestyle="--", linewidth=1.4,
    label=r"Floor-use fraction $n_F/n$",
)

ax_a.axvline(root_db, color=grey, linestyle="-.", linewidth=0.95)
ax_a.axvline(exhaustive_opt_db, color=orange, linestyle=":", linewidth=1.25)
ax_a.scatter([exhaustive_opt_db], [teq_opt], s=48, marker="*", color=orange,
             edgecolors="black", linewidths=0.35, zorder=5)

ax_a.annotate(
    f"Stationary condition ≈ {root_db:.1f} dB\n"
    f"Reported gaps: {throughput_gap_pct:.1f}% throughput; {distortion_gap_pct:.1f}% distortion",
    xy=(root_db, float(PchipInterpolator(g_db, teq)(root_db))),
    xytext=(6.35, 9.42),
    fontsize=6.5,
    arrowprops={"arrowstyle": "->", "color": grey, "lw": 0.7},
    color=grey,
)
ax_a.annotate(
    "Exhaustive grid optimum\n10.5 dB",
    xy=(exhaustive_opt_db, teq_opt),
    xytext=(11.15, 9.65),
    fontsize=6.5,
    color=orange,
    arrowprops={"arrowstyle": "->", "color": orange, "lw": 0.7},
)

ax_a.set_xlim(5.5, 14.5)
ax_a.set_ylim(9.1, 9.9)
ax_a_r.set_ylim(14.0, 36.0)
ax_a.set_xticks(g_db)
ax_a.set_xlabel(r"Floor SNR, $\gamma_F$ (dB)")
ax_a.set_ylabel(r"Equivalent throughput, $T_{eq}$ (Mbps)", color=blue)
ax_a_r.set_ylabel(r"Floor-use fraction, $100 n_F/n$ (%)", color=teal)
ax_a.tick_params(axis="y", labelcolor=blue)
ax_a_r.tick_params(axis="y", labelcolor=teal)
ax_a.grid(axis="both", color="#D9D9D9", linewidth=0.4, alpha=0.8)
ax_a.legend([line_teq, line_floor], [line_teq.get_label(), line_floor.get_label()],
            loc="lower right", frameon=False, ncol=2, fontsize=6.5)
ax_a.text(-0.055, 1.06, "(a)", transform=ax_a.transAxes,
          fontweight="bold", fontsize=9)

# Panel (b1): Eq. (24) balance.
left_group = gs[1, 0].subgridspec(2, 1, height_ratios=[1.0, 0.86], hspace=0.55)
ax_b1 = fig.add_subplot(left_group[0, 0])
ax_b1.plot(g_dense, lhs_dense, color=blue, linewidth=1.55,
           label=r"LHS: $(1+\gamma)\ln(1+\gamma)-\gamma$")
ax_b1.plot(g_dense, rhs_dense, color=orange, linewidth=1.55, linestyle="--",
           label=r"RHS: $2\zeta/(\xi\nu_c)$ (Table V interpolation)")
ax_b1.scatter(g_db, rhs_table, color=orange, marker="s", s=13, zorder=4)
ax_b1.scatter(g_db, lhs_table, facecolors="none", edgecolors=blue,
              marker="o", s=16, zorder=4)
ax_b1.axvline(root_db, color=grey, linestyle=":", linewidth=0.9)
ax_b1.scatter([root_db], [float(lhs_interp_exact(root_db))], marker="*",
              color=purple, edgecolors="black", linewidths=0.3, s=55, zorder=5)
ax_b1.annotate(f"Crossing ≈ {root_db:.2f} dB", xy=(root_db, float(lhs_interp_exact(root_db))),
               xytext=(9.7, 16.0), fontsize=6,
               arrowprops={"arrowstyle": "->", "lw": 0.65, "color": purple},
               color=purple)
ax_b1.set_xlim(6, 14)
ax_b1.set_ylim(0, 65)
ax_b1.set_ylabel("Marginal-cost balance")
ax_b1.set_xlabel(r"Floor SNR, $\gamma_F$ (dB)")
ax_b1.grid(axis="both", color="#D9D9D9", linewidth=0.4, alpha=0.8)
ax_b1.legend(loc="upper left", frameon=False, fontsize=5.7)
ax_b1.text(0.00, 1.13, "(b)", transform=ax_b1.transAxes,
           fontweight="bold", fontsize=9)

# Panel (b2): controller trajectory from reported local linearization.
ax_b2 = fig.add_subplot(left_group[1, 0])
styles = {
    (1.0, 6.0): (blue, "-", "o"),
    (1.0, 14.0): (orange, "-", "s"),
    (0.66, 6.0): (teal, "--", "o"),
    (0.66, 14.0): (purple, "--", "s"),
}
for (tau, start), vals in [
    ((tau, start), np.array(controller[f"tau={tau:g}, start={start:g} dB"]))
    for tau in taus for start in starts
]:
    color, ls, marker = styles[(tau, start)]
    ax_b2.plot(iterations, vals, color=color, linestyle=ls, marker=marker,
               markersize=2.7, linewidth=1.1,
               label=rf"$\tau={tau:g}$, start {start:g} dB")
ax_b2.axhline(gamma_star, color=grey, linestyle=":", linewidth=0.8,
              label=r"Fixed point $9.41$ dB")
ax_b2.set_xlim(0, 7)
ax_b2.set_ylim(5.5, 14.5)
ax_b2.set_xticks(np.arange(0, 8))
ax_b2.set_xlabel("Controller iteration")
ax_b2.set_ylabel(r"$\gamma_F$ (dB)")
ax_b2.grid(axis="both", color="#D9D9D9", linewidth=0.4, alpha=0.8)
ax_b2.legend(loc="upper right", frameon=False, ncol=2, fontsize=5.2)
ax_b2.text(0.02, 0.04, r"Local affine map, reported slope $s\approx-0.52$",
           transform=ax_b2.transAxes, fontsize=5.4, color=grey)

# Panel (c): mismatch regret. Main plot plus inset for values close to 1.
ax_c = fig.add_subplot(gs[1, 1])
ax_c.plot(delta, regret_random, color=blue, marker="o", linewidth=1.5,
          markersize=3.5, label="300 random instances")
ax_c.plot(delta, regret_corner, color=teal, marker="s", linewidth=1.5,
          markersize=3.3, label="Adversarial corners")
ax_c.plot(delta, regret_bound, color=orange, marker="^", linewidth=1.5,
          markersize=3.3, linestyle="--", label=r"Bound $(1+\delta)^2$")
ax_c.set_xlim(0.05, 1.05)
ax_c.set_ylim(0.95, 4.25)
ax_c.set_xticks(delta)
ax_c.set_xlabel(r"Mismatch magnitude, $\delta$")
ax_c.set_ylabel(r"Distortion regret ratio")
ax_c.grid(axis="both", color="#D9D9D9", linewidth=0.4, alpha=0.8)
ax_c.legend(loc="upper left", frameon=False, fontsize=5.8)
ax_c.text(-0.12, 1.08, "(c)", transform=ax_c.transAxes,
          fontweight="bold", fontsize=9)
ax_c.text(0.98, 0.05, r"$m=60$; 300 random draws per $\delta$",
          transform=ax_c.transAxes, ha="right", fontsize=5.4, color=grey)

# Zoom inset for the empirical regret values near 1.
ax_in = ax_c.inset_axes([0.43, 0.42, 0.52, 0.40])
ax_in.plot(delta, regret_random, color=blue, marker="o", linewidth=1.0, markersize=2.4)
ax_in.plot(delta, regret_corner, color=teal, marker="s", linewidth=1.0, markersize=2.3)
ax_in.set_xlim(0.05, 1.05)
ax_in.set_ylim(0.998, 1.07)
ax_in.set_xticks([0.1, 0.5, 1.0])
ax_in.set_yticks([1.00, 1.03, 1.06])
ax_in.tick_params(labelsize=5, length=2)
ax_in.grid(color="#E0E0E0", linewidth=0.35)
ax_in.set_title("Empirical-regret detail", fontsize=5.4, pad=2)

# Clean top spines and export.
for ax in [ax_a, ax_a_r, ax_b1, ax_b2, ax_c, ax_in]:
    ax.spines["top"].set_visible(False)
    ax.tick_params(direction="out", length=2.5, width=0.65)

fig.subplots_adjust(left=0.09, right=0.92, bottom=0.09, top=0.96)

pdf = RESULTS / "Figure_5_Floor_Power_Controller_Mismatch.pdf"
svg = RESULTS / "Figure_5_Floor_Power_Controller_Mismatch.svg"
png = RESULTS / "Figure_5_Floor_Power_Controller_Mismatch.png"
fig.savefig(pdf, bbox_inches="tight")
fig.savefig(svg, bbox_inches="tight")
fig.savefig(png, dpi=600, bbox_inches="tight")
plt.close(fig)

print("FIGURE 5 — SOURCE-TABLE AND EQUATION CHECKS: PASSED")
print(f"Table V exhaustive grid optimum: {exhaustive_opt_db:.2f} dB, Teq={teq_opt:.3f} Mbps")
print(f"Eq. (24) crossing from PCHIP of tabulated RHS: {root_db:.4f} dB")
print("Source-reported stationary gaps: 0.6% throughput; 0.7% distortion")
print("Controller: local linearization s=-0.52, fixed point=9.41 dB; not a full nonlinear Gamma recomputation")
print("Table VI: all random/corner regrets are below (1+delta)^2")
print("Scene-cut time-series inset: omitted (source does not provide complete reproducible time-series data)")
for path in (pdf, svg, png):
    print(f"Created: {path} ({path.stat().st_size:,} bytes)")
print(f"Validation JSON: {RESULTS / 'figure5_validation.json'}")
