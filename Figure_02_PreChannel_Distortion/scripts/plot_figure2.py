import json
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("results")
sweep = json.loads((ROOT / "fig2_bandwidth_sweep.json").read_text())
gop_records = json.loads((ROOT / "fig2_gop_metrics.json").read_text())

# ---------- Figure 2(a): bandwidth sweep ----------
rho = np.array([r["rho"] for r in sweep])
proposed = np.array([
    np.nan if r["proposed"] is None else r["proposed"]["beta_mean"]
    for r in sweep
])
prop7 = np.array([
    np.nan if r["proposed"] is None else r["proposed"]["prop7_mean"]
    for r in sweep
])
linear = np.array([
    np.nan if r["linear_analog"] is None
    else r["linear_analog"]["beta_mean"]
    for r in sweep
])

feasible = np.isfinite(proposed)

if feasible.sum() < 3:
    raise SystemExit("Insufficient feasible proposed-scheme points.")

# Check expected monotonic decrease on the tested feasible grid.
for name, values in [
    ("Proposed", proposed),
    ("Proposition 7", prop7),
    ("Linear analog", linear),
]:
    v = values[np.isfinite(values)]
    if np.any(np.diff(v) >= 1e-10):
        raise SystemExit(f"Monotonicity check failed for {name}.")

# ---------- Figure 2(b): per-GoP decomposition ----------
snrs = [6, 8, 10, 12]
beta_means, bound_means, remainder_means = [], [], []

for snr in snrs:
    rows = [
        r for r in gop_records
        if r["scheme"] == "proposed" and r["snr_db"] == snr
    ]

    if len(rows) != 16:
        raise SystemExit(f"Expected 16 proposed GoPs at {snr} dB.")

    beta = np.array([r["beta_fractional"] for r in rows])
    bound = np.array([r["prop7_bound_M"] for r in rows])
    remainder = beta - bound

    if np.min(remainder) < -1e-10:
        raise SystemExit(f"Proposition 7 bound violation at {snr} dB.")

    beta_means.append(float(beta.mean()))
    bound_means.append(float(bound.mean()))
    remainder_means.append(float(remainder.mean()))

beta_means = np.array(beta_means)
bound_means = np.array(bound_means)
remainder_means = np.array(remainder_means)

# Reproduction check against the manuscript's Table XII means.
table_xii = np.array([0.776, 0.777, 0.802, 0.837])

if np.max(np.abs(beta_means - table_xii)) > 0.001:
    raise SystemExit("Table XII reproduction check failed.")

if not np.allclose(
    bound_means + remainder_means,
    beta_means,
    atol=1e-12,
    rtol=1e-12
):
    raise SystemExit("Stacked-bar decomposition does not sum to beta_up.")

# ---------- Export figure data ----------
with (ROOT / "Figure_2a_bandwidth_data.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "total_bandwidth_ratio",
        "proposed_beta_up",
        "proposition7_lower_bound",
        "linear_analog_beta_up",
        "proposed_feasible",
    ])
    for i in range(len(rho)):
        writer.writerow([
            rho[i],
            proposed[i] if np.isfinite(proposed[i]) else "",
            prop7[i] if np.isfinite(prop7[i]) else "",
            linear[i] if np.isfinite(linear[i]) else "",
            bool(feasible[i]),
        ])

with (ROOT / "Figure_2b_decomposition_data.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "snr_db",
        "beta_up_mean",
        "proposition7_bandwidth_forced_mean",
        "energy_forced_remainder_mean",
    ])
    for values in zip(snrs, beta_means, bound_means, remainder_means):
        writer.writerow(values)

# ---------- Publication styling ----------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

blue = "#245B78"
orange = "#D18B35"
teal = "#32877D"
grey = "#777777"

fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(7.16, 3.15),
    gridspec_kw={"width_ratios": [1.35, 1.0]},
)

# Panel (a)
ax1.plot(
    rho, proposed,
    color=blue, marker="o", markersize=3.5,
    label="Proposed hybrid",
)
ax1.plot(
    rho, linear,
    color=orange, linestyle="--",
    marker="s", markersize=3.1,
    label="Linear analog",
)
ax1.plot(
    rho, prop7,
    color=teal, linestyle="-.",
    marker="^", markersize=3.2,
    label="Proposition 7 lower bound",
)

ax1.axhline(1.0, color=grey, linewidth=0.8, linestyle=":")
ax1.axhline(0.0, color=grey, linewidth=0.8, linestyle=":")

# Show tested infeasible proposed points explicitly, not as zero-valued data.
infeasible_rho = rho[~feasible]
if len(infeasible_rho):
    ax1.scatter(
        infeasible_rho,
        np.full(len(infeasible_rho), 0.965),
        marker="x", s=24, linewidths=1.2,
        color=grey, zorder=5,
        label="Hybrid infeasible",
    )
    ax1.annotate(
        "Infeasible at\nthese tested points",
        xy=(infeasible_rho[-1], 0.965),
        xytext=(0.135, 0.91),
        fontsize=6.5,
        color=grey,
        arrowprops={"arrowstyle": "->", "color": grey, "lw": 0.7},
    )

ax1.set_xlim(0.05, 0.51)
ax1.set_ylim(0.0, 1.04)
ax1.set_xticks(np.arange(0.05, 0.51, 0.05))
ax1.set_yticks(np.linspace(0, 1, 6))
ax1.set_xlabel(r"Total bandwidth ratio, $2n_{\rm eff}/k$")
ax1.set_ylabel(r"Pre-channel distortion fraction, $\beta_{\rm up}$")
ax1.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.8)
ax1.legend(loc="upper right", frameon=False, handlelength=2.1)
ax1.text(
    -0.12, 1.04, "(a)",
    transform=ax1.transAxes,
    fontweight="bold", fontsize=9,
)

# Panel (b)
x = np.arange(len(snrs))
width = 0.56

ax2.bar(
    x, bound_means, width,
    color=teal, edgecolor="white", linewidth=0.5,
    label="Bandwidth-forced (Prop. 7)",
)
ax2.bar(
    x, remainder_means, width,
    bottom=bound_means,
    color=orange, edgecolor="white", linewidth=0.5,
    label="Energy-forced remainder",
)

for i, (b, e, total) in enumerate(
    zip(bound_means, remainder_means, beta_means)
):
    ax2.text(
        i, b / 2, f"{b:.3f}",
        ha="center", va="center",
        color="white", fontsize=7, fontweight="bold",
    )
    ax2.text(
        i, b + e / 2, f"{e:.3f}",
        ha="center", va="center",
        color="#302713", fontsize=7,
    )
    ax2.text(
        i, total + 0.018, f"{total:.3f}",
        ha="center", va="bottom",
        fontsize=7.2, fontweight="bold",
    )

ax2.set_xticks(x, [str(s) for s in snrs])
ax2.set_ylim(0, 1.0)
ax2.set_yticks(np.linspace(0, 1, 6))
ax2.set_xlabel("SNR (dB)")
ax2.set_ylabel(r"Mean $\beta_{\rm up}$")
ax2.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.8)
ax2.legend(loc="upper left", frameon=False, fontsize=6.5)
ax2.text(
    -0.15, 1.04, "(b)",
    transform=ax2.transAxes,
    fontweight="bold", fontsize=9,
)

for ax in (ax1, ax2):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.7)

fig.tight_layout(pad=1.0, w_pad=2.0)

# Vector outputs for manuscript submission and a high-resolution preview.
pdf = ROOT / "Figure_2_PreChannel_Distortion.pdf"
svg = ROOT / "Figure_2_PreChannel_Distortion.svg"
png = ROOT / "Figure_2_PreChannel_Distortion.png"

fig.savefig(pdf, bbox_inches="tight")
fig.savefig(svg, bbox_inches="tight")
fig.savefig(png, dpi=600, bbox_inches="tight")
plt.close(fig)

print("FIGURE 2 VALIDATION: PASSED")
print(f"Panel (a): {len(rho)} tested bandwidth ratios")
print(f"Proposed feasible points: {int(feasible.sum())}/{len(rho)}")
print("\nPanel (b) mean decomposition:")

for snr, b, e, total in zip(
    snrs, bound_means, remainder_means, beta_means
):
    print(
        f"{snr:2d} dB | bandwidth={b:.6f} | "
        f"energy={e:.6f} | total={total:.6f}"
    )

for path in [pdf, svg, png]:
    print(f"Created: {path} ({path.stat().st_size:,} bytes)")

print("Created: results/Figure_2a_bandwidth_data.csv")
print("Created: results/Figure_2b_decomposition_data.csv")
