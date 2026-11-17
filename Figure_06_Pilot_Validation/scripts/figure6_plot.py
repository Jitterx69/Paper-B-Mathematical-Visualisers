"""Publication plotting for Figure 6: pilot performance and model validation.

Inputs expected in the Figure_06_Pilot_Validation project:
  data/table_x_reference.csv
  results/figure6_proposed_per_gop.csv

Panel (a) uses the published Table X aggregate means and per-16-GoP minima.
Panel (b) uses the regenerated 64 proposed-scheme GoP model/measurement pairs.
No synthetic or fabricated observations are generated here.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
RESULTS = PROJECT / "results"
TABLE_X = DATA / "table_x_reference.csv"
PER_GOP = RESULTS / "figure6_proposed_per_gop.csv"
RESULTS.mkdir(parents=True, exist_ok=True)

SNRS_ALL = [2, 4, 6, 8, 10, 12]
SNRS_PROPOSED = [6, 8, 10, 12]

# Table X row -> plotting label / style.
SERIES = {
    "proposed": {
        "label": "Hybrid (floor + SI)",
        "color": "#245B78", "marker": "o", "linestyle": "-", "zorder": 8,
    },
    "floor_no_side_info": {
        "label": "Floor, no SI",
        "color": "#32877D", "marker": "s", "linestyle": "--", "zorder": 6,
    },
    "no_floor_side_info": {
        "label": "No floor, with SI",
        "color": "#8064A2", "marker": "D", "linestyle": "-.", "zorder": 5,
    },
    "linear_analog": {
        "label": "Linear analog",
        "color": "#D18B35", "marker": "^", "linestyle": ":", "zorder": 4,
    },
    "h264_zero_latency": {
        "label": "H.264, zero latency",
        "color": "#AA5555", "marker": "v", "linestyle": "-", "zorder": 3,
    },
    "h264_default": {
        "label": "H.264, default",
        "color": "#666666", "marker": "P", "linestyle": "-", "zorder": 2,
    },
}

RESIDUAL_COLORS = {
    6: "#245B78",
    8: "#D18B35",
    10: "#32877D",
    12: "#8064A2",
}


def read_table_x(path: Path) -> dict[tuple[int, str], dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Table X reference CSV: {path}")
    rows: dict[tuple[int, str], dict] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["snr_db"]), row["series"])
            row["mean_psnr_db"] = (
                float(row["mean_psnr_db"]) if row["mean_psnr_db"] else None
            )
            row["minimum_psnr_db"] = (
                float(row["minimum_psnr_db"]) if row["minimum_psnr_db"] else None
            )
            rows[key] = row
    return rows


def read_per_gop(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing per-GoP results: {path}\n"
            "Run scripts/figure6_per_gop_patched.py first."
        )
    out = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            out.append({
                "snr_db": int(row["snr_db"]),
                "gop_index": int(row["gop_index"]),
                "measured": float(row["measured_psnr_db"]),
                "model": float(row["model_psnr_db"]),
                "residual": float(row["model_minus_measured_db"]),
            })
    return out


def validate_inputs(table_x: dict, records: list[dict]) -> dict[int, dict]:
    if len(records) != 64:
        raise RuntimeError(f"Expected 64 per-GoP pairs; found {len(records)}")

    by_snr: dict[int, list[dict]] = {s: [] for s in SNRS_PROPOSED}
    seen = set()
    for r in records:
        if r["snr_db"] not in by_snr:
            raise RuntimeError(f"Unexpected per-GoP SNR: {r['snr_db']}")
        key = (r["snr_db"], r["gop_index"])
        if key in seen:
            raise RuntimeError(f"Duplicate per-GoP record: {key}")
        seen.add(key)
        if abs(r["residual"] - (r["model"] - r["measured"])) > 1e-9:
            raise RuntimeError(f"Residual identity failed for {key}")
        by_snr[r["snr_db"]].append(r)

    stats = {}
    for snr in SNRS_PROPOSED:
        rows = sorted(by_snr[snr], key=lambda z: z["gop_index"])
        if len(rows) != 16 or [r["gop_index"] for r in rows] != list(range(16)):
            raise RuntimeError(f"SNR {snr}: expected GoP indices 0..15 exactly once")

        measured = np.array([r["measured"] for r in rows])
        model = np.array([r["model"] for r in rows])
        residual = np.array([r["residual"] for r in rows])

        tx = table_x[(snr, "proposed")]
        tx_model = table_x[(snr, "proposed_model")]
        # Table X is rounded to 0.01 dB; compare regenerated full precision to it.
        checks = {
            "mean_vs_table_x_db": abs(float(measured.mean()) - tx["mean_psnr_db"]),
            "minimum_vs_table_x_db": abs(float(measured.min()) - tx["minimum_psnr_db"]),
            "model_mean_vs_table_x_db": abs(float(model.mean()) - tx_model["mean_psnr_db"]),
        }
        if checks["mean_vs_table_x_db"] > 0.015:
            raise RuntimeError(f"SNR {snr}: measured mean no longer matches Table X: {checks}")
        if checks["minimum_vs_table_x_db"] > 0.015:
            raise RuntimeError(f"SNR {snr}: minimum no longer matches Table X: {checks}")
        if checks["model_mean_vs_table_x_db"] > 0.015:
            raise RuntimeError(f"SNR {snr}: model mean no longer matches Table X: {checks}")
        if float(residual.mean()) <= 0:
            raise RuntimeError(f"SNR {snr}: expected positive model-minus-measured bias")

        stats[snr] = {
            "measured_mean": float(measured.mean()),
            "measured_min": float(measured.min()),
            "model_mean": float(model.mean()),
            "residual_mean": float(residual.mean()),
            "residual_min": float(residual.min()),
            "residual_max": float(residual.max()),
            "positive_fraction": float(np.mean(residual > 0)),
            "records": rows,
            "checks": checks,
        }
    return stats


table_x = read_table_x(TABLE_X)
records = read_per_gop(PER_GOP)
stats = validate_inputs(table_x, records)

# Confirm every Table X series is represented at every operating point.
for name in SERIES:
    for snr in SNRS_ALL:
        if (snr, name) not in table_x:
            raise RuntimeError(f"Table X reference missing row: {name}, {snr} dB")

# ------------------------------------
# Publication styling and panel layout
# ------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.3,
    "axes.labelsize": 8.2,
    "axes.titlesize": 8.5,
    "legend.fontsize": 5.8,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "axes.linewidth": 0.75,
    "lines.linewidth": 1.35,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig = plt.figure(figsize=(7.16, 4.05))
gs = GridSpec(
    2, 2, figure=fig,
    width_ratios=[1.62, 1.0],
    height_ratios=[2.65, 1.0],
    left=0.085, right=0.985, bottom=0.13, top=0.96,
    wspace=0.34, hspace=0.34,
)
ax_a = fig.add_subplot(gs[:, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_res = fig.add_subplot(gs[1, 1])

# ---------------- Panel (a): Table X performance ----------------
for series, style in SERIES.items():
    means, mins = [], []
    xs = []
    for snr in SNRS_ALL:
        row = table_x[(snr, series)]
        if row["status"] in ("infeasible", "not_reported") or row["mean_psnr_db"] is None:
            continue
        xs.append(snr)
        means.append(row["mean_psnr_db"])
        mins.append(row["minimum_psnr_db"])

    if not xs:
        continue
    means_arr = np.asarray(means, dtype=float)
    mins_arr = np.asarray(mins, dtype=float)
    lower = np.maximum(means_arr - mins_arr, 0.0)
    ax_a.errorbar(
        xs, means_arr,
        yerr=np.vstack([lower, np.zeros_like(lower)]),
        color=style["color"], marker=style["marker"],
        linestyle=style["linestyle"], markersize=3.3,
        capsize=2.0, capthick=0.65, elinewidth=0.7,
        linewidth=1.25, label=style["label"],
        zorder=style["zorder"],
    )

# Proposed model: separate series, no fabricated minimum whiskers.
model_x, model_y = [], []
for snr in SNRS_PROPOSED:
    row = table_x[(snr, "proposed_model")]
    model_x.append(snr)
    model_y.append(row["mean_psnr_db"])
ax_a.plot(
    model_x, model_y, color=SERIES["proposed"]["color"],
    linestyle=(0, (4, 2)), linewidth=1.2, marker=None,
    label="Hybrid analytical model", zorder=9,
)

# Explicit infeasible region: no PSNR coordinate is assigned to infeasible cases.
ax_a.axvspan(1.65, 4.35, color="#999999", alpha=0.10, zorder=0)
ax_a.text(
    3.0, 49.0, "Floor-based schemes\ninfeasible at 2 and 4 dB",
    ha="center", va="top", fontsize=6.0, color="#555555",
)
ax_a.set_xlim(1.6, 12.4)
ax_a.set_ylim(20, 50)
ax_a.set_xticks(SNRS_ALL)
ax_a.set_xlabel(r"Channel SNR, $\gamma$ (dB)")
ax_a.set_ylabel("Peak signal-to-noise ratio (dB)")
ax_a.grid(axis="both", color="#D9D9D9", linewidth=0.42, alpha=0.85)
ax_a.legend(
    loc="lower right", frameon=True, framealpha=0.92,
    facecolor="white", edgecolor="#BBBBBB", ncol=2,
    handlelength=1.8, columnspacing=0.9, borderpad=0.45,
    labelspacing=0.35,
)
ax_a.text(-0.12, 1.025, "(a)", transform=ax_a.transAxes,
          fontweight="bold", fontsize=9)

# ---------------- Panel (b): model/measurement parity ----------------
all_measured = np.array([r["measured"] for r in records])
all_model = np.array([r["model"] for r in records])
lo = float(min(all_measured.min(), all_model.min()) - 0.35)
hi = float(max(all_measured.max(), all_model.max()) + 0.35)

ax_b.fill_between(
    [lo, hi], [lo - 0.5, hi - 0.5], [lo + 0.5, hi + 0.5],
    color="#A8C5D6", alpha=0.28, label=r"$\pm0.5$ dB band", zorder=0,
)
ax_b.plot([lo, hi], [lo, hi], color="#333333", linewidth=0.9,
          linestyle="--", label=r"Identity, $y=x$")

for snr in SNRS_PROPOSED:
    srows = stats[snr]["records"]
    x = [r["measured"] for r in srows]
    y = [r["model"] for r in srows]
    ax_b.scatter(
        x, y, s=17, alpha=0.83,
        color=RESIDUAL_COLORS[snr], edgecolors="white", linewidths=0.25,
        label=f"{snr} dB", zorder=3,
    )

ax_b.set_xlim(lo, hi)
ax_b.set_ylim(lo, hi)
ax_b.set_aspect("equal", adjustable="box")
ax_b.set_xlabel("Measured PSNR (dB)")
ax_b.set_ylabel("Model PSNR (dB)")
ax_b.grid(color="#DDDDDD", linewidth=0.4, alpha=0.8)
ax_b.legend(loc="upper left", frameon=True, fontsize=5.5,
            ncol=2, columnspacing=0.7, handlelength=1.4,
            borderpad=0.35, labelspacing=0.3)
ax_b.text(-0.18, 1.08, "(b)", transform=ax_b.transAxes,
          fontweight="bold", fontsize=9)

# ---------------- Residual distribution under panel (b) ----------------
residual_groups = [
    np.array([r["residual"] for r in stats[snr]["records"]])
    for snr in SNRS_PROPOSED
]
positions = np.arange(1, len(SNRS_PROPOSED) + 1)
box = ax_res.boxplot(
    residual_groups, positions=positions, widths=0.45,
    patch_artist=True, showfliers=False, whis=(0, 100),
    medianprops={"color": "black", "linewidth": 0.9},
    whiskerprops={"color": "#555555", "linewidth": 0.65},
    capprops={"color": "#555555", "linewidth": 0.65},
)
for patch, snr in zip(box["boxes"], SNRS_PROPOSED):
    patch.set_facecolor(RESIDUAL_COLORS[snr])
    patch.set_alpha(0.30)
    patch.set_edgecolor(RESIDUAL_COLORS[snr])

# Show individual groups deterministically, without random jitter.
for pos, snr in zip(positions, SNRS_PROPOSED):
    vals = np.array([r["residual"] for r in stats[snr]["records"]])
    offsets = np.linspace(-0.13, 0.13, len(vals))
    ax_res.scatter(
        pos + offsets, vals, s=7.5,
        color=RESIDUAL_COLORS[snr], alpha=0.75,
        edgecolors="none", zorder=3,
    )
    mean_res = stats[snr]["residual_mean"]
    ax_res.plot(pos, mean_res, marker="D", color="black",
                markersize=3.0, zorder=5)
    ax_res.annotate(
        f"{mean_res:+.2f}", (pos, mean_res),
        xytext=(0, 5), textcoords="offset points",
        ha="center", fontsize=5.4,
    )

ax_res.axhline(0, color="#333333", linewidth=0.75, linestyle="--")
ax_res.axhline(0.5, color="#888888", linewidth=0.6, linestyle=":")
ax_res.axhline(-0.5, color="#888888", linewidth=0.6, linestyle=":")
ax_res.set_xlim(0.55, 4.45)
ax_res.set_xticks(positions, [str(s) for s in SNRS_PROPOSED])
ax_res.set_xlabel("Channel SNR (dB)")
ax_res.set_ylabel("Model − measured (dB)")
ax_res.grid(axis="y", color="#DDDDDD", linewidth=0.4, alpha=0.8)
ax_res.text(0.01, 1.12, "Per-GoP residuals (16 groups per SNR)",
            transform=ax_res.transAxes, fontsize=6.2, ha="left")

for ax in (ax_a, ax_b, ax_res):
    ax.spines["top"].set_visible(False)
    ax.tick_params(direction="out", length=2.6, width=0.65)

# Save vector masters and raster proof.
pdf = RESULTS / "Figure_6_Pilot_Performance_Validation.pdf"
svg = RESULTS / "Figure_6_Pilot_Performance_Validation.svg"
png = RESULTS / "Figure_6_Pilot_Performance_Validation.png"
fig.savefig(pdf, bbox_inches="tight")
fig.savefig(svg, bbox_inches="tight")
fig.savefig(png, dpi=600, bbox_inches="tight")
plt.close(fig)

# Write transparent numerical summary for manuscript drafting.
summary_path = RESULTS / "figure6_plot_summary.csv"
with summary_path.open("w", newline="") as f:
    fields = [
        "snr_db", "measured_mean_psnr_db", "measured_min_psnr_db",
        "model_mean_psnr_db", "mean_residual_db", "minimum_residual_db",
        "maximum_residual_db", "positive_residual_fraction",
    ]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for snr in SNRS_PROPOSED:
        st = stats[snr]
        writer.writerow({
            "snr_db": snr,
            "measured_mean_psnr_db": st["measured_mean"],
            "measured_min_psnr_db": st["measured_min"],
            "model_mean_psnr_db": st["model_mean"],
            "mean_residual_db": st["residual_mean"],
            "minimum_residual_db": st["residual_min"],
            "maximum_residual_db": st["residual_max"],
            "positive_residual_fraction": st["positive_fraction"],
        })

print("FIGURE 6 PLOTTING VALIDATION: PASS")
print("Panel (a): published Table X means and minimum-over-16-GoP values")
print("Panel (a): proposed model series from Table X; infeasible points omitted")
print("Panel (b): 64 regenerated proposed-scheme per-GoP pairs")
print("\nSNR | measured mean | min GoP | model mean | bias | positive residuals")
for snr in SNRS_PROPOSED:
    st = stats[snr]
    print(
        f"{snr:2d} dB | {st['measured_mean']:.6f} | "
        f"{st['measured_min']:.6f} | {st['model_mean']:.6f} | "
        f"{st['residual_mean']:+.6f} | {100*st['positive_fraction']:.1f}%"
    )
print("\nOutput files:")
for path in (pdf, svg, png, summary_path):
    print(f"{path} ({path.stat().st_size:,} bytes)")
