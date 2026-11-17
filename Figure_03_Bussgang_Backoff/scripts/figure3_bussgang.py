from pathlib import Path
import csv
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.optimize import minimize_scalar


# ============================================================
# Configuration: derived from manuscript Section VI-A
# ============================================================

SEED = 20261116
N_DRAWS = 4_000_000
CHUNK_SIZE = 500_000

LAMBDA_MC = np.array([1.0, 2.0, 4.0, 8.0])
PEAK_SNR_DB = np.array([10, 15, 20, 25, 30], dtype=float)

# Table III reference values
TABLE_LAMBDA_DB = np.array([0.75, 2.85, 4.41, 5.63, 6.63])
TABLE_SNDR_DB = np.array([6.42, 10.35, 14.34, 18.42, 22.61])
TABLE_SHORTFALL_DB = np.array([2.83, 1.80, 1.25, 0.95, 0.76])

# Clipping-avoidance reference
N_SAMPLES = 1024
DELTA_CLIP = 1e-3

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "results"
DATA = PROJECT / "data"
OUT.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)


# ============================================================
# Equation (18): Bussgang decomposition of the hard limiter
# ============================================================

def bussgang_alpha(lam):
    """Closed-form Bussgang gain alpha(Lambda)."""
    lam = np.asarray(lam, dtype=float)
    root = np.sqrt(lam)

    return (
        1.0 - np.exp(-lam)
        + (np.sqrt(np.pi) / 2.0) * root * erfc(root)
    )


def normalized_distortion(lam):
    """Closed-form sigma_d^2 / sigma^2."""
    alpha = bussgang_alpha(lam)

    return np.maximum(
        1.0 - np.exp(-np.asarray(lam, dtype=float)) - alpha**2,
        0.0,
    )


# ============================================================
# Equation (19): signal-to-noise-and-distortion ratio
# ============================================================

def sndr_linear(lam, rho_pk):
    alpha = bussgang_alpha(lam)
    distortion = normalized_distortion(lam)

    return (
        alpha**2 * rho_pk
        / (distortion * rho_pk + lam)
    )


def optimize_backoff(peak_snr_db):
    """Maximize Eq. (19) using a bounded search in log Lambda."""
    rho_pk = 10.0 ** (peak_snr_db / 10.0)

    result = minimize_scalar(
        lambda log_lam: -np.log(sndr_linear(np.exp(log_lam), rho_pk)),
        bounds=(np.log(1e-3), np.log(1e3)),
        method="bounded",
        options={"xatol": 1e-13},
    )

    if not result.success:
        raise RuntimeError(
            f"Back-off optimization failed at {peak_snr_db} dB."
        )

    lam_star = float(np.exp(result.x))
    lam_star_db = float(10.0 * np.log10(lam_star))
    sndr_star_db = float(
        10.0 * np.log10(sndr_linear(lam_star, rho_pk))
    )

    # Distortion-free amplifier at the same back-off.
    ideal_snr_db = float(peak_snr_db - lam_star_db)
    shortfall_db = ideal_snr_db - sndr_star_db

    return {
        "peak_snr_db": float(peak_snr_db),
        "lambda_star": lam_star,
        "lambda_star_db": lam_star_db,
        "sndr_star_db": sndr_star_db,
        "ideal_snr_same_backoff_db": ideal_snr_db,
        "shortfall_db": shortfall_db,
    }


# ============================================================
# Monte Carlo verification of Eq. (18)
# x ~ CN(0,1); y = x inside the limiter, A*x/|x| outside.
# ============================================================

def monte_carlo_limiter():
    rng = np.random.default_rng(SEED)

    count = len(LAMBDA_MC)
    sxx = np.zeros(count, dtype=np.float64)
    sxy = np.zeros(count, dtype=np.complex128)
    syy = np.zeros(count, dtype=np.float64)

    processed = 0

    while processed < N_DRAWS:
        n = min(CHUNK_SIZE, N_DRAWS - processed)

        # Unit-variance circular complex Gaussian samples.
        x = (
            rng.standard_normal(n)
            + 1j * rng.standard_normal(n)
        ) / np.sqrt(2.0)

        radius = np.abs(x)
        x_energy = float(np.vdot(x, x).real)

        for j, lam in enumerate(LAMBDA_MC):
            amplitude = np.sqrt(lam)
            gain = np.minimum(1.0, amplitude / radius)
            y = x * gain

            sxx[j] += x_energy
            sxy[j] += np.vdot(x, y)
            syy[j] += np.vdot(y, y).real

        processed += n

    # Least-squares Bussgang coefficient.
    alpha_mc = np.real(sxy / sxx)

    # Orthogonal residual variance normalized by input variance.
    distortion_mc = (
        syy - (np.abs(sxy) ** 2) / sxx
    ) / sxx

    alpha_exact = bussgang_alpha(LAMBDA_MC)
    distortion_exact = normalized_distortion(LAMBDA_MC)

    alpha_error = np.abs(alpha_mc - alpha_exact)
    distortion_error = np.abs(distortion_mc - distortion_exact)

    records = []

    for j, lam in enumerate(LAMBDA_MC):
        records.append({
            "lambda_linear": float(lam),
            "backoff_db": float(10.0 * np.log10(lam)),
            "alpha_mc": float(alpha_mc[j]),
            "alpha_closed_form": float(alpha_exact[j]),
            "alpha_abs_error": float(alpha_error[j]),
            "distortion_mc": float(distortion_mc[j]),
            "distortion_closed_form": float(distortion_exact[j]),
            "distortion_abs_error": float(distortion_error[j]),
            "complex_draws": N_DRAWS,
            "seed": SEED,
        })

    # Tolerances stated in the manuscript.
    if np.max(alpha_error) > 3e-4:
        raise RuntimeError(
            "Monte Carlo Bussgang-gain tolerance exceeded."
        )

    if np.max(distortion_error) > 2e-5:
        raise RuntimeError(
            "Monte Carlo distortion-variance tolerance exceeded."
        )

    return records


# ============================================================
# Table III optimization and clipping-avoidance reference
# ============================================================

optima = [
    optimize_backoff(pk)
    for pk in PEAK_SNR_DB
]

computed_lambda_db = np.array([
    r["lambda_star_db"] for r in optima
])
computed_sndr_db = np.array([
    r["sndr_star_db"] for r in optima
])
computed_shortfall_db = np.array([
    r["shortfall_db"] for r in optima
])

if np.max(np.abs(computed_lambda_db - TABLE_LAMBDA_DB)) > 0.01:
    raise RuntimeError("Optimal back-off does not reproduce Table III.")

if np.max(np.abs(computed_sndr_db - TABLE_SNDR_DB)) > 0.01:
    raise RuntimeError("Optimal SNDR does not reproduce Table III.")

if np.max(np.abs(computed_shortfall_db - TABLE_SHORTFALL_DB)) > 0.01:
    raise RuntimeError("SNDR shortfall does not reproduce Table III.")

lambda_avoid = np.log(N_SAMPLES / DELTA_CLIP)
avoid_db = float(10.0 * np.log10(lambda_avoid))


# ============================================================
# Monte Carlo check
# ============================================================

mc_records = monte_carlo_limiter()

with (DATA / "figure3_monte_carlo.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(mc_records[0].keys()))
    writer.writeheader()
    writer.writerows(mc_records)


# ============================================================
# Numerical data export
# ============================================================

# Panel (a): closed-form curves
backoff_a_db = np.linspace(-10.0, 14.0, 500)
lambda_a = 10.0 ** (backoff_a_db / 10.0)
alpha_a = bussgang_alpha(lambda_a)
distortion_a = normalized_distortion(lambda_a)

with (DATA / "figure3_panel_a_curve.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "backoff_db", "lambda_linear",
        "alpha", "normalized_distortion_variance",
    ])
    writer.writerows(zip(
        backoff_a_db, lambda_a, alpha_a, distortion_a
    ))

# Panel (b): SNDR curves
backoff_b_db = np.linspace(-5.0, 14.0, 500)
lambda_b = 10.0 ** (backoff_b_db / 10.0)

sndr_records = []

for pk in PEAK_SNR_DB:
    rho = 10.0 ** (pk / 10.0)
    values_db = 10.0 * np.log10(sndr_linear(lambda_b, rho))

    for x, y in zip(backoff_b_db, values_db):
        sndr_records.append({
            "peak_snr_db": float(pk),
            "backoff_db": float(x),
            "sndr_db": float(y),
        })

with (DATA / "figure3_panel_b_curves.csv").open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["peak_snr_db", "backoff_db", "sndr_db"]
    )
    writer.writeheader()
    writer.writerows(sndr_records)

with (DATA / "figure3_table_iii_reproduction.csv").open(
    "w", newline=""
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "peak_snr_db",
            "lambda_star",
            "lambda_star_db",
            "sndr_star_db",
            "ideal_snr_same_backoff_db",
            "shortfall_db",
        ],
    )
    writer.writeheader()
    writer.writerows(optima)

(OUT / "figure3_validation.json").write_text(json.dumps({
    "seed": SEED,
    "complex_gaussian_draws": N_DRAWS,
    "monte_carlo": mc_records,
    "table_iii": optima,
    "clipping_avoidance_lambda": float(lambda_avoid),
    "clipping_avoidance_backoff_db": avoid_db,
}, indent=2))


# ============================================================
# Publication figure
# ============================================================

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
    "lines.linewidth": 1.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

blue = "#245B78"
orange = "#D18B35"
teal = "#32877D"
grey = "#777777"

fig, (ax_a, ax_b) = plt.subplots(
    1, 2,
    figsize=(7.16, 3.35),
    gridspec_kw={"width_ratios": [1.0, 1.2]},
)

# ---------------- Panel (a) ----------------

ax_a.plot(
    backoff_a_db, alpha_a,
    color=blue, linewidth=1.8,
    label=r"Closed form $\alpha(\Lambda)$",
)

ax_a.scatter(
    [r["backoff_db"] for r in mc_records],
    [r["alpha_mc"] for r in mc_records],
    marker="o", s=28,
    facecolors="none", edgecolors=blue,
    linewidths=1.1, zorder=5,
    label="Monte Carlo gain",
)

ax_a.set_xlim(-10, 14)
ax_a.set_ylim(0, 1.05)
ax_a.set_xlabel(r"Amplifier back-off, $10\log_{10}\Lambda$ (dB)")
ax_a.set_ylabel(r"Bussgang gain, $\alpha(\Lambda)$", color=blue)
ax_a.tick_params(axis="y", labelcolor=blue)
ax_a.set_yticks(np.linspace(0, 1, 6))
ax_a.grid(axis="both", color="#D9D9D9", linewidth=0.45, alpha=0.8)

# The distortion variance is small compared with alpha, so use a
# clearly labelled secondary axis rather than hiding it near zero.
ax_a_r = ax_a.twinx()

ax_a_r.plot(
    backoff_a_db, distortion_a,
    color=orange, linestyle="--", linewidth=1.8,
    label=r"Closed form $\sigma_d^2/\sigma^2$",
)

ax_a_r.scatter(
    [r["backoff_db"] for r in mc_records],
    [r["distortion_mc"] for r in mc_records],
    marker="^", s=30,
    facecolors="none", edgecolors=orange,
    linewidths=1.1, zorder=5,
    label="Monte Carlo distortion",
)

ax_a_r.set_ylim(0, 0.05)
ax_a_r.set_ylabel(
    r"Normalized distortion variance, $\sigma_d^2/\sigma^2$",
    color=orange,
)
ax_a_r.tick_params(axis="y", labelcolor=orange)
ax_a_r.set_yticks(np.linspace(0, 0.05, 6))

h1, l1 = ax_a.get_legend_handles_labels()
h2, l2 = ax_a_r.get_legend_handles_labels()
ax_a.legend(
    h1 + h2, l1 + l2,
    loc="center right",
    frameon=False,
    fontsize=6.2,
    handlelength=2.0,
)

ax_a.text(
    -0.13, 1.04, "(a)",
    transform=ax_a.transAxes,
    fontweight="bold", fontsize=9,
)


# ---------------- Panel (b) ----------------

colors = {
    10.0: "#8B8B8B",
    15.0: "#D18B35",
    20.0: "#245B78",
    25.0: "#32877D",
    30.0: "#8064A2",
}

for result in optima:
    pk = result["peak_snr_db"]
    rho = 10.0 ** (pk / 10.0)

    curve_db = 10.0 * np.log10(
        sndr_linear(lambda_b, rho)
    )

    highlight = (pk == 20.0)

    ax_b.plot(
        backoff_b_db, curve_db,
        color=colors[pk],
        linewidth=2.4 if highlight else 1.35,
        label=rf"$\rho_{{pk}}={int(pk)}$ dB",
        zorder=3 if highlight else 2,
    )

    ax_b.scatter(
        result["lambda_star_db"],
        result["sndr_star_db"],
        marker="*",
        s=85 if highlight else 58,
        color=colors[pk],
        edgecolors="black",
        linewidths=0.45,
        zorder=6,
    )

# Clipping-avoidance reference from Ns=1024, delta=1e-3.
ax_b.axvline(
    avoid_db,
    color=grey,
    linestyle=":",
    linewidth=1.2,
    label=f"No-clipping reference ({avoid_db:.1f} dB)",
)

ax_b.text(
    avoid_db - 0.18, 1.5,
    f"{avoid_db:.1f} dB",
    rotation=90,
    color=grey,
    ha="right",
    va="bottom",
    fontsize=6.5,
)

# Explicitly identify the main 20 dB result.
r20 = next(r for r in optima if r["peak_snr_db"] == 20.0)

ax_b.annotate(
    r"$\Lambda^*=4.41$ dB",
    xy=(r20["lambda_star_db"], r20["sndr_star_db"]),
    xytext=(r20["lambda_star_db"] + 1.0, r20["sndr_star_db"] + 2.5),
    fontsize=7,
    arrowprops={"arrowstyle": "->", "lw": 0.8, "color": blue},
    color=blue,
)

ax_b.set_xlim(-5, 14)
ax_b.set_ylim(0, 26)
ax_b.set_xlabel(r"Amplifier back-off, $10\log_{10}\Lambda$ (dB)")
ax_b.set_ylabel("Signal-to-noise-and-distortion ratio (dB)")
ax_b.set_yticks(np.arange(0, 26, 5))
ax_b.grid(axis="both", color="#D9D9D9", linewidth=0.45, alpha=0.8)
ax_b.legend(loc="upper left", frameon=False, fontsize=6.5)

ax_b.text(
    -0.10, 1.04, "(b)",
    transform=ax_b.transAxes,
    fontweight="bold", fontsize=9,
)


# ---------------- Final formatting and exports ----------------

for ax in [ax_a, ax_a_r, ax_b]:
    ax.spines["top"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.7)

ax_a_r.spines["top"].set_visible(False)

fig.tight_layout(pad=1.0, w_pad=2.0)

pdf = OUT / "Figure_3_Bussgang_Backoff.pdf"
svg = OUT / "Figure_3_Bussgang_Backoff.svg"
png = OUT / "Figure_3_Bussgang_Backoff.png"

fig.savefig(pdf, bbox_inches="tight")
fig.savefig(svg, bbox_inches="tight")
fig.savefig(png, dpi=600, bbox_inches="tight")
plt.close(fig)


# ============================================================
# Validation report
# ============================================================

print("\nMONTE CARLO VALIDATION — EQUATION (18)")
print(f"Complex Gaussian draws: {N_DRAWS:,}")
print(f"Fixed random seed: {SEED}")

for r in mc_records:
    print(
        f"Lambda={r['lambda_linear']:.0f} "
        f"({r['backoff_db']:.2f} dB) | "
        f"|Delta alpha|={r['alpha_abs_error']:.3e} | "
        f"|Delta distortion|={r['distortion_abs_error']:.3e}"
    )

print("\nTABLE III REPRODUCTION")
print("Peak SNR | Lambda* dB | SNDR* dB | Shortfall dB")

for r in optima:
    print(
        f"{r['peak_snr_db']:8.0f} | "
        f"{r['lambda_star_db']:10.4f} | "
        f"{r['sndr_star_db']:9.4f} | "
        f"{r['shortfall_db']:12.4f}"
    )

print(
    f"\nClipping-avoidance reference: {avoid_db:.4f} dB "
    f"(Ns={N_SAMPLES}, delta={DELTA_CLIP:g})"
)

print("\nVALIDATION: PASSED")
print("Equation (18) Monte Carlo tolerances: passed.")
print("Equation (19) Table III reproduction: passed.")

for path in [pdf, svg, png]:
    print(f"Created: {path} ({path.stat().st_size:,} bytes)")

print("Numerical CSV files created in ./data/")
