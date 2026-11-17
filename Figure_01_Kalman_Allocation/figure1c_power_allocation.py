from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

OUT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------
# 1. Same synthetic variance field and declared operating point as Fig. 1(b)
# ---------------------------------------------------------------------
K = 20_000
A = 0.9
THETA_F = 0.206
NU = np.ones(K, dtype=float)  # normalized real-dimension noise variance
REFERENCE_GAIN_SQUARED = 3.0
MAX_ITER = 1_000
FIXED_POINT_TOL = 1e-11

idx = np.arange(1, K + 1, dtype=float)
lambda_i = 1e3 * (1.0 + idx / 20.0) ** -2
omega_i = (1.0 - A**2) * lambda_i
vartheta_floor = np.minimum(lambda_i, THETA_F)
floor_precision = np.maximum(1.0 / THETA_F - 1.0 / lambda_i, 0.0)

# Exactly the same total refinement-energy budget as Figure 1(b).
P = REFERENCE_GAIN_SQUARED * float(np.sum(vartheta_floor))


# ---------------------------------------------------------------------
# 2. Proposition 4 closed-form allocator
# ---------------------------------------------------------------------
def allocate(kappa: np.ndarray, weight: np.ndarray, budget: float,
             noise: np.ndarray):
    """Return dual price, per-component energy, gain-square and distortion.

    D_i = min(kappa_i, sqrt(xi * nu_i * weight_i))
    E_i = weight_i * g_i^2
        = [sqrt(nu_i * weight_i / xi)
           - nu_i * weight_i / kappa_i]_+
    with sum_i E_i = budget.
    """
    if np.any(kappa <= 0) or np.any(weight <= 0) or np.any(noise <= 0):
        raise ValueError("kappa, energy weights, and noise must be positive")
    if budget <= 0:
        raise ValueError("The refinement energy budget must be positive")

    def energy_at(xi: float) -> np.ndarray:
        return np.maximum(
            np.sqrt(noise * weight / xi) - noise * weight / kappa,
            0.0,
        )

    def total_energy(xi: float) -> float:
        return float(np.sum(energy_at(xi)))

    lo, hi = 1e-12, 1.0
    while total_energy(lo) < budget:
        lo *= 0.1
        if lo < 1e-300:
            raise RuntimeError("Could not bracket the dual variable from below")
    while total_energy(hi) > budget:
        hi *= 10.0
        if hi > 1e300:
            raise RuntimeError("Could not bracket the dual variable from above")

    xi = brentq(
        lambda q: total_energy(q) - budget,
        lo, hi, xtol=1e-14, rtol=1e-12,
    )
    energy = energy_at(xi)
    gain_squared = energy / weight
    distortion = 1.0 / (1.0 / kappa + gain_squared / noise)
    pinned = np.sqrt(xi * noise * weight)
    return xi, energy, gain_squared, distortion, pinned


# ---------------------------------------------------------------------
# 3. Proposed scheme: coupled prediction + floor + analog refinement
# ---------------------------------------------------------------------
D_previous = vartheta_floor.copy()
converged = False

for iteration in range(1, MAX_ITER + 1):
    mu_proposed = omega_i + A**2 * D_previous
    kappa_proposed = 1.0 / (1.0 / mu_proposed + floor_precision)

    xi_proposed, E_proposed, g2_proposed, D_proposed, pin_proposed = allocate(
        kappa_proposed, vartheta_floor, P, NU
    )

    fixed_point_error = float(np.max(
        np.abs(D_proposed - D_previous) / np.maximum(vartheta_floor, 1e-12)
    ))
    D_previous = D_proposed.copy()
    if fixed_point_error < FIXED_POINT_TOL:
        converged = True
        break

if not converged:
    raise RuntimeError("Proposed prediction-aware allocation did not converge")

# Recompute all proposed quantities from the converged state.
mu_proposed = omega_i + A**2 * D_previous
kappa_proposed = 1.0 / (1.0 / mu_proposed + floor_precision)
xi_proposed, E_proposed, g2_proposed, D_proposed, pin_proposed = allocate(
    kappa_proposed, vartheta_floor, P, NU
)


# ---------------------------------------------------------------------
# 4. Baseline 1: same digital floor, but no receiver-side prediction
# ---------------------------------------------------------------------
# With no side information, kappa_i = vartheta_i.
kappa_floor_no_si = vartheta_floor.copy()
(
    xi_floor_no_si,
    E_floor_no_si,
    g2_floor_no_si,
    D_floor_no_si,
    pin_floor_no_si,
) = allocate(kappa_floor_no_si, vartheta_floor, P, NU)


# ---------------------------------------------------------------------
# 5. Baseline 2: no floor and no side information (SoftCast special case)
# ---------------------------------------------------------------------
# Corollary 1: kappa_i = vartheta_i = lambda_i.
kappa_softcast = lambda_i.copy()
weight_softcast = lambda_i.copy()
(
    xi_softcast,
    E_softcast,
    g2_softcast,
    D_softcast,
    pin_softcast,
) = allocate(kappa_softcast, weight_softcast, P, NU)


# ---------------------------------------------------------------------
# 6. Mathematical and equal-energy audits
# ---------------------------------------------------------------------
scenarios = {
    "Proposed: floor + prediction": {
        "xi": xi_proposed, "energy": E_proposed,
        "gain_squared": g2_proposed, "distortion": D_proposed,
        "kappa": kappa_proposed, "weight": vartheta_floor,
        "pinned": pin_proposed,
    },
    "Floor, no side information": {
        "xi": xi_floor_no_si, "energy": E_floor_no_si,
        "gain_squared": g2_floor_no_si, "distortion": D_floor_no_si,
        "kappa": kappa_floor_no_si, "weight": vartheta_floor,
        "pinned": pin_floor_no_si,
    },
    "SoftCast: no floor / no side information": {
        "xi": xi_softcast, "energy": E_softcast,
        "gain_squared": g2_softcast, "distortion": D_softcast,
        "kappa": kappa_softcast, "weight": weight_softcast,
        "pinned": pin_softcast,
    },
}

summary_rows = []
for name, s in scenarios.items():
    energy = s["energy"]
    kappa = s["kappa"]
    weight = s["weight"]
    distortion = s["distortion"]
    pinned = s["pinned"]
    active = energy > 1e-12

    assert np.all(np.isfinite(energy))
    assert np.all(energy >= 0.0)
    assert np.all(distortion > 0.0)
    assert np.all(distortion <= kappa + 1e-10)
    assert np.allclose(np.sum(energy), P, rtol=1e-10, atol=1e-9)
    assert np.allclose(
        distortion,
        np.minimum(kappa, pinned),
        rtol=1e-9, atol=1e-11,
    )
    assert np.allclose(energy, weight * s["gain_squared"], rtol=1e-11, atol=1e-12)

    active_indices = np.flatnonzero(active)
    first_active = int(active_indices[0] + 1) if active_indices.size else None
    last_active = int(active_indices[-1] + 1) if active_indices.size else None

    summary_rows.append({
        "scheme": name,
        "dual_multiplier_xi": s["xi"],
        "total_refinement_energy": float(np.sum(energy)),
        "distortion_sum": float(np.sum(distortion)),
        "active_components": int(np.sum(active)),
        "first_active_index": first_active,
        "last_active_index": last_active,
    })

assert len({round(row["total_refinement_energy"], 8) for row in summary_rows}) == 1
assert np.isclose(summary_rows[0]["total_refinement_energy"], P, rtol=1e-10)

# The floor/no-side-information and SoftCast limiting cases are explicit.
assert np.allclose(kappa_floor_no_si, vartheta_floor, rtol=1e-12)
assert np.allclose(kappa_softcast, lambda_i, rtol=1e-12)
assert np.allclose(weight_softcast, lambda_i, rtol=1e-12)

# SoftCast high-SNR asymptotic law: g_i^2 ~ C * lambda_i^(-1/2).
# We use lambda >= 25*xi to stay away from the activation threshold;
# this is an asymptotic reference, not an exact identity at finite SNR.
high_snr_mask = (E_softcast > 0.0) & (lambda_i >= 25.0 * xi_softcast)
if np.sum(high_snr_mask) < 10:
    raise RuntimeError("Insufficient high-SNR points for the asymptotic reference")

# Normalize the theoretical slope -1/2 at the geometric-mean point.
log_C = float(np.mean(
    np.log(g2_softcast[high_snr_mask])
    + 0.5 * np.log(lambda_i[high_snr_mask])
))
C_reference = float(np.exp(log_C))
g2_reference = C_reference * lambda_i[high_snr_mask] ** -0.5

# Finite-range fitted slope is diagnostic only, not a theorem/acceptance test.
finite_range_slope = float(np.polyfit(
    np.log(lambda_i[high_snr_mask]),
    np.log(g2_softcast[high_snr_mask]),
    1,
)[0])


# ---------------------------------------------------------------------
# 7. Publication plot: energy allocation + asymptotic inset
# ---------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, (ax, ax_ref) = plt.subplots(
    1, 2, figsize=(8.0, 4.6),
    gridspec_kw={"width_ratios": [2.35, 1.0], "wspace": 0.34},
)

plot_specs = [
    (E_proposed, "Proposed: floor + prediction", "-", 1.8),
    (E_floor_no_si, "Floor, no side information", "--", 1.7),
    (E_softcast, "SoftCast: no floor / no side information", "-.", 1.7),
]
active_counts = [
    int(np.sum(E_proposed > 1e-12)),
    int(np.sum(E_floor_no_si > 1e-12)),
    int(np.sum(E_softcast > 1e-12)),
]
for (energy, label, linestyle, linewidth), n_active in zip(plot_specs, active_counts):
    y = np.where(energy > 0.0, energy, np.nan)
    ax.plot(
        idx, y, linestyle=linestyle, linewidth=linewidth,
        label=rf"{label}  ($M_{{\rm act}}={n_active:,}$)",
    )

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1, K)
ax.set_xlabel("Transform-coefficient index, $i$")
ax.set_ylabel(r"Per-component refinement energy, $E_i=w_i g_i^2$")
ax.grid(True, which="both", linestyle=":", linewidth=0.55, alpha=0.5)
ax.legend(frameon=False, loc="lower right", fontsize=7.0)

# Companion axis: gain-square and its high-SNR asymptotic law are not
# component-energy quantities, so they receive their own labelled axis.
lam_hs = lambda_i[high_snr_mask]
g2_hs = g2_softcast[high_snr_mask]
order = np.argsort(lam_hs)
ax_ref.plot(lam_hs[order], g2_hs[order], linewidth=1.45,
            label=r"SoftCast $g_i^2$")
ax_ref.plot(lam_hs[order], g2_reference[order], linestyle="--",
            linewidth=1.35, label=r"$\lambda_i^{-1/2}$ reference")
ax_ref.set_xscale("log")
ax_ref.set_yscale("log")
ax_ref.set_xlabel(r"Source variance, $\lambda_i$")
ax_ref.set_ylabel(r"Gain square, $g_i^2$")
ax_ref.set_title(r"High-SNR: reference slope $-1/2$", fontsize=8.5)
ax_ref.tick_params(axis="both", labelsize=7.5)
ax_ref.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)
ax_ref.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.9, fontsize=6.7, loc="lower left")

fig.suptitle(
    rf"Synthetic equal-energy comparison: $P={P:.1f}$, $k={K:,}$",
    fontsize=10, y=0.985,
)
fig.subplots_adjust(left=0.095, right=0.98, bottom=0.15, top=0.91, wspace=0.55)
fig.savefig(OUT / "Figure_1c_Power_Allocation_Comparison.pdf", bbox_inches="tight")
fig.savefig(OUT / "Figure_1c_Power_Allocation_Comparison.png", dpi=600, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------------
# 8. Export per-coefficient data and reproducibility summary
# ---------------------------------------------------------------------
data = pd.DataFrame({
    "coefficient_index": idx.astype(int),
    "lambda_i": lambda_i,
    "omega_i": omega_i,
    "floor_threshold": np.full(K, THETA_F),
    "floor_residual_vartheta_i": vartheta_floor,
    "proposed_mu_i": mu_proposed,
    "proposed_kappa_i": kappa_proposed,
    "proposed_energy_E_i": E_proposed,
    "proposed_gain_squared_g2": g2_proposed,
    "proposed_distortion_D_i": D_proposed,
    "floor_no_si_kappa_i": kappa_floor_no_si,
    "floor_no_si_energy_E_i": E_floor_no_si,
    "floor_no_si_gain_squared_g2": g2_floor_no_si,
    "floor_no_si_distortion_D_i": D_floor_no_si,
    "softcast_kappa_i": kappa_softcast,
    "softcast_energy_E_i": E_softcast,
    "softcast_gain_squared_g2": g2_softcast,
    "softcast_distortion_D_i": D_softcast,
})
data.to_csv(OUT / "Figure_1c_Power_Allocation_Data.csv", index=False)

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT / "Figure_1c_Power_Allocation_Summary.csv", index=False)

report = [
    "FIGURE 1(c) — NUMERICAL VALIDATION",
    "----------------------------------",
    f"Coefficients k                 : {K}",
    f"Floor threshold theta_F        : {THETA_F}",
    f"Prediction fixed-point iters   : {iteration}",
    f"Prediction fixed-point error   : {fixed_point_error:.3e}",
    f"Common refinement budget P    : {P:.12f}",
    f"Proposed total energy          : {np.sum(E_proposed):.12f}",
    f"Floor/no-SI total energy       : {np.sum(E_floor_no_si):.12f}",
    f"SoftCast total energy          : {np.sum(E_softcast):.12f}",
    f"Proposed active coefficients   : {np.sum(E_proposed > 1e-12)}",
    f"Floor/no-SI active coefficients: {np.sum(E_floor_no_si > 1e-12)}",
    f"SoftCast active coefficients   : {np.sum(E_softcast > 1e-12)}",
    f"SoftCast asymptotic range n    : {np.sum(high_snr_mask)}",
    f"Finite-range fitted slope      : {finite_range_slope:.6f} (diagnostic only)",
    "Equal total energy             : PASSED",
    "Proposition 4 distortion rule  : PASSED",
    "Gain-energy identity           : PASSED",
    "Floor/no-side-information case : PASSED",
    "SoftCast special case          : PASSED",
    "STATUS: ALL ASSERTIONS PASSED",
]
(OUT / "Figure_1c_Validation.txt").write_text("\n".join(report) + "\n")

print("\n".join(report))
print("\nExported:")
print("  Figure_1c_Power_Allocation_Comparison.pdf")
print("  Figure_1c_Power_Allocation_Comparison.png")
print("  Figure_1c_Power_Allocation_Data.csv")
print("  Figure_1c_Power_Allocation_Summary.csv")
print("  Figure_1c_Validation.txt")
