"""Figure 1(b): Prediction-aware variance allocation.

Synthetic steady-state numerical illustration based on manuscript
Equations (2), (8), (9), (13) and (14).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.optimize import brentq, minimize

OUT = Path(__file__).resolve().parent

# ------------------------------------------------------------
# 1. Declared operating point (not reported pilot measurements)
# ------------------------------------------------------------
K = 20_000
A = 0.9
THETA_F = 0.206
NU = np.ones(K)                 # normalized real-dimension noise
REFERENCE_GAIN_SQUARED = 3.0   # uniform-gain reference for energy budget
MAX_ITER = 1_000
FIXED_POINT_TOL = 1e-11

idx = np.arange(1, K + 1, dtype=float)
lambda_i = 1e3 * (1.0 + idx / 20.0) ** -2
omega_i = (1.0 - A**2) * lambda_i
vartheta_i = np.minimum(lambda_i, THETA_F)
floor_precision = np.maximum(1.0 / THETA_F - 1.0 / lambda_i, 0.0)

# Same energy as the reference choice x_i = g_i^2 / nu_i = 3.
P = REFERENCE_GAIN_SQUARED * np.sum(vartheta_i)

# ------------------------------------------------------------
# 2. Closed-form power allocator, Proposition 4
# ------------------------------------------------------------
def allocate(kappa, budget, nu, vartheta):
    def energy_at(xi):
        return np.maximum(
            np.sqrt(nu * vartheta / xi) - nu * vartheta / kappa,
            0.0,
        )

    def total_energy(xi):
        return float(np.sum(energy_at(xi)))

    lo, hi = 1e-12, 1.0
    while total_energy(lo) < budget:
        lo *= 0.1
    while total_energy(hi) > budget:
        hi *= 10.0

    xi = brentq(
        lambda z: total_energy(z) - budget,
        lo, hi, xtol=1e-14, rtol=1e-12,
    )
    energy = energy_at(xi)
    pinned = np.sqrt(xi * nu * vartheta)
    distortion = np.minimum(kappa, pinned)
    return xi, energy, pinned, distortion

# ------------------------------------------------------------
# 3. Solve the self-consistent steady-state allocation
# ------------------------------------------------------------
# Conservative initialization: the previous GoP is reconstructed
# using the protected floor only. Iterate prediction + allocation
# until the previous- and current-GoP distortion fields agree.
D_previous = vartheta_i.copy()
converged = False

for iteration in range(1, MAX_ITER + 1):
    mu_i = omega_i + A**2 * D_previous
    kappa_i = 1.0 / (1.0 / mu_i + floor_precision)

    xi, energy_i, pinned_i, D_i = allocate(
        kappa_i, P, NU, vartheta_i
    )

    fixed_point_error = float(np.max(
        np.abs(D_i - D_previous) / np.maximum(vartheta_i, 1e-12)
    ))

    D_previous = D_i.copy()
    if fixed_point_error < FIXED_POINT_TOL:
        converged = True
        break

if not converged:
    raise RuntimeError("Steady-state allocation failed to converge.")

# Recompute all state variables using the converged previous-GoP field.
mu_i = omega_i + A**2 * D_previous
kappa_i = 1.0 / (1.0 / mu_i + floor_precision)
xi, energy_i, pinned_i, D_i = allocate(kappa_i, P, NU, vartheta_i)
gain_squared = energy_i / vartheta_i

floor_coded = lambda_i > THETA_F
analog_active = energy_i > 1e-12
energy_truncated = ~analog_active

# ------------------------------------------------------------
# 4. Numerical and limiting-case validation
# ------------------------------------------------------------
assert np.all(np.isfinite(D_i))
assert np.all(kappa_i <= vartheta_i + 1e-10)
assert np.all(D_i <= kappa_i + 1e-10)
assert np.allclose(np.sum(energy_i), P, rtol=1e-10, atol=1e-9)
assert np.array_equal(analog_active, kappa_i > pinned_i)

# No-side-information special case: posterior variance equals floor residual.
kappa_no_side_information = 1.0 / (
    1.0 / lambda_i + floor_precision
)
assert np.allclose(kappa_no_side_information, vartheta_i, rtol=1e-10)

# ------------------------------------------------------------
# 5. Independent optimizer audit on 20 small random instances
# ------------------------------------------------------------
rng = np.random.default_rng(20261116)
optimizer_gaps = []

for _ in range(20):
    m = 12
    k_test = 10.0 ** rng.uniform(-2.0, 1.0, size=m)
    v_test = k_test * 10.0 ** rng.uniform(0.0, 1.0, size=m)
    nu_test = 10.0 ** rng.uniform(-1.0, 1.0, size=m)
    p_test = float(10.0 ** rng.uniform(-1.0, 1.0) * np.sum(v_test))

    _, e_closed, _, _ = allocate(k_test, p_test, nu_test, v_test)
    x_closed = e_closed / v_test

    def objective(x):
        return float(np.sum(1.0 / (1.0 / k_test + x / nu_test)))

    def gradient(x):
        d = 1.0 / (1.0 / k_test + x / nu_test)
        return -(d**2) / nu_test

    result = minimize(
        objective,
        np.full(m, p_test / np.sum(v_test)),
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, None)] * m,
        constraints=[{
            "type": "ineq",
            "fun": lambda x: p_test - np.dot(v_test, x),
            "jac": lambda x: -v_test,
        }],
        options={"ftol": 1e-13, "maxiter": 3000},
    )

    if not result.success:
        raise RuntimeError(f"Independent optimizer failed: {result.message}")

    closed_objective = objective(x_closed)
    gap = abs(closed_objective - result.fun) / max(abs(result.fun), 1e-12)
    optimizer_gaps.append(gap)

max_optimizer_gap = max(optimizer_gaps)
assert max_optimizer_gap < 1e-8

# ------------------------------------------------------------
# 6. Publication plot
# ------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "legend.fontsize": 7.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(8.0, 5.0))

active_last = int(idx[analog_active][-1]) if np.any(analog_active) else 0
floor_last = int(idx[floor_coded][-1]) if np.any(floor_coded) else 0

ax.axvspan(1, active_last + 0.5, alpha=0.045, zorder=0)
ax.axvspan(active_last + 0.5, K, alpha=0.025, zorder=0)
ax.axvspan(
    1, floor_last + 0.5, facecolor="none", edgecolor="0.8",
    hatch="////", linewidth=0, zorder=0,
)

curves = [
    (lambda_i, r"Source variance, $\lambda_i$"),
    (mu_i, r"Prediction-conditioned variance, $\mu_i$"),
    (vartheta_i, r"Floor residual, $\vartheta_i$"),
    (kappa_i, r"Posterior before analog, $\kappa_i$"),
    (D_i, r"Allocated distortion, $D_i$"),
    (pinned_i, r"Pinned level, $\sqrt{\xi\nu_i\vartheta_i}$"),
]
for values, label in curves:
    ax.plot(idx, values, linewidth=1.55, label=label)

ax.axhline(
    THETA_F, color="black", linestyle="--", linewidth=1.0,
    label=r"Floor threshold, $\theta_F=0.206$",
)
ax.axvline(active_last + 0.5, linestyle=":", linewidth=1.0)
ax.axvline(floor_last + 0.5, linestyle="-.", linewidth=1.0)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1, K)
ax.set_xlabel("Transform-coefficient index, $i$")
ax.set_ylabel("Variance / distortion (log scale)")
ax.grid(True, which="both", linestyle=":", linewidth=0.55, alpha=0.55)

region_legend = [
    Patch(alpha=0.12, label=r"Analog-active: $E_i>0$"),
    Patch(alpha=0.08, label=r"Energy-truncated: $E_i=0$"),
    Patch(facecolor="white", edgecolor="0.6", hatch="////",
          label=r"Floor-coded: $\lambda_i>\theta_F$"),
]
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles + region_legend,
    labels + [item.get_label() for item in region_legend],
    ncol=2, frameon=False, loc="best",
)

fig.tight_layout()
fig.savefig(OUT / "Figure_1b_Variance_Allocation.pdf", bbox_inches="tight")
fig.savefig(OUT / "Figure_1b_Variance_Allocation.png", dpi=600, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------
# 7. Export reproducible numerical data
# ------------------------------------------------------------
pd.DataFrame({
    "coefficient_index": idx.astype(int),
    "lambda_i": lambda_i,
    "omega_i": omega_i,
    "D_previous_steady_state": D_previous,
    "mu_i": mu_i,
    "vartheta_i": vartheta_i,
    "kappa_i": kappa_i,
    "pinned_level": pinned_i,
    "D_allocated": D_i,
    "energy_i": energy_i,
    "gain_squared": gain_squared,
    "noise_variance": NU,
    "floor_coded": floor_coded,
    "analog_active": analog_active,
    "energy_truncated": energy_truncated,
}).to_csv(OUT / "Figure_1b_Variance_Allocation.csv", index=False)

print("FIGURE 1(b) — NUMERICAL VALIDATION")
print("----------------------------------")
print(f"Steady-state iterations       : {iteration}")
print(f"Fixed-point relative error    : {fixed_point_error:.3e}")
print(f"Floor-coded components        : {floor_coded.sum()} / {K}")
print(f"Analog-active components      : {analog_active.sum()} / {K}")
print(f"Energy-truncated components   : {energy_truncated.sum()} / {K}")
print(f"Refinement energy budget P    : {P:.12f}")
print(f"Allocated total energy        : {energy_i.sum():.12f}")
print(f"Dual multiplier xi            : {xi:.12e}")
print(f"Independent optimizer gap     : {max_optimizer_gap:.3e}")
print("Energy constraint             : PASSED")
print("Posterior/floor bounds        : PASSED")
print("No-side-information limit     : PASSED")
print("STATUS: ALL ASSERTIONS PASSED")
print("Exported PDF, PNG and CSV.")
