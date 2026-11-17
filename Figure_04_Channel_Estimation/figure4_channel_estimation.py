#!/usr/bin/env python3
"""Figure 4: Channel-estimation precision ceiling and power reallocation.

Implements manuscript Section VI-B, equations (21)-(23), Proposition 8,
and the Figure 4 development plan. Outputs are anchored to this script's
project root (the parent directory of scripts/).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------
# Project and reproducibility configuration
# ---------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
DATA = PROJECT / "data"
RESULTS = PROJECT / "results"
DATA.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

SEED = 20261117
N_MC = 400_000                     # Table IV: 4 x 10^5 Rayleigh draws
M_BLOCK = 100
PILOT_SNR_DB = (5, 10, 20)

# Synthetic spectrum shared with Figure 1 allocation illustration.
K = 20_000
A = 0.9
THETA_F = 0.206
NU_VALUE = 1.0
ENERGY_MULTIPLIER = 3.0

SIGMA2_LEVELS = [0.0, 1.0 / 401.0, 1.0 / 41.0, 1.0 / 13.65]
SIGMA_LABELS = {
    0.0: r"Perfect CSI ($\sigma_e^2=0$)",
    1.0 / 401.0: r"$\sigma_e^2=1/401$",
    1.0 / 41.0: r"$\sigma_e^2=1/41$",
    1.0 / 13.65: r"$\sigma_e^2=1/13.65$",
}
COLORS = {
    0.0: "#245B78",
    1.0 / 401.0: "#32877D",
    1.0 / 41.0: "#D18B35",
    1.0 / 13.65: "#8064A2",
}

TABLE_IV_M100 = {
    5: {"np": 8, "rate": 1.455},
    10: {"np": 6, "rate": 2.552},
    20: {"np": 4, "rate": 5.351},
}
TABLE_IV_PERFECT = {5: 1.716, 10: 2.906, 20: 5.884}


# ---------------------------------------------------------------------
# Proposition 4 and Proposition 8 allocation solvers
# ---------------------------------------------------------------------
def solve_energy_allocation(
    kappa: np.ndarray,
    vartheta: np.ndarray,
    nu: np.ndarray,
    total_energy: float,
    sigma2: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve the allocation by its scalar dual variable.

    sigma2 == 0 implements Proposition 4 (perfect CSI).
    sigma2 > 0 implements Proposition 8, equation (22).

    Returns (component energies u_i, distortions D_i, dual multiplier xi).
    """
    kappa = np.asarray(kappa, dtype=np.float64)
    vartheta = np.asarray(vartheta, dtype=np.float64)
    nu = np.asarray(nu, dtype=np.float64)

    if not (kappa.shape == vartheta.shape == nu.shape):
        raise ValueError("kappa, vartheta, and nu must have identical shapes")
    if np.any(kappa <= 0) or np.any(vartheta <= 0) or np.any(nu <= 0):
        raise ValueError("All component variances/noise values must be positive")
    if total_energy <= 0:
        raise ValueError("Total energy must be positive")
    if sigma2 < 0:
        raise ValueError("sigma2 cannot be negative")

    xi_max = float(np.max(kappa**2 / (nu * vartheta)))

    if sigma2 == 0.0:
        # Proposition 4: u_i = [sqrt(nu_i vartheta_i / xi)
        #                         - nu_i vartheta_i / kappa_i]^+.
        def energy_at_xi(xi: float) -> np.ndarray:
            return np.maximum(
                np.sqrt(nu * vartheta / xi) - nu * vartheta / kappa,
                0.0,
            )
    else:
        # Proposition 8, equation (22):
        # c_i = 1/kappa_i + 1/(sigma_i^2 vartheta_i)
        # z_i = sqrt(xi)c_i /
        #       [sqrt(nu_i/vartheta_i) + sqrt(xi)nu_i/(sigma_i^2 vartheta_i)]
        # u_i = [(1/z_i - nu_i)/sigma_i^2]^+.
        c = 1.0 / kappa + 1.0 / (sigma2 * vartheta)
        denom0 = np.sqrt(nu / vartheta)
        denom1 = nu / (sigma2 * vartheta)

        def energy_at_xi(xi: float) -> np.ndarray:
            root_xi = np.sqrt(xi)
            z = (root_xi * c) / (denom0 + root_xi * denom1)
            return np.maximum((1.0 / z - nu) / sigma2, 0.0)

    def budget_residual(xi: float) -> float:
        return float(np.sum(energy_at_xi(xi), dtype=np.float64) - total_energy)

    lo = xi_max * 1e-14
    hi = xi_max
    # Robustly expand the lower bracket if an unusual parameter set needs it.
    for _ in range(30):
        if budget_residual(lo) > 0:
            break
        lo *= 1e-2
    else:
        raise RuntimeError("Could not bracket the positive dual root from below")

    if budget_residual(hi) >= 0:
        raise RuntimeError("Could not bracket the positive dual root from above")

    xi = float(brentq(
        budget_residual, lo, hi,
        xtol=1e-14,
        rtol=1e-13,
        maxiter=500,
    ))
    u = energy_at_xi(xi)
    distortion = 1.0 / (
        1.0 / kappa + u / (vartheta * (nu + sigma2 * u))
    )

    energy_error = abs(float(u.sum(dtype=np.float64)) - total_energy)
    if energy_error > 2e-9 * max(1.0, total_energy):
        raise RuntimeError(f"Energy constraint mismatch: {energy_error:g}")
    if np.any(u < -1e-12) or np.any(distortion > kappa + 1e-10):
        raise RuntimeError("Non-negativity or posterior-distortion check failed")

    return u, distortion, xi


# ---------------------------------------------------------------------
# Build the common synthetic posterior field using the perfect-CSI
# steady-state recursion, matching Figure 1's synthetic example.
# ---------------------------------------------------------------------
def build_synthetic_field():
    i = np.arange(1, K + 1, dtype=np.float64)
    lam = 1e3 * (1.0 + i / 20.0) ** -2.0
    omega = (1.0 - A**2) * lam
    vartheta = np.minimum(lam, THETA_F)
    nu = np.full(K, NU_VALUE, dtype=np.float64)
    floor_precision = np.maximum(1.0 / THETA_F - 1.0 / lam, 0.0)
    total_energy = float(ENERGY_MULTIPLIER * vartheta.sum(dtype=np.float64))

    D_prev = vartheta.copy()
    converged = False
    for iteration in range(1, 1001):
        mu = omega + A**2 * D_prev
        kappa = 1.0 / (1.0 / mu + floor_precision)
        u0, D0, xi0 = solve_energy_allocation(
            kappa, vartheta, nu, total_energy, sigma2=0.0
        )
        rel_error = float(np.max(
            np.abs(D0 - D_prev) / np.maximum(D_prev, 1e-12)
        ))
        D_prev = D0
        if rel_error < 1e-10:
            converged = True
            break

    if not converged:
        raise RuntimeError("Perfect-CSI synthetic recursion did not converge")

    # Recompute the posterior field at the converged previous distortion.
    mu = omega + A**2 * D_prev
    kappa = 1.0 / (1.0 / mu + floor_precision)
    u0, D0, xi0 = solve_energy_allocation(
        kappa, vartheta, nu, total_energy, sigma2=0.0
    )

    return {
        "lambda": lam,
        "omega": omega,
        "vartheta": vartheta,
        "nu": nu,
        "mu": mu,
        "kappa": kappa,
        "D_prev": D_prev,
        "perfect_energy": u0,
        "perfect_distortion": D0,
        "perfect_xi": xi0,
        "total_energy": total_energy,
        "iterations": iteration,
        "fixed_point_relative_error": rel_error,
    }


# ---------------------------------------------------------------------
# Independent numerical optimizer check of Proposition 8.
# ---------------------------------------------------------------------
def validate_against_slsqp(n_cases: int = 10) -> float:
    rng = np.random.default_rng(7319)
    sigma2 = 1.0 / 41.0
    max_relative_gap = 0.0

    for _ in range(n_cases):
        n = 24
        kappa = np.exp(rng.normal(-0.2, 0.7, n))
        vartheta = np.exp(rng.normal(-0.3, 0.5, n))
        nu = np.exp(rng.normal(0.0, 0.2, n))
        P = float(rng.uniform(0.3, 1.2) * n)

        u_closed, D_closed, _ = solve_energy_allocation(
            kappa, vartheta, nu, P, sigma2
        )

        def objective(u):
            return float(np.sum(
                1.0 / (1.0 / kappa + u / (vartheta * (nu + sigma2 * u)))
            ))

        def gradient(u):
            D = 1.0 / (
                1.0 / kappa + u / (vartheta * (nu + sigma2 * u))
            )
            q_prime = nu / (vartheta * (nu + sigma2 * u) ** 2)
            return -D**2 * q_prime

        numerical = minimize(
            objective,
            x0=np.full(n, P / n),
            jac=gradient,
            method="SLSQP",
            bounds=[(0.0, None)] * n,
            constraints=[{
                "type": "ineq",
                "fun": lambda u, P=P: P - float(np.sum(u)),
                "jac": lambda u: -np.ones_like(u),
            }],
            options={"ftol": 1e-12, "maxiter": 1000, "disp": False},
        )
        if not numerical.success:
            raise RuntimeError(f"SLSQP validation failed: {numerical.message}")

        j_closed = objective(u_closed)
        rel_gap = abs(j_closed - float(numerical.fun)) / max(
            abs(float(numerical.fun)), 1e-12
        )
        max_relative_gap = max(max_relative_gap, rel_gap)

        if rel_gap > 1e-8:
            raise RuntimeError(f"Proposition 8 optimizer gap too large: {rel_gap:g}")

    return max_relative_gap


# ---------------------------------------------------------------------
# Panel (c): equation (21), Rayleigh-fading Monte Carlo rate.
# |h_hat|^2 ~ Exponential(mean=1-sigma_e^2).
# ---------------------------------------------------------------------
def compute_pilot_rate_curves():
    rng = np.random.default_rng(SEED)
    unit_exponential = rng.exponential(scale=1.0, size=N_MC)
    npilot = np.arange(1, M_BLOCK, dtype=int)
    output = {}
    summary = []

    for snr_db in PILOT_SNR_DB:
        gamma = 10.0 ** (snr_db / 10.0)
        rates = np.empty(M_BLOCK - 1, dtype=np.float64)

        for idx, n_p in enumerate(npilot):
            sigma_e2 = 1.0 / (1.0 + n_p * gamma)
            hhat_power = (1.0 - sigma_e2) * unit_exponential
            effective_sinr = gamma * hhat_power / (sigma_e2 * gamma + 1.0)
            per_draw_rate = np.log2(1.0 + effective_sinr)
            rates[idx] = (M_BLOCK - n_p) / M_BLOCK * float(per_draw_rate.mean())

        # Perfect CSI, no-pilot Rayleigh reference (last row in Table IV).
        perfect_draw_rate = np.log2(1.0 + gamma * unit_exponential)
        perfect_rate = float(perfect_draw_rate.mean())

        best_index = int(np.argmax(rates))
        best_np = int(npilot[best_index])
        best_rate = float(rates[best_index])
        standard_error = float(
            (M_BLOCK - best_np) / M_BLOCK
            * np.std(
                np.log2(
                    1.0 + gamma * (1.0 - 1.0 / (1.0 + best_np * gamma))
                    * unit_exponential
                    / (gamma / (1.0 + best_np * gamma) + 1.0)
                ),
                ddof=1,
            ) / np.sqrt(N_MC)
        )

        output[snr_db] = {
            "np": npilot.copy(),
            "rates": rates,
            "perfect_rate": perfect_rate,
            "best_np": best_np,
            "best_rate": best_rate,
            "best_rate_standard_error": standard_error,
        }
        summary.append({
            "m": M_BLOCK,
            "snr_db": snr_db,
            "best_np": best_np,
            "pilot_overhead_percent": 100.0 * best_np / M_BLOCK,
            "best_rate_bits_per_channel_use": best_rate,
            "perfect_csi_no_pilot_rate": perfect_rate,
            "best_rate_standard_error": standard_error,
            "mc_draws": N_MC,
            "seed": SEED,
        })

    return output, summary


# ---------------------------------------------------------------------
# Run model, checks, data exports, and figure creation.
# ---------------------------------------------------------------------
def main():
    field = build_synthetic_field()
    lam = field["lambda"]
    vartheta = field["vartheta"]
    nu = field["nu"]
    kappa = field["kappa"]
    P = field["total_energy"]

    allocations = {}
    for sigma2 in SIGMA2_LEVELS:
        u, D, xi = solve_energy_allocation(kappa, vartheta, nu, P, sigma2)
        allocations[sigma2] = {"u": u, "D": D, "xi": xi}

        if abs(float(u.sum()) - P) > 2e-9 * P:
            raise RuntimeError(f"Power constraint failed for sigma2={sigma2}")

        if sigma2 > 0:
            lower_bound = 1.0 / (1.0 / kappa + 1.0 / (sigma2 * vartheta))
            if np.any(D + 1e-10 < lower_bound):
                raise RuntimeError(f"Equation (23) ceiling bound failed for sigma2={sigma2}")

    # Fixed-budget reallocation must conserve total energy exactly.
    reallocation_balance = {}
    reallocation_transfer = {}
    u_perfect = allocations[0.0]["u"]

    for sigma2 in SIGMA2_LEVELS:
        delta_u = allocations[sigma2]["u"] - u_perfect
        net_shift = float(delta_u.sum(dtype=np.float64))
        transferred = 0.5 * float(np.abs(delta_u).sum(dtype=np.float64))

        reallocation_balance[str(sigma2)] = net_shift
        reallocation_transfer[str(sigma2)] = {
            "transferred_energy": transferred,
            "fraction_of_total_energy": transferred / P,
            "max_absolute_component_shift": float(np.max(np.abs(delta_u))),
            "energy_balance_residual": net_shift,
        }

        if abs(net_shift) > 2e-9 * P:
            raise RuntimeError(
                f"Energy conservation failed for sigma2={sigma2}"
            )

        if sigma2 > 0 and transferred <= 1e-10 * P:
            raise RuntimeError(
                f"No measurable component redistribution for sigma2={sigma2}"
            )

    # Explicit sigma_e^2 -> 0 limit check at sigma_e^2 = 1e-6.
    u_eps, D_eps, xi_eps = solve_energy_allocation(
        kappa, vartheta, nu, P, sigma2=1e-6
    )
    D_perfect = allocations[0.0]["D"]
    distortion_limit_relative_error = abs(
        float(D_eps.sum()) - float(D_perfect.sum())
    ) / float(D_perfect.sum())
    if distortion_limit_relative_error > 1e-6:
        raise RuntimeError(
            f"sigma_e^2 -> 0 limit check failed: {distortion_limit_relative_error:g}"
        )

    optimizer_gap = validate_against_slsqp(10)

    # Panel (a): effective SINR u/(nu + sigma_e^2 u), nu=1, and ceilings.
    energy_grid = np.logspace(-2, 4, 600)
    panel_a = {}
    for sigma2 in SIGMA2_LEVELS:
        effective_sinr = energy_grid / (NU_VALUE + sigma2 * energy_grid)
        effective_sinr_db = 10.0 * np.log10(effective_sinr)
        ceiling_db = None if sigma2 == 0 else float(10.0 * np.log10(1.0 / sigma2))
        panel_a[sigma2] = {
            "effective_sinr_db": effective_sinr_db,
            "ceiling_db": ceiling_db,
        }

    # Validate monotone approach to each finite ceiling.
    for sigma2 in SIGMA2_LEVELS:
        y = panel_a[sigma2]["effective_sinr_db"]
        if np.any(np.diff(y) < -1e-10):
            raise RuntimeError("Panel (a) effective-SINR curve is not monotone")
        ceiling = panel_a[sigma2]["ceiling_db"]
        if ceiling is not None and y[-1] > ceiling + 1e-8:
            raise RuntimeError("Effective-SINR curve exceeds its asymptotic ceiling")

    # Panel (b): export per-component allocation on the common variance field.
    order = np.argsort(-lam)
    rank = np.arange(1, K + 1)
    allocation_rows = []
    allocation_summary = []
    for sigma2 in SIGMA2_LEVELS:
        u = allocations[sigma2]["u"]
        D = allocations[sigma2]["D"]
        allocation_summary.append({
            "sigma_e2": float(sigma2),
            "dual_xi": float(allocations[sigma2]["xi"]),
            "total_energy": float(u.sum(dtype=np.float64)),
            "total_distortion_sum": float(D.sum(dtype=np.float64)),
            "active_components": int(np.count_nonzero(u > 1e-10)),
            "max_component_energy": float(u.max()),
        })
    for j in range(K):
        row = {
            "component_rank_descending_lambda": int(rank[j]),
            "lambda": float(lam[order[j]]),
            "vartheta": float(vartheta[order[j]]),
            "kappa": float(kappa[order[j]]),
        }
        for sigma2 in SIGMA2_LEVELS:
            key = "perfect_csi" if sigma2 == 0 else f"sigma2_{sigma2:.12g}"
            row[key] = float(allocations[sigma2]["u"][order[j]])
        allocation_rows.append(row)

    with (DATA / "figure4_panel_a_precision_ceiling.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["component_energy_u", "sigma_e2", "effective_sinr_db", "ceiling_db"])
        for sigma2 in SIGMA2_LEVELS:
            for u_val, sinr_val in zip(energy_grid, panel_a[sigma2]["effective_sinr_db"]):
                writer.writerow([
                    float(u_val), float(sigma2), float(sinr_val),
                    "" if panel_a[sigma2]["ceiling_db"] is None else panel_a[sigma2]["ceiling_db"],
                ])

    with (DATA / "figure4_panel_b_allocations.csv").open("w", newline="") as f:
        fieldnames = list(allocation_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(allocation_rows)

    with (DATA / "figure4_panel_b_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(allocation_summary[0].keys()))
        writer.writeheader()
        writer.writerows(allocation_summary)

    # Panel (c): the pilot-count rate curves and Table IV references.
    pilot_curves, pilot_summary = compute_pilot_rate_curves()
    with (DATA / "figure4_panel_c_pilot_rates.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["m", "mean_snr_db", "pilot_count", "rate_bits_per_channel_use"])
        for snr_db, values in pilot_curves.items():
            for n_p, rate in zip(values["np"], values["rates"]):
                writer.writerow([M_BLOCK, snr_db, int(n_p), float(rate)])

    for row in pilot_summary:
        snr_db = int(row["snr_db"])
        ref = TABLE_IV_M100[snr_db]
        if row["best_np"] != ref["np"]:
            raise RuntimeError(
                f"Table IV Np* mismatch at {snr_db} dB: {row['best_np']} vs {ref['np']}"
            )
        if abs(row["best_rate_bits_per_channel_use"] - ref["rate"]) > 0.002:
            raise RuntimeError(
                f"Table IV rate mismatch at {snr_db} dB: "
                f"{row['best_rate_bits_per_channel_use']:.6f} vs {ref['rate']:.3f}"
            )
        if abs(row["perfect_csi_no_pilot_rate"] - TABLE_IV_PERFECT[snr_db]) > 0.002:
            raise RuntimeError(
                f"Perfect-CSI Table IV reference mismatch at {snr_db} dB"
            )

    # Save a full machine-readable validation summary.
    validation = {
        "seed": SEED,
        "synthetic_model": {
            "k": K,
            "a": A,
            "theta_F": THETA_F,
            "nu_i": NU_VALUE,
            "energy_multiplier": ENERGY_MULTIPLIER,
            "total_energy_P": P,
            "fixed_point_iterations": field["iterations"],
            "fixed_point_relative_error": field["fixed_point_relative_error"],
            "perfect_csi_active_components": int(np.count_nonzero(allocations[0.0]["u"] > 1e-10)),
        },
        "allocation_summary": allocation_summary,
        "sigma_to_zero_relative_distortion_error": distortion_limit_relative_error,
        "energy_balance_residual_vs_perfect_csi_by_sigma2": reallocation_balance,
        "component_energy_transfer_vs_perfect_csi_by_sigma2": reallocation_transfer,
        "slsqp_max_relative_objective_gap_10_cases": optimizer_gap,
        "pilot_count_mc_draws": N_MC,
        "pilot_count_table_iv_reproduction": pilot_summary,
    }
    (RESULTS / "figure4_validation.json").write_text(json.dumps(validation, indent=2))

    # Cross-check the repeated Figure 1 synthetic example parameters.
    if abs(P - 1649.716411924705) > 1e-7:
        raise RuntimeError(f"Synthetic energy budget differs from Figure 1: P={P}")
    if int(np.count_nonzero(allocations[0.0]["u"] > 1e-10)) != 3529:
        raise RuntimeError("Perfect-CSI active count differs from the Figure 1 reference")

    # -----------------------------------------------------------------
    # Publication plot: two top panels, one full-width bottom panel.
    # -----------------------------------------------------------------
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.5,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "legend.fontsize": 6.2,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.75,
        "lines.linewidth": 1.45,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(7.16, 5.25))
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1.0, 0.92],
        hspace=0.43,
        wspace=0.36,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    # Panel (a): normalized effective SINR and precision ceilings.
    for sigma2 in SIGMA2_LEVELS:
        color = COLORS[sigma2]
        ax_a.plot(
            energy_grid,
            panel_a[sigma2]["effective_sinr_db"],
            color=color,
            linewidth=1.65 if sigma2 == 1.0 / 41.0 else 1.35,
            label=SIGMA_LABELS[sigma2],
        )
        ceiling = panel_a[sigma2]["ceiling_db"]
        if ceiling is not None:
            ax_a.axhline(
                ceiling, color=color, linestyle=":", linewidth=0.85, alpha=0.6
            )

    ax_a.annotate(
        r"$16.1$ dB ceiling ($1/41$)",
        xy=(1.7e3, 10.0 * np.log10(1.0 / (1.0 / 41.0))),
        xytext=(18.0, 20.0),
        fontsize=6.1,
        color=COLORS[1.0 / 41.0],
        arrowprops={"arrowstyle": "->", "lw": 0.7, "color": COLORS[1.0 / 41.0]},
    )
    ax_a.set_xscale("log")
    ax_a.set_xlim(1e-2, 1e4)
    ax_a.set_ylim(-20, 32.5)
    ax_a.set_xlabel(r"Allocated component energy, $u$ (with $\nu=1$)")
    ax_a.set_ylabel(r"Effective SINR, $10\log_{10}\!\left(\frac{u}{\nu+\sigma_e^2u}\right)$ (dB)")
    ax_a.grid(axis="both", color="#D9D9D9", linewidth=0.42, alpha=0.85)
    ax_a.legend(loc="lower right", frameon=False, fontsize=5.7, handlelength=2.0)
    ax_a.text(-0.13, 1.04, "(a)", transform=ax_a.transAxes, fontweight="bold", fontsize=9)

    # Panel (b): per-component energies, ordered by descending lambda_i.
    # Emphasize the 4-pilot, 10-dB case and a stronger-estimation-error case.
    rank = np.arange(1, K + 1)
    displayed_sigmas = [0.0, 1.0 / 41.0, 1.0 / 13.65]
    for sigma2 in displayed_sigmas:
        color = COLORS[sigma2]
        u_ranked = allocations[sigma2]["u"][order]
        if sigma2 == 0.0:
            label = "Proposition 4 (perfect CSI)"
        elif np.isclose(sigma2, 1.0 / 41.0):
            label = r"Proposition 8 ($\sigma_e^2=1/41$)"
        else:
            label = r"Proposition 8 ($\sigma_e^2=1/13.65$)"
        ax_b.semilogx(
            rank,
            u_ranked,
            color=color,
            linewidth=1.8 if np.isclose(sigma2, 1.0 / 41.0) else 1.35,
            label=label,
        )

    n0 = allocation_summary[0]["active_components"]
    n41 = next(x["active_components"] for x in allocation_summary if np.isclose(x["sigma_e2"], 1.0 / 41.0))
    nstrong = next(x["active_components"] for x in allocation_summary if np.isclose(x["sigma_e2"], 1.0 / 13.65))
    ax_b.text(
        0.97, 0.96,
        f"Active components\nPerfect CSI: {n0:,}\n$\\sigma_e^2=1/41$: {n41:,}\n$\\sigma_e^2=1/13.65$: {nstrong:,}",
        transform=ax_b.transAxes,
        ha="right", va="top", fontsize=5.8,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#BBBBBB", "lw": 0.5, "alpha": 0.92},
    )

    # Inset exposes redistribution that is small relative to each u_i:
    # positive and negative changes balance because total energy is fixed.
    delta_ax = ax_b.inset_axes([0.08, 0.09, 0.43, 0.36])
    show = rank <= 5000
    for sigma2 in [1.0 / 41.0, 1.0 / 13.65]:
        delta_u = (allocations[sigma2]["u"] - allocations[0.0]["u"])[order]
        delta_ax.plot(
            rank[show], delta_u[show],
            color=COLORS[sigma2], linewidth=0.9,
            label=(r"$1/41$" if np.isclose(sigma2, 1.0 / 41.0) else r"$1/13.65$"),
        )
    delta_ax.axhline(0.0, color="#666666", linestyle=":", linewidth=0.7)
    delta_ax.set_xlim(1, 5000)
    delta_ax.set_xlabel("Rank", fontsize=5.2, labelpad=1)
    delta_ax.set_ylabel(r"$\Delta u_i$", fontsize=5.4, labelpad=1)
    delta_ax.tick_params(labelsize=4.8, length=1.8, pad=1)
    delta_ax.grid(axis="y", color="#DDDDDD", linewidth=0.35)
    delta_ax.legend(loc="lower right", frameon=True, facecolor="white",
                    edgecolor="none", framealpha=0.88, fontsize=4.7,
                    handlelength=1.2, borderpad=0.2)

    ax_b.set_xlim(1, K)
    ax_b.set_ylim(bottom=0)
    ax_b.set_xlabel(r"Component rank (descending $\lambda_i$)")
    ax_b.set_ylabel(r"Allocated component energy, $u_i$")
    ax_b.grid(axis="both", color="#D9D9D9", linewidth=0.42, alpha=0.85)
    ax_b.legend(loc="lower right", frameon=True, facecolor="white",
                edgecolor="none", framealpha=0.9, fontsize=5.15, handlelength=1.7)
    ax_b.text(-0.13, 1.04, "(b)", transform=ax_b.transAxes, fontweight="bold", fontsize=9)

    # Panel (c): pilot-count rate curves for m=100.
    rate_colors = {5: "#245B78", 10: "#D18B35", 20: "#32877D"}
    n_show = np.arange(1, 21)
    for snr_db in PILOT_SNR_DB:
        vals = pilot_curves[snr_db]
        idx_show = vals["np"] <= 20
        color = rate_colors[snr_db]
        ax_c.plot(
            vals["np"][idx_show],
            vals["rates"][idx_show],
            color=color,
            marker="o",
            markersize=2.2,
            linewidth=1.35,
            label=rf"$\bar{{\gamma}}={snr_db}$ dB",
        )
        ax_c.scatter(
            [vals["best_np"]], [vals["best_rate"]],
            marker="*", s=62, color=color,
            edgecolors="black", linewidths=0.45, zorder=6,
        )
        ax_c.axhline(
            vals["perfect_rate"], color=color,
            linestyle="--", linewidth=0.85, alpha=0.55,
        )
        ax_c.annotate(
            f"$N_p^*={vals['best_np']}$",
            xy=(vals["best_np"], vals["best_rate"]),
            xytext=(vals["best_np"] + 0.8, vals["best_rate"] + (0.15 if snr_db != 20 else 0.25)),
            fontsize=6.2,
            color=color,
            arrowprops={"arrowstyle": "->", "lw": 0.65, "color": color},
        )

    ax_c.set_xlim(0, 20.5)
    ax_c.set_ylim(bottom=0)
    ax_c.set_xlabel(r"Pilot count, $N_p$ (block length $m=100$)")
    ax_c.set_ylabel(r"Achievable rate, $R(N_p)$ (bit/channel use)")
    ax_c.grid(axis="both", color="#D9D9D9", linewidth=0.42, alpha=0.85)
    perfect_proxy = Line2D([0], [0], color="#666666", linestyle="--", linewidth=0.9,
                           label="Perfect CSI, no pilots")
    handles, labels = ax_c.get_legend_handles_labels()
    ax_c.legend(handles + [perfect_proxy], labels + ["Perfect CSI, no pilots"],
                loc="upper right", frameon=True, facecolor="white",
                edgecolor="none", framealpha=0.92, ncol=4, fontsize=6.0)
    ax_c.text(0.005, 1.025, "(c)", transform=ax_c.transAxes,
              fontweight="bold", fontsize=9, ha="left", va="bottom")

    for ax in (ax_a, ax_b, ax_c):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=2.7, width=0.65)

    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.09, top=0.97,
                        wspace=0.34, hspace=0.43)
    pdf = RESULTS / "Figure_4_Channel_Estimation.pdf"
    svg = RESULTS / "Figure_4_Channel_Estimation.svg"
    png = RESULTS / "Figure_4_Channel_Estimation.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(png, dpi=600, bbox_inches="tight")
    plt.close(fig)

    # -----------------------------------------------------------------
    # Terminal validation report.
    # -----------------------------------------------------------------
    print("\nFIGURE 4 — SYNTHETIC ALLOCATION MODEL")
    print(f"k={K:,}; a={A}; theta_F={THETA_F}; nu_i={NU_VALUE}")
    print(f"Fixed-point iterations: {field['iterations']}")
    print(f"Fixed-point relative error: {field['fixed_point_relative_error']:.3e}")
    print(f"Total energy P: {P:.12f}")
    print(f"Perfect-CSI active components: {n0:,}")
    print("\nProposition 8 allocation summary")
    print("sigma_e^2 | active | sum(u_i) | sum(D_i)")
    for row in allocation_summary:
        print(
            f"{row['sigma_e2']:<10.6g} | {row['active_components']:>6,} | "
            f"{row['total_energy']:.9f} | {row['total_distortion_sum']:.9f}"
        )

    print("\nVALIDATION: PASS")
    print(f"10-case SLSQP maximum relative objective gap: {optimizer_gap:.3e}")
    print(f"sigma_e^2 -> 0 relative distortion difference: {distortion_limit_relative_error:.3e}")
    print("Energy constraints: passed for all allocation cases")
    print("Proposition 8 precision ceiling (Eq. 23): passed")
    print("Effective-SINR monotonicity/ceiling checks: passed")
    print("Component-wise energy transfer relative to perfect CSI:")
    for sigma2, item in reallocation_transfer.items():
        print(
            f"  sigma_e^2={sigma2}: "
            f"transferred={item['transferred_energy']:.6f}, "
            f"fraction={100*item['fraction_of_total_energy']:.4f}%, "
            f"max_component_shift={item['max_absolute_component_shift']:.6f}, "
            f"balance_residual={item['energy_balance_residual']:.3e}"
        )
    print("\nTABLE IV — m=100 pilot-count reproduction")
    print("SNR | Np* | MC rate | Table IV | Perfect CSI no-pilot rate")
    for row in pilot_summary:
        print(
            f"{row['snr_db']:>2} dB | {row['best_np']:>3} | "
            f"{row['best_rate_bits_per_channel_use']:.6f} | "
            f"{TABLE_IV_M100[int(row['snr_db'])]['rate']:.3f} | "
            f"{row['perfect_csi_no_pilot_rate']:.6f}"
        )
    print(f"Rayleigh Monte Carlo draws per curve: {N_MC:,}; seed={SEED}")

    for path in (pdf, svg, png):
        print(f"Created: {path} ({path.stat().st_size:,} bytes)")
    print(f"Numerical CSV files: {DATA}")
    print(f"Validation JSON: {RESULTS / 'figure4_validation.json'}")


if __name__ == "__main__":
    main()
