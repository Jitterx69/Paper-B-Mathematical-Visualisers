from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ---------------------------------------------------------
# 1. MODEL PARAMETERS
# ---------------------------------------------------------

A_VALUES = [0.0, 0.5, 0.9, 0.99]

LAMBDA = 1.0
BETA = 3.0
INITIAL_DISTORTION = LAMBDA

N_GOPS = 60

OUTPUT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------
# 2. RICCATI RECURSION
# ---------------------------------------------------------

def riccati_recursion(a, omega, beta, d0, n_gops):

    distortion = np.zeros(n_gops + 1)
    distortion[0] = d0

    for g in range(1, n_gops + 1):

        predicted_variance = (
            omega + a**2 * distortion[g - 1]
        )

        distortion[g] = (
            predicted_variance
            / (1.0 + beta * predicted_variance)
        )

    return distortion


# ---------------------------------------------------------
# 3. ANALYTICAL STEADY-STATE SOLUTION
# ---------------------------------------------------------

def riccati_steady_state(a, omega, beta):

    B = 1.0 + beta * omega - a**2

    discriminant = (
        B**2 + 4.0 * beta * a**2 * omega
    )

    if discriminant < 0:
        raise ValueError("Invalid Riccati discriminant.")

    # Numerically stable positive root
    denominator = B + np.sqrt(discriminant)

    if denominator <= 0:
        raise ValueError("Invalid steady-state denominator.")

    return 2.0 * omega / denominator


# ---------------------------------------------------------
# 4. COMPUTE PARAMETER SWEEP
# ---------------------------------------------------------

gops = np.arange(N_GOPS + 1)

results = {}
csv_records = []

for a in A_VALUES:

    # Preserve the stationary source variance:
    # lambda = omega / (1 - a^2)

    omega = (1.0 - a**2) * LAMBDA

    distortion = riccati_recursion(
        a=a,
        omega=omega,
        beta=BETA,
        d0=INITIAL_DISTORTION,
        n_gops=N_GOPS
    )

    steady_state = riccati_steady_state(
        a=a,
        omega=omega,
        beta=BETA
    )

    results[a] = {
        "omega": omega,
        "distortion": distortion,
        "steady_state": steady_state
    }

    # Check convergence and theoretical bounds
    assert np.all(np.isfinite(distortion))
    assert np.all(distortion >= -1e-12)
    assert abs(distortion[-1] - steady_state) < 1e-8

    for g in gops:

        csv_records.append({
            "a": a,
            "omega": omega,
            "beta": BETA,
            "gop": g,
            "distortion": distortion[g],
            "steady_state": steady_state
        })


# ---------------------------------------------------------
# 5. VERIFY MANUSCRIPT TABLE VIII
# ---------------------------------------------------------

reference_a = 0.9
reference_omega = 0.19
reference_beta = 3.0

reference_value = riccati_steady_state(
    reference_a,
    reference_omega,
    reference_beta
)

expected_value = 0.164001746899620

error = abs(reference_value - expected_value)

assert error < 5e-13, (
    f"Table VIII validation failed: {reference_value}"
)

print("\nTABLE VIII VALIDATION")
print("------------------------------")
print(f"Calculated D_inf : {reference_value:.15f}")
print(f"Reference D_inf  : {expected_value:.15f}")
print(f"Absolute error   : {error:.3e}")
print("STATUS: PASSED")


# ---------------------------------------------------------
# 6. GEOMETRIC CONVERGENCE ENVELOPE
# ---------------------------------------------------------

a_ref = 0.9

d_ref = results[a_ref]["distortion"]
d_inf_ref = results[a_ref]["steady_state"]

# Contraction bound:
# |D(g) - D_inf| <= a^(2g) |D(0) - D_inf|

envelope = (
    d_inf_ref
    + (a_ref**2)**gops
    * abs(d_ref[0] - d_inf_ref)
)

assert np.all(d_ref <= envelope + 1e-12)

# ---------------------------------------------------------
# 7. PUBLICATION PLOT
# ---------------------------------------------------------

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

fig, ax = plt.subplots(figsize=(7.2, 4.5))

colors = {}

for a in A_VALUES:

    data = results[a]

    line, = ax.plot(
        gops,
        data["distortion"],
        linewidth=1.8,
        label=rf"$a={a}$"
    )

    colors[a] = line.get_color()

    ax.axhline(
        data["steady_state"],
        color=colors[a],
        linewidth=1.0,
        linestyle="--",
        alpha=0.85
    )


# Geometric upper convergence envelope for a = 0.9
ax.plot(
    gops,
    envelope,
    color=colors[a_ref],
    linewidth=1.2,
    linestyle=":",
    label="Geometric upper bound ($a=0.9$)"
)


ax.set_xlabel("Group-of-Pictures index, $g$")
ax.set_ylabel("Distortion variance, $D(g)$")

ax.set_xlim(0, N_GOPS)
ax.set_ylim(bottom=0)

ax.grid(
    True,
    which="major",
    linestyle=":",
    linewidth=0.6,
    alpha=0.5
)

handles, labels = ax.get_legend_handles_labels()

handles.append(
    Line2D(
        [0], [0],
        color="black",
        linewidth=1.0,
        linestyle="--"
    )
)

labels.append("Analytical steady state")

ax.legend(
    handles,
    labels,
    frameon=False,
    ncol=2,
    loc="upper right"
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "Figure_1a_Riccati_Convergence.pdf",
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT_DIR / "Figure_1a_Riccati_Convergence.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close(fig)


# ---------------------------------------------------------
# 8. EXPORT NUMERICAL DATA
# ---------------------------------------------------------

df = pd.DataFrame(csv_records)

df.to_csv(
    OUTPUT_DIR / "Figure_1a_Riccati_Data.csv",
    index=False
)


# ---------------------------------------------------------
# 9. PRINT VALIDATION SUMMARY
# ---------------------------------------------------------

print("\nPARAMETER SWEEP RESULTS")
print("-------------------------------------------")

for a, data in results.items():

    print(
        f"a={a:4.2f} | "
        f"omega={data['omega']:.6f} | "
        f"D_inf={data['steady_state']:.12f} | "
        f"D_final={data['distortion'][-1]:.12f}"
    )

print("\nFIGURE EXPORT")
print("-------------------------------------------")
print("PDF: Figure_1a_Riccati_Convergence.pdf")
print("PNG: Figure_1a_Riccati_Convergence.png")
print("CSV: Figure_1a_Riccati_Data.csv")

print("\nALL CHECKS PASSED.")
