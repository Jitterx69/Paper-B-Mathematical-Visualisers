import json
import numpy as np
from pathlib import Path
import evallab_fig2 as E

H, W, NG = 352, 640, 16
SNR = 10

planes = E.load_yuv("bbb352.yuv", H, W, NG * E.G)
L = E.Layout([(H, W), (H // 2, W // 2), (H // 2, W // 2)])

Tg = E.G / 25
scale = (352 * 640) / (360 * 640)
Bc = 500e3 * Tg * scale
n_eff_pilot = 2.5e6 * Tg / np.log2(1 + 10) * scale
rho_pilot = 2 * n_eff_pilot / L.k

def coefs(g):
    return np.concatenate([
        E.fwd(p[g * E.G:(g + 1) * E.G]).ravel()
        for p in planes
    ]).astype(np.float32)

def z_orig(g):
    return np.concatenate([
        E.si_plane(p[g * E.G - 1]).ravel()
        for p in planes
    ]).astype(np.float32)

C = [coefs(g) for g in range(NG)]
ST, FL = [], []

for g in range(NG):
    z0 = z_orig(g) if g > 0 else None
    st = E.gop_stats(L, C[g], z0)
    ST.append(st)
    FL.append(E.floor_design(L, st[0], st[1], st[2], Bc, C[g]))

def prop7(kappa, M):
    k = len(kappa)
    q = np.clip(k - float(M), 0.0, float(k))

    if q <= 0:
        return 0.0
    if q >= k:
        return 1.0

    j = int(np.floor(q))
    rem = q - j
    part = np.partition(kappa, j)
    mass = part[:j].sum(dtype=np.float64)
    mass += rem * float(part[j])
    return float(mass / kappa.sum(dtype=np.float64))

def evaluate(rho, use_floor, use_si):
    n_eff = rho * L.k / 2
    rng = np.random.default_rng(0)
    Dprev = None
    rows = []

    for g in range(NG):
        st = ST[g]

        if g == 0 or not use_si:
            st = (st[0], st[1], np.zeros_like(st[2]), st[3])

        out = E.run_gop(
            L, C[g], None, st, FL[g],
            SNR, n_eff, rng,
            use_floor, use_si, Dprev, 0.0
        )

        if out is None:
            return None

        cid = L.cid
        kappa_c = out["kappa"][cid]
        Dm_c = out["Dm"][cid]
        act = out["act"]

        Dcoef = np.where(act, Dm_c, kappa_c)
        beta_active = (
            kappa_c[~act].sum(dtype=np.float64)
            / Dcoef.sum(dtype=np.float64)
        )

        beta_fractional = (
            np.dot(L.n, (1.0 - out["frac"]) * out["kappa"])
            / np.dot(L.n, out["Dmodel"])
        )

        rows.append({
            "gop": g,
            "beta_active": float(beta_active),
            "beta_fractional": float(beta_fractional),
            "prop7_bound": (
                prop7(kappa_c, out["M"]) if use_floor and use_si else None
            ),
            "active_count": int(act.sum()),
            "M": float(out["M"]),
            "nF": float(out["nF"]),
        })

        Dprev = out["Dmodel"]

    beta = np.array([r["beta_fractional"] for r in rows])
    result = {
        "feasible": True,
        "n_eff": float(n_eff),
        "rho": float(rho),
        "beta_mean": float(beta.mean()),
        "beta_std_gop": float(beta.std(ddof=1)),
        "beta_active_mean": float(np.mean([r["beta_active"] for r in rows])),
        "gop_records": rows,
    }

    if use_floor and use_si:
        lb = np.array([r["prop7_bound"] for r in rows])
        result.update({
            "prop7_mean": float(lb.mean()),
            "energy_remainder_mean": float(np.mean(beta - lb)),
            "minimum_gop_bound_margin": float(np.min(beta - lb)),
        })

    return result

# Include the exact pilot operating point in the sweep.
rho_grid = np.unique(np.round(
    np.concatenate([np.linspace(0.06, 0.50, 17), [rho_pilot]]),
    12
))

results = []

print(f"Coefficient count k: {L.k}")
print(f"Pilot n_eff: {n_eff_pilot:.3f}")
print(f"Pilot total bandwidth ratio: {rho_pilot:.8f}")
print("\nRunning sweep at 10 dB...")

for idx, rho in enumerate(rho_grid, 1):
    proposed = evaluate(rho, True, True)
    linear = evaluate(rho, False, False)

    row = {
        "rho": float(rho),
        "proposed": proposed,
        "linear_analog": linear,
    }
    results.append(row)

    p = (
        f"{proposed['beta_mean']:.6f}"
        if proposed is not None else "INFEASIBLE"
    )
    lb = (
        f"{proposed['prop7_mean']:.6f}"
        if proposed is not None else "—"
    )
    a = (
        f"{linear['beta_mean']:.6f}"
        if linear is not None else "INFEASIBLE"
    )

    print(
        f"{idx:02d}/{len(rho_grid):02d} "
        f"rho={rho:.6f} | proposed={p} | "
        f"Prop7={lb} | linear={a}",
        flush=True
    )

path = Path("results/fig2_bandwidth_sweep.json")
path.write_text(json.dumps(results, indent=2))
print(f"\nSaved: {path}")

# Validate the exact pilot point against Step 16.
pilot = min(results, key=lambda x: abs(x["rho"] - rho_pilot))
p = pilot["proposed"]
a = pilot["linear_analog"]

if p is None or a is None:
    raise SystemExit("FAIL: Pilot operating point unexpectedly infeasible.")

print("\nPilot-point regression check:")
print(f"Proposed beta_up: {p['beta_mean']:.6f} (expected 0.801844)")
print(f"Linear beta_up:   {a['beta_mean']:.6f} (expected 0.073997)")
print(f"Proposition 7:    {p['prop7_mean']:.6f} (expected 0.308013)")

if not (
    abs(p["beta_mean"] - 0.801844) < 2e-5
    and abs(a["beta_mean"] - 0.073997) < 2e-5
    and abs(p["prop7_mean"] - 0.308013) < 2e-5
):
    raise SystemExit("FAIL: Pilot-point regression mismatch.")

if p["minimum_gop_bound_margin"] < -1e-10:
    raise SystemExit("FAIL: Proposition 7 bound violated in sweep.")

print("BANDWIDTH SWEEP PILOT CHECK: PASSED")
