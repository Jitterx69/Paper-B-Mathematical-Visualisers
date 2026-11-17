import json
import numpy as np
from pathlib import Path
import evallab_fig2 as E

# Reproduce the validated pilot setup.
H, W, NG = 352, 640, 16
planes = E.load_yuv("bbb352.yuv", H, W, NG * E.G)
L = E.Layout([(H, W), (H // 2, W // 2), (H // 2, W // 2)])

Tg = E.G / 25
scale = (352 * 640) / (360 * 640)
Bc = 500e3 * Tg * scale
n_eff = 2.5e6 * Tg / np.log2(1 + 10) * scale

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

def proposition7_bound(kappa_coef, M):
    """Fractional order-statistic extension of the Proposition 7 bound."""
    k = len(kappa_coef)
    q = np.clip(k - float(M), 0.0, float(k))

    if q <= 0:
        return 0.0
    if q >= k:
        return 1.0

    j = int(np.floor(q))
    rem = q - j
    part = np.partition(kappa_coef, j)
    smallest_mass = part[:j].sum(dtype=np.float64)
    smallest_mass += rem * float(part[j])

    return float(smallest_mass / kappa_coef.sum(dtype=np.float64))

records = []

schemes = {
    "proposed": (True, True),
    "linear_analog": (False, False),
}

for snr in [6, 8, 10, 12]:
    for scheme, (use_floor, use_si) in schemes.items():
        rng = np.random.default_rng(0)
        Dprev = None

        for g in range(NG):
            st = ST[g]

            # Match the original runner's first-GoP and no-SI initialization.
            if g == 0 or not use_si:
                st = (st[0], st[1], np.zeros_like(st[2]), st[3])

            # z is only needed for reconstructed samples. The model allocation
            # recursion is driven by Dprev, exactly as in run_gop().
            out = E.run_gop(
                L, C[g], None, st, FL[g],
                snr, n_eff, rng,
                use_floor, use_si, Dprev, 0.0
            )

            if out is None:
                raise RuntimeError(
                    f"Infeasible allocation: {scheme}, {snr} dB, GoP {g}"
                )

            cid = L.cid
            kappa_c = out["kappa"][cid]
            Dm_c = out["Dm"][cid]
            act = out["act"]

            if not (
                len(kappa_c) == len(Dm_c) == len(act) == L.k
            ):
                raise RuntimeError(f"Coefficient-array shape mismatch at GoP {g}")

            # Realized coefficient-mask definition.
            Dcoef = np.where(act, Dm_c, kappa_c)
            beta_num_active = kappa_c[~act].sum(dtype=np.float64)
            beta_den_active = Dcoef.sum(dtype=np.float64)
            beta_active = beta_num_active / beta_den_active

            # Fractional chunk-allocation definition.
            frac = out["frac"]
            Dmodel = out["Dmodel"]
            beta_num_fractional = np.dot(L.n, (1.0 - frac) * out["kappa"])
            beta_den_fractional = np.dot(L.n, Dmodel)
            beta_fractional = beta_num_fractional / beta_den_fractional

            active_count = int(act.sum())
            M = float(out["M"])

            lb_M = proposition7_bound(kappa_c, M)
            lb_realized = proposition7_bound(
                kappa_c, max(M, float(active_count))
            )

            record = {
                "scheme": scheme,
                "snr_db": snr,
                "gop": g,
                "beta_active": float(beta_active),
                "beta_fractional": float(beta_fractional),
                "prop7_bound_M": float(lb_M),
                "prop7_bound_realized": float(lb_realized),
                "energy_remainder_fractional": float(beta_fractional - lb_M),
                "active_coefficients": active_count,
                "M_channel_dimensions": M,
                "active_minus_M": float(active_count - M),
                "kappa_sum": float(kappa_c.sum(dtype=np.float64)),
                "D_sum_active_mask": float(beta_den_active),
                "D_sum_fractional": float(beta_den_fractional),
                "nF": float(out["nF"]),
                "P": float(out["P"]),
            }
            records.append(record)

            Dprev = out["Dmodel"]

# Save the complete per-GoP record.
output = Path("results/fig2_gop_metrics.json")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(records, indent=2))

print(f"Saved {len(records)} GoP records to {output}")

# Summaries for the proposed design and linear analog reference.
for snr in [6, 8, 10, 12]:
    print(f"\nSNR = {snr} dB")

    for scheme in schemes:
        rows = [
            r for r in records
            if r["snr_db"] == snr and r["scheme"] == scheme
        ]

        beta_a = np.array([r["beta_active"] for r in rows])
        beta_f = np.array([r["beta_fractional"] for r in rows])

        print(
            f"{scheme:14s} "
            f"beta_active={beta_a.mean():.6f} "
            f"beta_fractional={beta_f.mean():.6f}"
        )

        if scheme == "proposed":
            lb = np.array([r["prop7_bound_M"] for r in rows])
            remainder = np.array([
                r["energy_remainder_fractional"] for r in rows
            ])
            excess = np.array([r["active_minus_M"] for r in rows])

            print(
                f"{'':14s} "
                f"Prop7_LB={lb.mean():.6f} "
                f"energy_remainder={remainder.mean():.6f} "
                f"min_bound_margin={remainder.min():.6e}"
            )
            print(
                f"{'':14s} "
                f"active_count_minus_M: "
                f"min={excess.min():.3f}, max={excess.max():.3f}"
            )

print("\nTable XII reference: proposed mean beta_up =")
print("6 dB: 0.776 | 8 dB: 0.777 | 10 dB: 0.802 | 12 dB: 0.837")
