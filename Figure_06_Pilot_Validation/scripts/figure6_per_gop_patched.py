"""Regenerate the proposed-scheme per-GoP model/measurement pairs.

Reproduces the validated runeval.py recursion for the four feasible SNRs.
Requires data/bbb352.yuv, data/eval_res_baseline.json, and
scripts/evallab_fig2.py in this project.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
RESULTS = PROJECT / "results"
SCRIPTS = PROJECT / "scripts"
RESULTS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SCRIPTS))

import evallab_fig2 as E  # noqa: E402

def recon_last_exact(chat):
    frames = []
    offset = 0

    for plane, size in zip(planes, L.sizes):
        block = chat[offset:offset + size].reshape(
            (E.G, plane.shape[1] // 8, plane.shape[2] // 8, 8, 8)
        )
        frames.append(E.last_frame(block))
        offset += size

    return frames


H, W, NG = 352, 640, 16
SNRS = [6, 8, 10, 12]
SEED = 0

video_path = DATA / "bbb352.yuv"
baseline_path = DATA / "eval_res_baseline.json"
reference_path = DATA / "table_x_reference.csv"

for required in [video_path, baseline_path, reference_path]:
    if not required.exists():
        raise FileNotFoundError(f"Required input missing: {required}")

expected_bytes = 43_253_760
if video_path.stat().st_size != expected_bytes:
    raise RuntimeError(
        f"Unexpected YUV size {video_path.stat().st_size:,}; "
        f"expected {expected_bytes:,} bytes."
    )

# Same source transform, group layout, floor and metadata construction as runeval.py.
planes = E.load_yuv(str(video_path), H, W, NG * E.G)
L = E.Layout([(H, W), (H // 2, W // 2), (H // 2, W // 2)])
Tg = E.G / 25
scale = (352 * 640) / (360 * 640)
Bc = 500e3 * Tg * scale
n_eff = 2.5e6 * Tg / np.log2(1 + 10) * scale


def coefs(g: int) -> np.ndarray:
    return np.concatenate([
        E.fwd(p[g * E.G:(g + 1) * E.G]).ravel()
        for p in planes
    ]).astype(np.float32)


def z_orig(g: int) -> np.ndarray:
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

baseline = json.loads(baseline_path.read_text())
with reference_path.open(newline="") as f:
    table_x = {
        (int(row["snr_db"]), row["series"]): row
        for row in csv.DictReader(f)
    }

# Cross-check the validated aggregate run against Table X.
# The floor-without-SI row has a known small mean discrepancy in the
# independently reproduced pilot grid (largest observed: 0.0581 dB).
# Do not alter simulation outputs to force a match; preserve and report it.
baseline_to_table = {
    "ours": "proposed",
    "ours_noSI": "floor_no_side_info",
    "nofloor_SI": "no_floor_side_info",
    "softcast": "linear_analog",
}
audit_records = []
for scheme_key, table_series in baseline_to_table.items():
    for snr in [2, 4, 6, 8, 10, 12]:
        result = baseline[f"{scheme_key}|{snr}"]
        row = table_x[(snr, table_series)]
        if row["status"] == "infeasible":
            if result is not None:
                raise RuntimeError(f"Expected infeasible {scheme_key} at {snr} dB")
            audit_records.append({
                "scheme": scheme_key, "table_series": table_series,
                "snr_db": snr, "status": "infeasible_match",
                "mean_abs_error_db": "", "minimum_abs_error_db": "",
            })
            continue

        if result is None:
            raise RuntimeError(f"Unexpected infeasibility: {scheme_key} at {snr} dB")

        mean_error = abs(float(result["psnr"]) - float(row["mean_psnr_db"]))
        min_error = abs(float(result["psnr_min"]) - float(row["minimum_psnr_db"]))

        # Strict tolerance for proposed / no-floor SI / linear. A documented
        # wider tolerance is allowed only for the floor-without-SI row.
        known_floor_no_si_deviation = scheme_key == "ours_noSI"
        mean_tolerance = 0.065 if known_floor_no_si_deviation else 0.015
        min_tolerance = 0.015

        if mean_error > mean_tolerance or min_error > min_tolerance:
            raise RuntimeError(
                f"Table X regression mismatch: {scheme_key}, {snr} dB; "
                f"mean error={mean_error}, minimum error={min_error}"
            )

        status = "known_small_deviation" if (mean_error > 0.015 or min_error > 0.015) else "match"
        audit_records.append({
            "scheme": scheme_key, "table_series": table_series,
            "snr_db": snr, "status": status,
            "rerun_mean_psnr_db": float(result["psnr"]),
            "table_x_mean_psnr_db": float(row["mean_psnr_db"]),
            "mean_abs_error_db": mean_error,
            "rerun_minimum_psnr_db": float(result["psnr_min"]),
            "table_x_minimum_psnr_db": float(row["minimum_psnr_db"]),
            "minimum_abs_error_db": min_error,
        })

audit_path = RESULTS / "table_x_reproduction_audit.csv"
with audit_path.open("w", newline="") as f:
    fields = sorted({key for record in audit_records for key in record})
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(audit_records)


records: list[dict] = []
summary: dict = {}

for snr in SNRS:
    rng = np.random.default_rng(SEED)
    Dprev = None
    zrec = None
    measured_values = []
    model_values = []

    for g in range(NG):
        z = None
        if zrec is not None:
            z = np.concatenate([
                E.si_plane(frame).ravel() for frame in zrec
            ]).astype(np.float32)

        st = ST[g]
        if g == 0:
            st = (st[0], st[1], np.zeros_like(st[2]), st[3])

        out = E.run_gop(
            L, C[g], z, st, FL[g], snr, n_eff, rng,
            True, True, Dprev, 0.0
        )
        if out is None:
            raise RuntimeError(f"Proposed scheme infeasible at {snr} dB, GoP {g}")

        mse = float(np.mean((out["chat"] - C[g]) ** 2))
        measured = float(10 * np.log10(255**2 / mse))
        model_distortion = float(np.sum(L.n * out["Dmodel"]) / L.k)
        model = float(10 * np.log10(255**2 / model_distortion))
        residual = model - measured

        records.append({
            "snr_db": snr,
            "gop_index": g,
            "measured_psnr_db": measured,
            "model_psnr_db": model,
            "model_minus_measured_db": residual,
            "mse_measured": mse,
            "distortion_model": model_distortion,
            "floor_uses_fraction": float(out["nF"] / n_eff),
        })
        measured_values.append(measured)
        model_values.append(model)

        Dprev = out["Dmodel"]
        zrec = [np.clip(frame, 0, 255) for frame in recon_last_exact(out["chat"])]

    measured_mean = float(np.mean(measured_values))
    measured_min = float(np.min(measured_values))
    model_mean = float(np.mean(model_values))
    residual_mean = float(np.mean(np.array(model_values) - np.array(measured_values)))

    key = f"ours|{snr}"
    if key not in baseline:
        raise RuntimeError(f"Baseline result missing key: {key}")
    expected = baseline[key]

    checks = {
        "measured_mean_abs_error": abs(measured_mean - float(expected["psnr"])),
        "measured_min_abs_error": abs(measured_min - float(expected["psnr_min"])),
        "model_mean_abs_error": abs(model_mean - float(expected["model"])),
    }
    if max(checks.values()) > 2e-6:
        raise RuntimeError(
            f"Regression mismatch at {snr} dB: {checks}; baseline={expected}"
        )
    if residual_mean <= 0:
        raise RuntimeError(f"Expected positive mean model bias at {snr} dB")

    summary[str(snr)] = {
        "measured_mean_psnr_db": measured_mean,
        "measured_min_psnr_db": measured_min,
        "model_mean_psnr_db": model_mean,
        "mean_residual_db": residual_mean,
        "residual_min_db": float(np.min(np.array(model_values) - np.array(measured_values))),
        "residual_max_db": float(np.max(np.array(model_values) - np.array(measured_values))),
        "fraction_positive_residuals": float(np.mean(np.array(model_values) > np.array(measured_values))),
        "baseline_regression_checks": checks,
        "gop_count": NG,
    }

out_csv = RESULTS / "figure6_proposed_per_gop.csv"
with out_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
    writer.writeheader()
    writer.writerows(records)

out_json = RESULTS / "figure6_per_gop_summary.json"
out_json.write_text(json.dumps({
    "source_video": "Big Buck Bunny, precompressed 1280x720 H.264; processed to 640x352 yuv420p",
    "channel": "AWGN",
    "noise_realizations": 1,
    "rng_seed_per_snr": SEED,
    "snr_db": SNRS,
    "gop_count_per_snr": NG,
    "total_per_gop_pairs": len(records),
    "coefficient_count_per_gop": L.k,
    "n_eff_complex_uses": float(n_eff),
    "bandwidth_ratio_2n_eff_over_k": float(2 * n_eff / L.k),
    "summary_by_snr": summary,
}, indent=2))

print("FIGURE 6 PER-GoP EXTRACTION")
print("Table X aggregate comparison audit:")
for rec in audit_records:
    if rec["status"] == "known_small_deviation":
        print(
            f"WARNING: {rec['scheme']} at {rec['snr_db']} dB differs from Table X "
            f"by {rec['mean_abs_error_db']:.4f} dB in mean PSNR; "
            f"minimum difference={rec['minimum_abs_error_db']:.4f} dB. "
            "Published Table X values will be retained in Panel (a)."
        )
print(f"Audit CSV: {audit_path}")
print(f"Source bytes: {video_path.stat().st_size:,}")
print(f"Coefficient count: {L.k:,}")
print(f"Effective complex channel uses: {n_eff:.6f}")
print(f"Total bandwidth ratio: {2*n_eff/L.k:.8f}")
print(f"Per-GoP model/measurement pairs: {len(records)}")
print("\nSNR | measured mean | min GoP | model mean | mean residual")
for snr in SNRS:
    s = summary[str(snr)]
    print(
        f"{snr:>2} dB | {s['measured_mean_psnr_db']:.6f} | "
        f"{s['measured_min_psnr_db']:.6f} | "
        f"{s['model_mean_psnr_db']:.6f} | {s['mean_residual_db']:+.6f}"
    )

print("\nREGRESSION CHECKS")
for snr in SNRS:
    s = summary[str(snr)]
    c = s["baseline_regression_checks"]
    print(
        f"{snr} dB: mean={c['measured_mean_abs_error']:.3e}, "
        f"minimum={c['measured_min_abs_error']:.3e}, "
        f"model={c['model_mean_abs_error']:.3e}"
    )

if len(records) != 64:
    raise RuntimeError("Expected exactly 64 per-GoP records")
print("\nVALIDATION: PASS — 64 per-GoP records; baseline regression passed.")
print(f"Created: {out_csv}")
print(f"Created: {out_json}")
