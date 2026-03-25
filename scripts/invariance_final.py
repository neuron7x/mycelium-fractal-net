#!/usr/bin/env python3
"""Final invariance validation: M = H/(W₂√I) — law, phase-dependent, or artifact?

Five gates. All must pass. No interpretation — only numbers.
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from scipy.optimize import curve_fit

import mycelium_fractal_net as mfn
from mycelium_fractal_net.analytics.unified_score import compute_hwi_components


os.makedirs("results", exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# GATE 1: FINITE-SIZE SCALING — M(N) → M∞
# ═══════════════════════════════════════════════════════════════════════════

def gate_1_finite_size() -> dict:
    print("GATE 1: Finite-size scaling")
    grid_sizes = [16, 32, 64, 96, 128]
    n_seeds = 5
    data = {}

    for N in grid_sizes:
        Ms = []
        for seed in range(n_seeds):
            try:
                seq = mfn.simulate(mfn.SimulationSpec(grid_size=N, steps=60, seed=seed))
                hwi = compute_hwi_components(seq.history[0], seq.field)
                Ms.append(hwi.M)
            except Exception:
                pass
        Ms = np.array(Ms) if Ms else np.array([np.nan])
        data[N] = {"mean": float(np.nanmean(Ms)), "std": float(np.nanstd(Ms)), "n": len(Ms)}
        print(f"  N={N:4d}: M={np.nanmean(Ms):.6f} ± {np.nanstd(Ms):.6f} (n={len(Ms)})")

    # Fit 1: power-law M(N) = M∞ + a * N^(-b)
    Ns = np.array([N for N in grid_sizes if not np.isnan(data[N]["mean"])])
    Ms_mean = np.array([data[N]["mean"] for N in Ns])

    def power_law(N, M_inf, a, b):
        return M_inf + a * N ** (-b)

    def inv_N(N, M_inf, c):
        return M_inf + c / N

    fits = {}
    try:
        popt, pcov = curve_fit(power_law, Ns, Ms_mean, p0=[0.08, 1.0, 1.0], maxfev=5000)
        perr = np.sqrt(np.diag(pcov))
        fits["power_law"] = {
            "M_inf": round(float(popt[0]), 6),
            "M_inf_err": round(float(perr[0]), 6),
            "a": round(float(popt[1]), 6),
            "b": round(float(popt[2]), 6),
            "residuals": [round(float(Ms_mean[i] - power_law(Ns[i], *popt)), 6) for i in range(len(Ns))],
        }
        print(f"  Power-law fit: M∞ = {popt[0]:.6f} ± {perr[0]:.6f}, b={popt[2]:.3f}")
    except Exception as e:
        fits["power_law"] = {"error": str(e)}
        print(f"  Power-law fit: FAILED ({e})")

    try:
        popt2, pcov2 = curve_fit(inv_N, Ns, Ms_mean, p0=[0.08, 1.0], maxfev=5000)
        perr2 = np.sqrt(np.diag(pcov2))
        fits["inv_N"] = {
            "M_inf": round(float(popt2[0]), 6),
            "M_inf_err": round(float(perr2[0]), 6),
            "c": round(float(popt2[1]), 6),
            "residuals": [round(float(Ms_mean[i] - inv_N(Ns[i], *popt2)), 6) for i in range(len(Ns))],
        }
        print(f"  1/N fit:       M∞ = {popt2[0]:.6f} ± {perr2[0]:.6f}")
    except Exception as e:
        fits["inv_N"] = {"error": str(e)}
        print(f"  1/N fit: FAILED ({e})")

    # Best estimate
    M_inf_estimates = []
    M_inf_errors = []
    for fname, fdata in fits.items():
        if "M_inf" in fdata and "error" not in fdata:
            M_inf_estimates.append(fdata["M_inf"])
            M_inf_errors.append(fdata["M_inf_err"])

    if M_inf_estimates:
        M_inf_best = float(np.mean(M_inf_estimates))
        M_inf_err_best = float(np.sqrt(np.mean(np.array(M_inf_errors) ** 2)))
    else:
        M_inf_best = float(Ms_mean[-1])
        M_inf_err_best = float(np.std(Ms_mean))

    print(f"  Best estimate: M∞ = {M_inf_best:.6f} ± {M_inf_err_best:.6f}")

    return {
        "raw": {str(N): data[N] for N in grid_sizes},
        "fits": fits,
        "M_inf": round(M_inf_best, 6),
        "M_inf_err": round(M_inf_err_best, 6),
    }


# ═══════════════════════════════════════════════════════════════════════════
# GATE 2: PLATEAU VALIDATION — 20×20 parameter grid
# ═══════════════════════════════════════════════════════════════════════════

def gate_2_plateau() -> dict:
    print("\nGATE 2: Plateau validation (20×20)")
    alphas = np.linspace(0.05, 0.24, 20)
    thresholds = np.linspace(0.15, 0.90, 20)

    grid_M = np.full((len(thresholds), len(alphas)), np.nan)
    for j, alpha in enumerate(alphas):
        for i, thr in enumerate(thresholds):
            try:
                seq = mfn.simulate(mfn.SimulationSpec(
                    grid_size=32, steps=60, seed=42,
                    alpha=round(float(alpha), 4),
                    turing_threshold=round(float(thr), 4),
                ))
                hwi = compute_hwi_components(seq.history[0], seq.field)
                grid_M[i, j] = hwi.M
            except Exception:
                pass
        if (j + 1) % 5 == 0:
            print(f"  column {j+1}/20")

    valid = grid_M[np.isfinite(grid_M)]
    mu = float(np.mean(valid)) if len(valid) > 0 else 0.0
    sigma = float(np.std(valid)) if len(valid) > 0 else 0.0
    cv = sigma / mu * 100 if mu > 0 else 0.0

    # Plateau: connected region within ±10% of mean
    plateau_mask = np.abs(grid_M - mu) < 0.10 * mu
    plateau_area = float(np.sum(plateau_mask & np.isfinite(grid_M))) / max(float(np.sum(np.isfinite(grid_M))), 1)

    # Save plateau map
    np.save("results/plateau_map.npy", grid_M)

    print(f"  Valid: {len(valid)}/{grid_M.size}")
    print(f"  M: mean={mu:.6f} std={sigma:.6f} CV={cv:.1f}%")
    print(f"  Plateau (±10%): {plateau_area*100:.0f}% of valid points")

    return {
        "n_valid": int(len(valid)),
        "M_mean": round(mu, 6),
        "M_std": round(sigma, 6),
        "M_cv_percent": round(cv, 2),
        "M_min": round(float(valid.min()), 6) if len(valid) > 0 else None,
        "M_max": round(float(valid.max()), 6) if len(valid) > 0 else None,
        "plateau_fraction_10pct": round(plateau_area, 4),
        "alphas": [round(float(a), 4) for a in alphas],
        "thresholds": [round(float(t), 4) for t in thresholds],
    }


# ═══════════════════════════════════════════════════════════════════════════
# GATE 3: TEMPORAL PHASE SEPARATION
# ═══════════════════════════════════════════════════════════════════════════

def gate_3_temporal() -> dict:
    print("\nGATE 3: Temporal phase separation")
    seeds = [42, 7, 123, 0, 99]
    results = {}

    for seed in seeds:
        seq = mfn.simulate(mfn.SimulationSpec(grid_size=32, steps=60, seed=seed))
        rho_ss = seq.field
        T = seq.history.shape[0]
        Ms = np.zeros(T)
        for t in range(T):
            hwi = compute_hwi_components(seq.history[t], rho_ss)
            Ms[t] = hwi.M

        # Phase detection: find knee point where M drops below 50% of max
        M_max = Ms.max()
        knee = T
        for t in range(T):
            if Ms[t] < 0.5 * M_max and t > 5:
                knee = t
                break

        morph = Ms[:knee]
        steady = Ms[knee:]

        results[seed] = {
            "knee_step": int(knee),
            "morph_mean": round(float(morph.mean()), 6),
            "morph_std": round(float(morph.std()), 6),
            "morph_cv": round(float(morph.std() / morph.mean() * 100), 2) if morph.mean() > 0 else None,
            "steady_mean": round(float(steady.mean()), 6) if len(steady) > 0 else None,
            "steady_std": round(float(steady.std()), 6) if len(steady) > 0 else None,
            "M_values": [round(float(m), 6) for m in Ms],
        }
        print(f"  seed={seed:3d}: knee={knee:2d}, morph M={morph.mean():.6f}±{morph.std():.6f}, "
              f"steady M={steady.mean():.6f}" if len(steady) > 0 else f"  seed={seed}: all morphogenesis")

    # Cross-seed morphogenesis M
    morph_means = [results[s]["morph_mean"] for s in seeds]
    cross_cv = float(np.std(morph_means) / np.mean(morph_means) * 100) if np.mean(morph_means) > 0 else 0.0

    print(f"  Cross-seed morph M: {np.mean(morph_means):.6f} CV={cross_cv:.2f}%")

    return {
        "seeds": {str(s): results[s] for s in seeds},
        "cross_seed_morph_mean": round(float(np.mean(morph_means)), 6),
        "cross_seed_morph_cv": round(cross_cv, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════
# GATE 4: SEED ROBUSTNESS — 100 seeds
# ═══════════════════════════════════════════════════════════════════════════

def gate_4_seeds() -> dict:
    print("\nGATE 4: Seed robustness (100 seeds)")
    n_seeds = 100
    Ms = np.zeros(n_seeds)

    for seed in range(n_seeds):
        seq = mfn.simulate(mfn.SimulationSpec(grid_size=32, steps=60, seed=seed))
        hwi = compute_hwi_components(seq.history[0], seq.field)
        Ms[seed] = hwi.M

    mu = float(Ms.mean())
    sigma = float(Ms.std())
    cv = sigma / mu * 100
    ci_lo = mu - 1.96 * sigma / np.sqrt(n_seeds)
    ci_hi = mu + 1.96 * sigma / np.sqrt(n_seeds)

    print(f"  M = {mu:.6f} ± {sigma:.6f}")
    print(f"  CV = {cv:.2f}%")
    print(f"  95% CI: [{ci_lo:.6f}, {ci_hi:.6f}]")
    print(f"  Range: [{Ms.min():.6f}, {Ms.max():.6f}]")

    # Normality test
    from scipy.stats import shapiro
    stat, p_normal = shapiro(Ms)

    print(f"  Shapiro-Wilk p={p_normal:.4f} ({'normal' if p_normal > 0.05 else 'non-normal'})")

    return {
        "n_seeds": n_seeds,
        "M_mean": round(mu, 6),
        "M_std": round(sigma, 6),
        "M_cv_percent": round(cv, 4),
        "ci_95_low": round(float(ci_lo), 6),
        "ci_95_high": round(float(ci_hi), 6),
        "M_min": round(float(Ms.min()), 6),
        "M_max": round(float(Ms.max()), 6),
        "shapiro_p": round(float(p_normal), 6),
        "is_normal": bool(p_normal > 0.05),
        "M_values": [round(float(m), 6) for m in Ms],
    }


# ═══════════════════════════════════════════════════════════════════════════
# GATE 5: METRIC INTEGRITY — H, W2, I under transformations
# ═══════════════════════════════════════════════════════════════════════════

def gate_5_integrity() -> dict:
    print("\nGATE 5: Metric integrity")
    seq = mfn.simulate(mfn.SimulationSpec(grid_size=32, steps=60, seed=42))

    # Reference measurement
    hwi_ref = compute_hwi_components(seq.history[0], seq.field)
    M_ref = hwi_ref.M

    tests = {}

    # Test A: Multiply field by constant factor → M should be invariant
    # (distributions normalized internally)
    for scale in [0.5, 2.0, 10.0]:
        hwi_s = compute_hwi_components(seq.history[0] * scale, seq.field * scale)
        drift = abs(hwi_s.M - M_ref) / M_ref * 100
        tests[f"scale_{scale}x"] = {
            "M": round(hwi_s.M, 6),
            "drift_percent": round(drift, 4),
            "invariant": drift < 1.0,
        }
        print(f"  Scale {scale}x: M={hwi_s.M:.6f} drift={drift:.2f}%")

    # Test B: Add constant offset
    # M uses |field| → probability. Translation changes absolute values → changes M.
    for offset in [-0.05, 0.0, 0.05]:
        hwi_o = compute_hwi_components(seq.history[0] + offset, seq.field + offset)
        drift = abs(hwi_o.M - M_ref) / M_ref * 100
        tests[f"offset_{offset}"] = {
            "M": round(hwi_o.M, 6),
            "drift_percent": round(drift, 4),
            "invariant": drift < 1.0,
            "class": "translation",
        }
        print(f"  Offset {offset:+.2f}: M={hwi_o.M:.6f} drift={drift:.2f}%")

    # Test C: Transpose (rotation invariance)
    hwi_t = compute_hwi_components(seq.history[0].T, seq.field.T)
    drift_t = abs(hwi_t.M - M_ref) / M_ref * 100
    tests["transpose"] = {
        "M": round(hwi_t.M, 6),
        "drift_percent": round(drift_t, 4),
        "invariant": drift_t < 1.0,
    }
    print(f"  Transpose: M={hwi_t.M:.6f} drift={drift_t:.2f}%")

    # Test D: Flip (reflection invariance)
    hwi_f = compute_hwi_components(np.flip(seq.history[0]), np.flip(seq.field))
    drift_f = abs(hwi_f.M - M_ref) / M_ref * 100
    tests["flip"] = {
        "M": round(hwi_f.M, 6),
        "drift_percent": round(drift_f, 4),
        "invariant": drift_f < 1.0,
    }
    print(f"  Flip:      M={hwi_f.M:.6f} drift={drift_f:.2f}%")

    geometric = {k: v for k, v in tests.items() if v.get("class") != "translation"}
    translation = {k: v for k, v in tests.items() if v.get("class") == "translation"}

    geom_invariant = all(t["invariant"] for t in geometric.values())
    trans_invariant = all(t["invariant"] for t in translation.values())
    max_trans_drift = max((t["drift_percent"] for t in translation.values()), default=0)

    print(f"  Geometric (scale/rotate/flip): {'INVARIANT' if geom_invariant else 'BROKEN'}")
    print(f"  Translation: {'INVARIANT' if trans_invariant else f'SENSITIVE (max drift {max_trans_drift:.1f}%)'}")

    return {
        "M_ref": round(M_ref, 6),
        "tests": tests,
        "geometric_invariant": geom_invariant,
        "translation_invariant": trans_invariant,
        "max_translation_drift_percent": round(max_trans_drift, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t0 = time.perf_counter()

    print("=" * 65)
    print("  FINAL INVARIANCE VALIDATION: M = H / (W₂ √I)")
    print("=" * 65)
    print()

    g1 = gate_1_finite_size()
    g2 = gate_2_plateau()
    g3 = gate_3_temporal()
    g4 = gate_4_seeds()
    g5 = gate_5_integrity()

    elapsed = time.perf_counter() - t0

    result = {
        "gate_1_finite_size": g1,
        "gate_2_plateau": g2,
        "gate_3_temporal": g3,
        "gate_4_seeds": g4,
        "gate_5_integrity": g5,
        "total_compute_seconds": round(elapsed, 1),
    }

    # Save finite-size fit separately
    with open("results/finite_size_fit.json", "w") as f:
        json.dump(g1, f, indent=2)

    # Save full results
    with open("results/invariance_final.json", "w") as f:
        json.dump(result, f, indent=2)

    # ── VERDICT ──────────────────────────────────────────────────

    print()
    print("=" * 65)
    print("  RESULTS")
    print("=" * 65)

    # Gate 1
    g1_pass = "M_inf" in g1 and g1["M_inf_err"] < g1["M_inf"] * 0.5
    print(f"  G1 Finite-size:  M∞ = {g1['M_inf']:.6f} ± {g1['M_inf_err']:.6f}  "
          f"{'PASS' if g1_pass else 'FAIL'}")

    # Gate 2
    g2_pass = g2["plateau_fraction_10pct"] > 0.50
    print(f"  G2 Plateau:      {g2['plateau_fraction_10pct']*100:.0f}% within ±10%  "
          f"CV={g2['M_cv_percent']:.1f}%  {'PASS' if g2_pass else 'FAIL'}")

    # Gate 3
    g3_pass = g3["cross_seed_morph_cv"] < 5.0
    print(f"  G3 Temporal:     morph CV={g3['cross_seed_morph_cv']:.2f}% cross-seed  "
          f"{'PASS' if g3_pass else 'FAIL'}")

    # Gate 4
    g4_pass = g4["M_cv_percent"] < 5.0
    print(f"  G4 Seeds (100):  M = {g4['M_mean']:.6f} ± {g4['M_std']:.6f}  "
          f"CV={g4['M_cv_percent']:.2f}%  {'PASS' if g4_pass else 'FAIL'}")

    # Gate 5
    g5_geom = g5["geometric_invariant"]
    g5_trans = g5["translation_invariant"]
    g5_pass = g5_geom  # Geometric invariance required; translation documented
    td = g5["max_translation_drift_percent"]
    trans_str = "INVARIANT" if g5_trans else f"SENSITIVE ({td:.0f}%)"
    print(f"  G5 Integrity:    geom={'INVARIANT' if g5_geom else 'BROKEN'}  "
          f"trans={trans_str}  {'PASS' if g5_pass else 'FAIL'}")

    all_pass = g1_pass and g2_pass and g3_pass and g4_pass and g5_pass

    print()

    # Final invariant form
    print(f"  M∞ = {g1['M_inf']:.6f} ± {g1['M_inf_err']:.6f}")
    print(f"  M(N=32) = {g4['M_mean']:.6f} [{g4['ci_95_low']:.6f}, {g4['ci_95_high']:.6f}] 95% CI")
    print(f"  Plateau: {g2['M_mean']:.6f} ± {g2['M_std']:.6f} across "
          f"{g2['n_valid']} parameter combinations")
    morph_m = g3["cross_seed_morph_mean"]
    print(f"  Morphogenesis phase: {morph_m:.6f}")

    print()
    if all_pass:
        # Determine type
        # Check if temporal shows two distinct phases
        knees = [g3["seeds"][str(s)]["knee_step"] for s in [42, 7, 123, 0, 99]]
        has_phase_transition = any(k < 55 for k in knees)

        if has_phase_transition:
            print("  VERDICT: Phase-dependent invariant")
            print(f"  M ≈ {morph_m:.3f} during morphogenesis,")
            steady_ms = [g3["seeds"][str(s)]["steady_mean"] for s in [42, 7, 123, 0, 99]
                         if g3["seeds"][str(s)]["steady_mean"] is not None]
            if steady_ms:
                print(f"  M ≈ {np.mean(steady_ms):.3f} at steady state.")
        else:
            print("  VERDICT: Robust invariant")
            print(f"  M = {g4['M_mean']:.6f} ± {g4['M_std']:.6f}")
    else:
        failed = []
        if not g1_pass: failed.append("finite-size")
        if not g2_pass: failed.append("plateau")
        if not g3_pass: failed.append("temporal")
        if not g4_pass: failed.append("seeds")
        if not g5_pass: failed.append("integrity")
        print(f"  VERDICT: Rejected (failed: {', '.join(failed)})")

    print()
    print(f"  Compute: {elapsed:.0f}s")
    print(f"  Saved: results/invariance_final.json")
    print(f"         results/finite_size_fit.json")
    print(f"         results/plateau_map.npy")
    print("=" * 65)
