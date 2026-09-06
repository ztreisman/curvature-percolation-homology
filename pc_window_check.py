"""
Robustness check: does the H1/vertex ~ 1/lambda(q) scaling law survive
restriction to a window around p_c, or is it dominated by the supercritical
bulk regime?

Total H1 persistence (as used in full_sweep.py / Finding 1) integrates
beta_1(p) over the ENTIRE filtration p in [0,1]. That mixes together the
subcritical, critical, and supercritical regimes, and the loop-density peak
is empirically well above p_c (~0.7 for the flat lattice, vs p_c~0.347), so
most of that integral's mass likely comes from the dense supercritical
regime, not from the critical window Hutchcroft's mean-field theorem and
Mertens-Moore's tau=5/2 result are actually about.

This script computes, for each q in the same 12-point sweep as Finding 1, a
SECOND statistic: total H1 persistence restricted to a multiplicative window
around p_c(q), namely [p_c/2, min(2*p_c, 1)]. For each percolation
realization, the windowed contribution of an H1 bar (birth, death) is the
length of its overlap with that window, summed over all bars and divided by
N. Both statistics are then fit to the same intercept + slope/lambda(q) form
used for Finding 1, and the R^2 values are compared.

p_c(q) sources:
  q=6        : exact, triangular lattice, p_c = 2*sin(pi/18)
  q=7,8,9    : Mertens & Moore (arXiv:1708.05876) Table I, P=3 row, six
               decimal places: 0.199351, 0.160156, 0.135565 (bond)
  q=10..20   : not tabulated in Mertens & Moore (their grid stops at Q=9).
               Extrapolated from the same three exact points via
                   p_c(q) = [1 + c/(q-1)^3] / (q-1)
               which matches the P=3 case of their asymptotic form (eq. 19),
               p_c({3,Q}) = 1/(Q-1) + c'/mu_Q^3 for some mu_Q = O(Q). Fitting
               c from q=7,8,9 gives c ~= 42.4 (consistent across all three to
               within 2%). This is an approximation, not a measured value;
               it is used only to place a *window*, not for high-precision
               fitting, so the multiplicative width of the window (factor of
               2 either side) is chosen to be generous relative to this
               extrapolation's uncertainty.
"""
import numpy as np
import json
import time
from mesh_topology import generate_mesh_topology, to_graph
from percolation_topology import relabel_to_int, random_edge_filtration, build_filtered_complex
from full_sweep import available_sizes, lambda_growth


def pc_of_q(q):
    if q == 6:
        return 2 * np.sin(np.pi / 18)  # exact, triangular lattice
    exact = {7: 0.199351, 8: 0.160156, 9: 0.135565}  # Mertens & Moore Table I
    if q in exact:
        return exact[q]
    c = 42.4  # fit to the three exact points above, see module docstring
    return (1 + c / (q - 1) ** 3) / (q - 1)


def windowed_and_total_stats(st, n_vertices, lo, hi):
    """Total H1 persistence per vertex, and the same restricted to the
    overlap of each bar with [lo, hi], per vertex."""
    h1 = st.persistence_intervals_in_dimension(1)
    total = 0.0
    windowed = 0.0
    for b, d in h1:
        if np.isinf(d):
            continue  # none expected (the full disk at p=1 is simply connected)
        total += d - b
        overlap = max(0.0, min(d, hi) - max(b, lo))
        windowed += overlap
    return total / n_vertices, windowed / n_vertices


def run_one(q, rings, n_trials=15, seed0=0):
    pairs, tris, rc = generate_mesh_topology(rings=rings, q=q)
    G = to_graph(pairs)
    V = G.number_of_nodes()
    G2, _ = relabel_to_int(G)
    n = G2.number_of_nodes()

    pc = pc_of_q(q)
    lo, hi = pc / 2, min(2 * pc, 1.0)

    totals, windows = [], []
    for t in range(n_trials):
        filt = random_edge_filtration(G2, seed=seed0 + t)
        st = build_filtered_complex(G2, filt, max_dim=2)
        st.compute_persistence()
        tot, win = windowed_and_total_stats(st, n, lo, hi)
        totals.append(tot)
        windows.append(win)

    totals = np.array(totals)
    windows = np.array(windows)
    return {
        "q": q, "rings": rings, "V": V, "pc": pc, "window": [lo, hi],
        "n_trials": n_trials,
        "total_mean": float(totals.mean()), "total_std": float(totals.std()),
        "window_mean": float(windows.mean()), "window_std": float(windows.std()),
        "window_frac_of_total": float(windows.mean() / totals.mean()),
    }


def fit_1_over_lambda(qs, lambdas, ys):
    x = 1.0 / np.array(lambdas)
    y = np.array(ys)
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    intercept, slope = coef
    pred = intercept + slope * x
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return intercept, slope, r2


if __name__ == "__main__":
    q_values = [6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20]
    results = []
    t_start = time.time()
    for q in q_values:
        sizes = available_sizes(q, cap=35000)
        rings, n_expected = sizes[-1]  # largest available, as in Finding 1
        r = run_one(q, rings, n_trials=15)
        r["lambda"] = lambda_growth(q)
        results.append(r)
        print(f"q={q:3d} rings={rings:2d} N={r['V']:6d} pc={r['pc']:.4f} "
              f"window=[{r['window'][0]:.4f},{r['window'][1]:.4f}]  "
              f"total={r['total_mean']:.4f}+/-{r['total_std']:.4f}  "
              f"windowed={r['window_mean']:.4f}+/-{r['window_std']:.4f}  "
              f"(window/total={r['window_frac_of_total']:.3f})  "
              f"[{time.time()-t_start:.1f}s elapsed]")

    qs = [r["q"] for r in results]
    lambdas = [r["lambda"] for r in results]
    total_i, total_s, total_r2 = fit_1_over_lambda(qs, lambdas, [r["total_mean"] for r in results])
    win_i, win_s, win_r2 = fit_1_over_lambda(qs, lambdas, [r["window_mean"] for r in results])

    print("\n--- Fits: y = intercept + slope / lambda(q) ---")
    print(f"Full-range   [0,1]      : y = {total_i:.4f} + {total_s:.4f}/lambda(q)   R^2={total_r2:.4f}")
    print(f"Windowed near p_c       : y = {win_i:.4f} + {win_s:.4f}/lambda(q)   R^2={win_r2:.4f}")

    out = {
        "results": results,
        "fit_full": {"intercept": total_i, "slope": total_s, "r2": total_r2},
        "fit_windowed": {"intercept": win_i, "slope": win_s, "r2": win_r2},
    }
    with open("pc_window_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nTotal time: {time.time()-t_start:.1f}s. Saved pc_window_results.json")
