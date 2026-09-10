"""
Converged bulk MST-weight density mu(q) = lim_{V->inf} E[w_MST]/V, across the
same q values as the original curvature sweep.

This is the "bulk" term isolated in Finding 5 (README.md / Section 6 of the
paper): E[total H1 persistence]/V = mu(q) - b/(4V), where b/V -> 1-1/lambda(q)
is now a proven boundary effect. mu(q) is the remaining open piece. Since MST
weight needs no persistent-homology computation (just Kruskal via
scipy.sparse.csgraph, which is extremely fast), this can be pushed to much
larger N than the H1 sweep, per q, to get well-converged asymptotic estimates
and test whether mu(q) itself obeys a clean closed form in lambda(q).

Usage:
    python mu_q_convergence.py [--workers 24] [--out mu_q_results.json]
"""
import os
from _paths import HERE, RESULTS, FIGURES
import argparse
import json
import time
from multiprocessing import Pool

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import minimum_spanning_tree

from mesh_topology import generate_mesh_topology, to_graph
from percolation_topology import relabel_to_int


def lambda_growth(q):
    a = q - 4
    disc = a * a - 4
    if disc < 0:
        return 1.0
    return (a + np.sqrt(disc)) / 2


def ring_depths_for_q(q, v_cap=3_000_000, max_rings=1000, n_depths=8):
    """Log-spaced ring depths (by resulting V) up to v_cap."""
    rc = [0, q]
    cum = 1
    pairs = []
    for r in range(1, max_rings + 1):
        if r >= 2:
            rc.append((q - 4) * rc[r - 1] - rc[r - 2])
        cum = 1 + sum(rc[1:r + 1])
        if cum > v_cap:
            break
        pairs.append((r, cum))
    if len(pairs) <= n_depths:
        return pairs
    # log-spaced subsample, always keep smallest & largest
    idx_v = np.array([p[1] for p in pairs], dtype=float)
    targets = np.linspace(np.log(idx_v[0]), np.log(idx_v[-1]), n_depths)
    idxs = sorted(set(int(np.argmin(np.abs(np.log(idx_v) - t))) for t in targets))
    return [pairs[i] for i in idxs]


def mst_trial(args):
    q, rings, seed = args
    pairs, tris, rc = generate_mesh_topology(rings=rings, q=q)
    G = to_graph(pairs)
    G2, mapping = relabel_to_int(G)
    V = G2.number_of_nodes()
    edges = np.array(list(G2.edges()))
    rng = np.random.default_rng(seed)
    w = rng.uniform(0, 1, size=len(edges))
    A = sp.coo_matrix((w, (edges[:, 0], edges[:, 1])), shape=(V, V))
    mst = minimum_spanning_tree(A)
    return dict(q=q, rings=rings, seed=seed, V=V, mst_weight=float(mst.sum()))


def model(V, A, B, alpha):
    return A - B / V ** alpha


def main(workers, out_path, n_trials, v_cap):
    q_values = [6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20]
    jobs = []
    depths_by_q = {}
    for q in q_values:
        depths = ring_depths_for_q(q, v_cap=v_cap)
        depths_by_q[q] = depths
        for rings, V in depths:
            for seed in range(n_trials):
                jobs.append((q, rings, seed))
    print(f"{len(jobs)} MST jobs across {len(q_values)} q values, {workers} workers")

    t0 = time.time()
    with Pool(workers) as pool:
        raw = pool.map(mst_trial, jobs)
    print(f"done in {time.time()-t0:.1f}s")

    from scipy.optimize import curve_fit
    results = {}
    for q in q_values:
        rows_by_rings = {}
        for r in raw:
            if r['q'] == q:
                rows_by_rings.setdefault(r['rings'], []).append(r)
        points = []
        for rings, rows in sorted(rows_by_rings.items()):
            V = rows[0]['V']
            vals = np.array([r['mst_weight'] for r in rows]) / V
            points.append((V, vals.mean(), vals.std(), len(vals)))
        points.sort()
        Vs = np.array([p[0] for p in points], dtype=float)
        means = np.array([p[1] for p in points])

        lam = lambda_growth(q)
        if len(Vs) >= 3:
            try:
                popt, pcov = curve_fit(model, Vs, means, p0=[means[-1], 1.0, 0.3],
                                       maxfev=20000)
                A, B, alpha = popt
                perr = np.sqrt(np.diag(pcov))
            except Exception as e:
                A, B, alpha, perr = means[-1], 0.0, 0.0, (0, 0, 0)
        else:
            A, B, alpha, perr = means[-1], 0.0, 0.0, (0, 0, 0)

        results[q] = dict(
            q=q, lam=lam, points=[(float(v), float(m)) for v, m in zip(Vs, means)],
            mu_asymptote=float(A), mu_asymptote_err=float(perr[0]),
            largest_V=float(Vs[-1]), largest_mean=float(means[-1]),
        )
        print(f"q={q:>3}  largest_V={Vs[-1]:>10.0f}  mu(largest)={means[-1]:.5f}  "
              f"mu(asymptote)={A:.5f}+/-{perr[0]:.5f}")

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"saved {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--v-cap", type=int, default=3_000_000)
    ap.add_argument("--out", default=os.path.join(RESULTS, "mu_q_results.json"))
    args = ap.parse_args()
    main(args.workers, args.out, args.n_trials, args.v_cap)
