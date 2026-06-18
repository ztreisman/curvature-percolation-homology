"""
Full sweep: for a range of q (curvature, via the {3,q} triangulation family),
run percolation + persistent homology at several ring depths (system sizes N)
per q, to get both the q-dependence of loop persistence AND the finite-size
(N-dependence) behavior at fixed q, which is what's needed for a real
finite-size-scaling extrapolation rather than a fixed-N snapshot.
"""
import numpy as np
import time
import json
from mesh_topology import generate_mesh_topology, to_graph, ring_counts_seq
from percolation_topology import relabel_to_int, random_edge_filtration, build_filtered_complex, loopiness_stats
import networkx as nx


def available_sizes(q, cap=35000, max_rings=40, min_n=120):
    """All (rings, N) pairs for this q up to the size cap."""
    out = []
    rc = [0, q]
    cum = 1
    for r in range(1, max_rings + 1):
        if r >= 2:
            rc.append((q - 4) * rc[r - 1] - rc[r - 2])
        cum = 1 + sum(rc[1:r + 1])
        if cum > cap:
            break
        if cum >= min_n:
            out.append((r, cum))
    return out


def pick_sizes(pairs, n_targets=5):
    """Subsample to ~n_targets points, roughly log-spaced in N, always keeping
    the smallest and largest available."""
    if len(pairs) <= n_targets:
        return pairs
    ns = np.array([p[1] for p in pairs], dtype=float)
    log_ns = np.log(ns)
    targets = np.linspace(log_ns[0], log_ns[-1], n_targets)
    idxs = sorted(set(int(np.argmin(np.abs(log_ns - t))) for t in targets))
    return [pairs[i] for i in idxs]


def lambda_growth(q):
    """Larger root of x^2 - (q-4)x + 1 = 0: the asymptotic ring-growth rate,
    a natural curvature/nonamenability parameter (lambda=1 at q=6, flat)."""
    a = q - 4
    disc = a * a - 4
    if disc < 0:
        return 1.0
    return (a + np.sqrt(disc)) / 2


def run_one(q, rings, n_trials=15, seed0=0):
    pairs, tris, rc = generate_mesh_topology(rings=rings, q=q)
    G = to_graph(pairs)
    V, E, F = G.number_of_nodes(), G.number_of_edges(), len(tris)
    euler = V - E + F
    G2, _ = relabel_to_int(G)
    n = G2.number_of_nodes()
    stats = []
    for t in range(n_trials):
        filt = random_edge_filtration(G2, seed=seed0 + t)
        st = build_filtered_complex(G2, filt, max_dim=2)
        st.compute_persistence()
        stats.append(loopiness_stats(st, n))
    per_vertex = np.array([s["total_h1_persistence_per_vertex"] for s in stats])
    return {
        "q": q, "rings": rings, "V": V, "E": E, "F": F, "euler": euler,
        "mean": float(per_vertex.mean()), "std": float(per_vertex.std()),
        "n_trials": n_trials,
    }


if __name__ == "__main__":
    q_values = [6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20]
    results = []
    t_start = time.time()
    for q in q_values:
        sizes = pick_sizes(available_sizes(q), n_targets=5)
        for rings, n_expected in sizes:
            r = run_one(q, rings, n_trials=15)
            r["lambda"] = lambda_growth(q)
            results.append(r)
            print(f"q={q:3d} rings={rings:3d} N={r['V']:6d} euler={r['euler']:2d} "
                  f"mean={r['mean']:.4f} std={r['std']:.4f}  [{time.time()-t_start:.1f}s elapsed]")

    bad = [r for r in results if r["euler"] != 1]
    print(f"\nEuler-characteristic failures: {len(bad)} / {len(results)}")

    with open("sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nTotal time: {time.time()-t_start:.1f}s, {len(results)} data points")
