"""
Extended finite-size convergence check for q=6 (the flat, amenable control).

full_sweep.py's available_sizes() capped q=6 at rings<=40 (N=4921) purely via
an artificial max_rings=40 default -- q=6's ring counts grow linearly (lambda=1),
so unlike the hyperbolic q's, reaching large N takes many more rings, not more
vertices per ring. The original sweep found q=6 still drifting upward by >3%
between its two largest sizes (N=1951 -> 4921) and never established whether it
converges. This script pushes q=6 alone to much larger N (up to ~3M vertices) to
settle that, reusing the exact same pipeline (mesh_topology + percolation_topology)
so the results are directly comparable to sweep_results.json.

Parallelized at the level of individual (ring, trial) percolation realizations
via multiprocessing, since each realization is independent and this machine has
many cores.

Usage:
    python q6_convergence.py [--workers 24] [--out q6_convergence_results.json]
"""
import argparse
import json
import time
from multiprocessing import Pool

import numpy as np

from mesh_topology import generate_mesh_topology, to_graph
from percolation_topology import (
    relabel_to_int, random_edge_filtration, build_filtered_complex, loopiness_stats,
)

Q = 6

# (rings, n_trials) -- extends sweep_results.json's q=6 series (rings up to 40)
# to much larger N. Trial counts taper off at the largest sizes since variance
# is expected to shrink with N; cost also grows there.
NEW_DEPTHS = [
    (63, 15),
    (100, 15),
    (160, 15),
    (250, 15),
    (400, 15),
    (630, 12),
    (1000, 8),
]


def run_one_trial(args):
    rings, seed = args
    t0 = time.time()
    pairs, tris, rc = generate_mesh_topology(rings=rings, q=Q)
    G = to_graph(pairs)
    V, E, F = G.number_of_nodes(), G.number_of_edges(), len(tris)
    euler = V - E + F
    G2, _ = relabel_to_int(G)
    filt = random_edge_filtration(G2, seed=seed)
    st = build_filtered_complex(G2, filt, max_dim=2)
    st.compute_persistence()
    stats = loopiness_stats(st, V)
    return dict(rings=rings, seed=seed, V=V, E=E, F=F, euler=euler,
                per_vertex=stats["total_h1_persistence_per_vertex"],
                elapsed=time.time() - t0)


def main(workers, out_path):
    jobs = [(rings, seed) for rings, n_trials in NEW_DEPTHS for seed in range(n_trials)]
    print(f"{len(jobs)} jobs across {len(NEW_DEPTHS)} ring depths, {workers} workers")

    t_start = time.time()
    with Pool(workers) as pool:
        raw = pool.map(run_one_trial, jobs)
    print(f"All jobs done in {time.time() - t_start:.1f}s")

    results = []
    for rings, n_trials in NEW_DEPTHS:
        rows = [r for r in raw if r["rings"] == rings]
        per_vertex = np.array([r["per_vertex"] for r in rows])
        results.append(dict(
            q=Q, rings=rings, V=rows[0]["V"], E=rows[0]["E"], F=rows[0]["F"],
            euler=rows[0]["euler"],
            mean=float(per_vertex.mean()), std=float(per_vertex.std()),
            n_trials=len(rows), lambda_growth=1.0,
            mean_trial_time=float(np.mean([r["elapsed"] for r in rows])),
        ))
        print(f"rings={rings:5d} V={rows[0]['V']:8d} euler={rows[0]['euler']:2d} "
              f"mean={results[-1]['mean']:.5f} std={results[-1]['std']:.5f} "
              f"n_trials={len(rows)} mean_trial_time={results[-1]['mean_trial_time']:.1f}s")

    bad = [r for r in results if r["euler"] != 1]
    print(f"\nEuler-characteristic failures: {len(bad)} / {len(results)}")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default="q6_convergence_results.json")
    args = ap.parse_args()
    main(args.workers, args.out)
