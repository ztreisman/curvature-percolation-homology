"""
Directly verify x*(q) = (q-3-lambda(q))/(lambda(q)+1) -- the predicted
limiting fraction of v-type (2-parent) vertices per ring -- against the
actual graph, by counting, for each ring-n vertex, how many of its
neighbors lie in ring n-1 (1 = e-type, 2 = v-type).
"""
import os
from _paths import HERE, RESULTS, FIGURES
import numpy as np
import networkx as nx
from mesh_topology import generate_mesh_topology, to_graph


def lam(q):
    a = q - 4
    disc = a * a - 4
    return (a + np.sqrt(disc)) / 2 if disc >= 0 else 1.0


def xstar_formula(q):
    L = lam(q)
    return (q - 3 - L) / (L + 1)


def measure_x(q, rings):
    pairs, tris, rc = generate_mesh_topology(rings=rings, q=q)
    G = to_graph(pairs)
    dist = nx.single_source_shortest_path_length(G, 0)
    ring_of = dist
    # last full ring only (ring `rings`), to see the converged fraction
    last_ring_nodes = [v for v, r in ring_of.items() if r == rings]
    prev_ring_nodes = set(v for v, r in ring_of.items() if r == rings - 1)
    v_count = 0
    for v in last_ring_nodes:
        n_parents = sum(1 for u in G.neighbors(v) if u in prev_ring_nodes)
        if n_parents == 2:
            v_count += 1
        elif n_parents not in (1, 2):
            raise RuntimeError(f"vertex {v} has {n_parents} parent-ring neighbors")
    return v_count / len(last_ring_nodes)


for q in [7, 8, 9, 12, 16, 20]:
    pred = xstar_formula(q)
    # cap total vertex count at ~200k: rings ~ log(200000)/log(lambda)
    rings = max(4, min(10, int(np.log(200000) / np.log(lam(q)))))
    meas = measure_x(q, rings=rings)
    print(f"q={q:3d}  rings={rings}  predicted x*={pred:.4f}  measured={meas:.4f}  diff={meas-pred:+.4f}")
