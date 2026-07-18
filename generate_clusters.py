"""
Bond percolation cluster extraction from {3,Q} hyperbolic tilings.

Each call to `generate_clusters` generates a lattice of specified ring depth,
runs `n_trials` independent bond percolation realizations at occupation
probability `p`, and returns all connected components (clusters) that fall
within the requested size window.

p_c (bond) reference values from Mertens & Moore Table I:
  Q=7:  p_c ≈ 0.1994   Q=8:  p_c ≈ 0.1602
  Q=9:  p_c ≈ 0.1356   Q=12: p_c ≈ 0.10 (extrapolated near Bethe 1/11)
  Q=20: p_c ≈ 0.055    (extrapolated near Bethe 1/19)
"""
import numpy as np
import networkx as nx
from mesh_topology import generate_mesh_topology, to_graph
from percolation_topology import relabel_to_int

# Default p values: ≈ 0.75 * p_c for each Q, giving subcritical percolation
# with a useful distribution of finite clusters.
DEFAULT_P = {
    6:  0.26,   # flat triangular lattice (p_c = 0.347)
    7:  0.15,   # {3,7}  (p_c ≈ 0.199)
    8:  0.12,   # {3,8}  (p_c ≈ 0.160)
    9:  0.10,   # {3,9}  (p_c ≈ 0.136)
    12: 0.075,  # {3,12} (p_c ≈ 0.100)
    20: 0.040,  # {3,20} (p_c ≈ 0.055)
}


def _bond_percolation(G, p, rng):
    """Occupy each edge with probability p; return occupied subgraph."""
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    for u, v in G.edges():
        if rng.uniform() < p:
            H.add_edge(u, v)
    return H


def get_lattice(q, rings):
    """Build {3,Q} tiling with `rings` rings; return integer-relabeled graph."""
    pairs, tris, rc = generate_mesh_topology(rings=rings, q=q)
    G = to_graph(pairs)
    G, _ = relabel_to_int(G)
    return G


def generate_clusters(q, rings, p=None, n_trials=50,
                      min_size=5, max_size=None, seed=0):
    """
    Extract bond percolation clusters from the {3,Q} tiling.

    Parameters
    ----------
    q         : int   Tiling parameter.
    rings     : int   Ring depth (controls lattice size N).
    p         : float Bond occupation probability; defaults to DEFAULT_P[q].
    n_trials  : int   Independent percolation realizations.
    min_size  : int   Minimum cluster size to include.
    max_size  : int   Maximum cluster size (None = no upper limit).
    seed      : int   RNG seed.

    Returns
    -------
    clusters : list[nx.Graph]  All extracted clusters across all trials.
    meta     : dict            {q, rings, p, N, n_trials, n_clusters}
    """
    if p is None:
        p = DEFAULT_P.get(q, 0.15)

    G = get_lattice(q, rings)
    N = G.number_of_nodes()
    rng = np.random.default_rng(seed)

    clusters = []
    for _ in range(n_trials):
        H = _bond_percolation(G, p, rng)
        for nodes in nx.connected_components(H):
            n = len(nodes)
            if n < min_size:
                continue
            if max_size is not None and n > max_size:
                continue
            clusters.append(H.subgraph(nodes).copy())

    meta = dict(q=q, rings=rings, p=p, N=N,
                n_trials=n_trials, n_clusters=len(clusters))
    return clusters, meta
