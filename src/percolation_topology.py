"""
Percolation -> persistent homology pipeline.

Core idea: treat each graph edge as carrying a uniform random "occupation
threshold" u_e ~ U[0,1]. Build the clique complex of the graph with edge
filtration values u_e (vertices at filtration 0). Expanding to dimension 2
means a triangle enters the filtration at max(u_e) over its three edges --
exactly bond percolation's "all three edges present" condition. This gives
the *entire* percolation process (all p in [0,1]) as a single filtration,
from which we extract persistent homology, Betti curves vs p, and summary
"loopiness" statistics.

This only requires a graph (vertices + edges), not embedded coordinates,
PROVIDED the graph's only 3-cliques are genuine 2-cells of the underlying
triangulation (true for {3,q} regular tilings away from small pathological
cases) -- non-face triangles would otherwise be spuriously filled in.
"""
import numpy as np
import networkx as nx
import gudhi


def relabel_to_int(G):
    """GUDHI simplex labels must be ints; map arbitrary node labels to 0..n-1."""
    nodes = list(G.nodes())
    mapping = {n: i for i, n in enumerate(nodes)}
    return nx.relabel_nodes(G, mapping), mapping


def random_edge_filtration(G, seed=None):
    """Assign i.i.d. Uniform[0,1] filtration values to each edge.
    G's nodes must already be ints (see relabel_to_int)."""
    rng = np.random.default_rng(seed)
    filt = {}
    for u, v in G.edges():
        filt[(u, v)] = float(rng.uniform(0, 1))
    return filt


def build_filtered_complex(G, edge_filt, max_dim=2):
    """Build a GUDHI SimplexTree: vertices at filtration 0, edges at their
    random threshold, then expand to triangles (and optionally higher) so
    a k-simplex enters at the max filtration value of its edges."""
    st = gudhi.SimplexTree()
    for v in G.nodes():
        st.insert([int(v)], filtration=0.0)
    for (u, v), f in edge_filt.items():
        st.insert([int(u), int(v)], filtration=f)
    st.expansion(max_dim)  # fills in triangles etc. from the clique complex
    st.make_filtration_non_decreasing()
    return st


def compute_persistence(st):
    st.compute_persistence()
    diag = st.persistence()
    return diag  # list of (dim, (birth, death))


def betti_curve(st, dim, ps):
    """Betti_dim(p) for a grid of p values, via gudhi's persistence_intervals."""
    intervals = st.persistence_intervals_in_dimension(dim)
    betti = np.zeros(len(ps))
    for birth, death in intervals:
        if np.isinf(death):
            death = np.inf
        alive = (ps >= birth) & (ps < death)
        betti += alive.astype(int)
    return betti


def loopiness_stats(st, n_vertices):
    """Summary scalar: total finite H1 persistence (sum of death-birth over
    bars), normalized by vertex count, as a size-independent 'how loopy'
    measure. Also report number of H1 bars and the max bar length."""
    h1 = st.persistence_intervals_in_dimension(1)
    finite = [(b, d) for b, d in h1 if not np.isinf(d)]
    total = sum(d - b for b, d in finite)
    n_bars = len(h1)
    max_bar = max([d - b for b, d in finite], default=0.0)
    return {
        "n_h1_bars": n_bars,
        "n_finite_h1_bars": len(finite),
        "total_h1_persistence": total,
        "total_h1_persistence_per_vertex": total / n_vertices,
        "max_h1_bar_length": max_bar,
    }
