"""
Branching-random-walk (BRW) embedding of a percolation cluster.

Each cluster is rooted at an arbitrary node; a BFS spanning tree is built
and each tree edge is assigned an i.i.d. Gaussian step in R^d.  The
embedding of node v is the cumulative sum of steps along the unique
spanning-tree path from root to v.

Note: because the cluster may have cycles, the spanning tree discards some
edges.  The per-cluster "cycle inconsistency" (sum of step vectors around
each fundamental cycle) is non-zero in general -- this is the analogue of
Brill's lattice embedding acquiring a non-trivial holonomy around loops.
We make that inconsistency explicit as a second output so callers can study
how much each cluster's embedding "conflicts with itself."
"""
import numpy as np
import networkx as nx


def brw_embed_cluster(G_cluster, d=64, sigma=1.0, rng=None):
    """
    BRW-embed all nodes of cluster G_cluster into R^d.

    Parameters
    ----------
    G_cluster : nx.Graph
        Connected subgraph (the percolation cluster).
    d : int
        Embedding dimension.
    sigma : float
        Step standard deviation; each edge step ~ N(0, sigma^2/d * I_d).
    rng : np.random.Generator, optional

    Returns
    -------
    embeddings : dict {node: np.ndarray shape (d,)}
    cycle_inconsistency : float
        RMS norm of holonomy vectors over all fundamental cycles
        (non-tree edges).  Zero for trees; positive for cyclic clusters.
    """
    if rng is None:
        rng = np.random.default_rng()

    nodes = list(G_cluster.nodes())
    root = nodes[0]

    # BFS spanning tree
    tree = nx.bfs_tree(G_cluster, root)

    # Assign a random step to each tree edge (parent -> child direction)
    step = {}  # (parent, child) -> R^d vector
    for parent, child in tree.edges():
        v = rng.standard_normal(d) * (sigma / np.sqrt(d))
        step[(parent, child)] = v
        step[(child, parent)] = -v   # keep antisymmetry for cycle calc

    # Accumulate embeddings along BFS tree paths
    embeddings = {root: np.zeros(d)}
    for parent, child in nx.bfs_edges(G_cluster, root):
        embeddings[child] = embeddings[parent] + step[(parent, child)]

    # Compute cycle inconsistency: for each non-tree edge (u,v),
    # the holonomy = embedding[v] - embedding[u] - step_uv_if_tree_edge
    # Since (u,v) is NOT a tree edge we can't look it up in `step`;
    # instead we assign a fresh Gaussian step and compare to the already
    # determined embedding difference.
    tree_edge_set = set(tree.edges()) | {(v, u) for u, v in tree.edges()}
    nontree_edges = [(u, v) for u, v in G_cluster.edges()
                     if (u, v) not in tree_edge_set]

    holonomy_sq = 0.0
    for u, v in nontree_edges:
        # Assign a Gaussian step for this non-tree edge
        z = rng.standard_normal(d) * (sigma / np.sqrt(d))
        # Holonomy around the cycle this edge closes:
        # embed[v] = embed[u] + z  should hold if the graph were a tree.
        # The inconsistency is (embed[v] - embed[u]) - z.
        h = embeddings[v] - embeddings[u] - z
        holonomy_sq += float(np.dot(h, h))

    n_nontree = len(nontree_edges)
    cycle_inconsistency = np.sqrt(holonomy_sq / n_nontree) if n_nontree > 0 else 0.0

    return embeddings, cycle_inconsistency


def embed_clusters(clusters, d=64, sigma=1.0, seed=0):
    """
    Embed a list of cluster graphs.  Returns:
      X          : np.ndarray (N_total_nodes, d)  -- stacked embeddings
      labels     : np.ndarray (N_total_nodes,)    -- cluster index per node
      sizes      : list[int]                       -- cluster sizes
      inconsistencies : list[float]               -- cycle inconsistency per cluster
    """
    rng = np.random.default_rng(seed)
    X_list, label_list, sizes, inconsistencies = [], [], [], []

    for idx, G in enumerate(clusters):
        n = G.number_of_nodes()
        emb, incon = brw_embed_cluster(G, d=d, sigma=sigma, rng=rng)
        X_list.append(np.stack([emb[v] for v in G.nodes()]))
        label_list.extend([idx] * n)
        sizes.append(n)
        inconsistencies.append(incon)

    X = np.vstack(X_list)
    labels = np.array(label_list)
    return X, labels, sizes, inconsistencies
