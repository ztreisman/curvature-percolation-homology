"""
Feature-splitting score for a trained SAE.

For each percolation cluster C, we ask: how many SAE dictionary atoms are
needed to jointly "cover" ≥ threshold of C's nodes, where atom j covers a
node if j = argmax of that node's hidden activations?

A greedy set-cover finds this minimum count.  Averaging over clusters gives
the feature-splitting score for a given q value.

Intuition: a tree-like cluster (high q, near-Bethe) should be representable
by fewer distinct SAE atoms than a loop-rich cluster (low q, Euclidean),
because its node-embedding distribution is simpler (fewer holonomy-induced
"directions" in feature space).  If feature splitting tracks loop content,
the score should correlate with H1 persistence per vertex.
"""
import numpy as np


def greedy_cover(H_c: np.ndarray, threshold: float = 0.9) -> int:
    """
    Minimum number of atoms (argmax-winner atoms) to cover >= threshold of
    the n_c rows of H_c (shape n_c × dict_size).

    Returns 0 if H_c is empty; returns dict_size if threshold is unreachable
    (e.g., > threshold of rows are all-zero).
    """
    n_c = H_c.shape[0]
    if n_c == 0:
        return 0

    # For each node: which atom has the highest activation?
    # Nodes with all-zero activations get assigned atom -1 (uncoverable).
    max_acts = H_c.max(axis=1)
    coverable = max_acts > 0
    winners = np.where(coverable, H_c.argmax(axis=1), -1)

    n_coverable = coverable.sum()
    target = int(np.ceil(threshold * n_c))
    if n_coverable < target:
        # More than (1-threshold) fraction is permanently uncoverable.
        return H_c.shape[1]  # sentinel: "needs all atoms"

    covered = np.zeros(n_c, dtype=bool)
    n_atoms = 0

    while covered.sum() < target:
        # Count uncovered nodes per atom (exclude atom -1)
        uncov_winners = winners[~covered]
        if len(uncov_winners) == 0 or (uncov_winners == -1).all():
            break
        # Fast argmax via bincount (ignoring -1 sentinel)
        valid = uncov_winners[uncov_winners >= 0]
        counts = np.bincount(valid, minlength=H_c.shape[1])
        best = int(counts.argmax())
        covered |= (winners == best)
        n_atoms += 1

    return n_atoms


def feature_splitting_scores(H: np.ndarray,
                             labels: np.ndarray,
                             n_clusters: int,
                             threshold: float = 0.9) -> np.ndarray:
    """
    Compute the greedy-cover feature-splitting score for each cluster.

    Parameters
    ----------
    H         : (N_total, dict_size)  SAE hidden activations for all nodes.
    labels    : (N_total,)            Cluster index per node (0-indexed).
    n_clusters: int                   Total number of clusters.
    threshold : float                 Coverage threshold (default 0.9).

    Returns
    -------
    scores : (n_clusters,)  Feature-splitting score per cluster.
    """
    scores = np.empty(n_clusters, dtype=float)
    for idx in range(n_clusters):
        H_c = H[labels == idx]
        scores[idx] = greedy_cover(H_c, threshold=threshold)
    return scores
