# Curvature-Filtered Persistent Homology of Percolation Complexes

A study of how negative curvature suppresses loop content in percolation clusters,
and whether that suppression leaves a measurable signature in sparse autoencoders
trained on branching-random-walk embeddings of those clusters.

---

## Context: Brill's data model

Brill (2024, 2025) models a natural data distribution as the cluster process of
critical bond percolation on a high-dimensional lattice.  The key analytical
simplification is that, in high dimensions, random paths almost never
self-intersect, so percolation clusters are essentially trees: they contain
negligible H1 (independent cycles) and can be replaced by a Bethe lattice for
exact computation of thresholds and critical exponents.  Brill's hierarchical
fine-graining algorithm then generates synthetic data by embedding those clusters
into a vector space via a **branching random walk (BRW)**: each node's feature
vector is the cumulative sum of i.i.d. Gaussian steps along the unique path from
the cluster root, so the embedding exactly preserves the tree metric and any
departures from it (loops in the cluster) appear as inconsistencies in the
embedded coordinates.

This project asks two questions in that setting:

1. **Homological question.** How accurately does curvature serve as a substitute
   for high dimension in suppressing cluster loops, and does that suppression
   follow a closed-form law?

2. **SAE question.** When synthetic clusters are BRW-embedded and fed to a sparse
   autoencoder, does the SAE's feature-splitting behavior track the loop content
   of the underlying clusters?

---

## Approach: {3, q} tilings as a curvature dial

The regular triangulation **{3, q}** — q triangles meeting at every vertex — gives
an exact, one-parameter family spanning the loop-rich-to-tree-like spectrum:

| q  | geometry | amenable? | λ(q) |
|----|----------|-----------|------|
| 6  | flat (Euclidean triangular lattice) | yes | 1 |
| 7  | mildly hyperbolic | no | 2.618 |
| 8  | more hyperbolic | no | 3.732 |
| 12 | strongly hyperbolic | no | 8.873 |
| 20 | deep hyperbolic | no | 17.944 |

λ(q) = [(q − 4) + √((q − 4)² − 4)] / 2 is the dominant root of the ring-count
recursion rc[n] = (q − 4)·rc[n − 1] − rc[n − 2]; it is the tiling's asymptotic
per-ring growth rate.  For this specific family, λ(q) = 1 (linear growth) versus
λ(q) > 1 (exponential growth) is exactly the amenable/non-amenable distinction.

Non-amenable hyperbolic graphs are provably in the mean-field universality class
at p_c (Hutchcroft, GAFA 2019), for the same structural reason high-dimensional
Euclidean percolation is: boundary scales with volume, so the "loops are rare"
argument holds globally rather than only asymptotically in dimension.

Bond percolation on each {3, q} lattice is run as a filtered simplicial complex
(each edge assigned an i.i.d. Uniform[0, 1] threshold; 2-simplices added wherever
all three boundary edges are present), and GUDHI extracts persistent H1 across the
full filtration from a single draw of thresholds.

---

## Finding 1 — H1 persistence follows a closed-form scaling law

Using each q's largest available system size, **total H1 persistence per vertex**
(sum of bar lengths in the persistence diagram, divided by N) fits:

```
H1/vertex  =  0.0394  +  0.1387 / λ(q),    R² = 0.9996
```

| q  | N      | H1/vertex        | 1/λ(q) |
|----|--------|-----------------|--------|
| 6  | 4,921  | 0.1789 ± 0.0030 | 1.000  |
| 7  | 29,261 | 0.0910 ± 0.0009 | 0.382  |
| 8  | 31,809 | 0.0755 ± 0.0005 | 0.268  |
| 9  | 30,025 | 0.0676 ± 0.0007 | 0.207  |
| 10 | 14,351 | 0.0630 ± 0.0010 | 0.170  |
| 11 | 29,041 | 0.0594 ± 0.0004 | 0.143  |
| 12 |  6,817 | 0.0578 ± 0.0013 | 0.113  |
| 13 | 10,414 | 0.0552 ± 0.0004 | 0.097  |
| 14 | 15,261 | 0.0535 ± 0.0008 | 0.085  |
| 16 | 29,761 | 0.0516 ± 0.0006 | 0.068  |
| 18 |  3,781 | 0.0500 ± 0.0015 | 0.057  |
| 20 |  5,441 | 0.0488 ± 0.0016 | 0.050  |

The intercept (~0.039) is a curvature-independent floor from purely local loop
content (individual triangles).  The 1/λ term captures everything global curvature
suppresses, decaying at the tiling's own per-ring growth rate — the same quantity
that governs the amenable/non-amenable transition.  This inverts directly: given a
target loop-persistence value, solve for λ and hence q, picking off the curvature
needed to hit any specified distance from the treelike limit.

---

## Finding 2 — Finite-size convergence splits cleanly at the amenability boundary

- **q = 6 (amenable):** H1/vertex drifts upward by > 3 % between the two largest
  sizes tested (N = 1951 → 4921) and has not converged.
- **q ≥ 7 (non-amenable):** Every value converges to within 1–2 % within the first
  few ring depths and stays flat across two orders of magnitude in N.

This is an independent empirical fingerprint of the isoperimetric distinction at
the heart of Hutchcroft's theorem: non-amenable graphs have boundary scaling with
volume, so finite-size corrections decay fast; amenable graphs do not.

*Note:* the q = 6 intercept in Finding 1 is therefore slightly underestimated; the
fitted floor (~0.039) should be read as provisional pending a larger-N Euclidean run.

---

## Finding 3 — BRW cycle inconsistency directly tracks loop content

Each subcritical percolation cluster is BRW-embedded: a BFS spanning tree is
chosen, each edge receives an i.i.d. Gaussian step in ℝ⁶⁴, and node coordinates
are cumulative sums along the unique tree path.  For non-tree (cycle-closing) edges
the implied coordinate difference is generally nonzero; the **cycle inconsistency**
is the RMS norm of these holonomy vectors over all non-tree edges in the cluster.

Running at p ≈ 0.75 p_c for each q (subcritical, finite clusters; 300 clusters per
q, balanced for equal training budget):

| q  | p     | cycle inconsistency |
|----|-------|---------------------|
|  7 | 0.150 | 0.237               |
|  8 | 0.120 | 0.113               |
| 12 | 0.085 | 0.081               |
| 20 | 0.047 | 0.045               |

**Perfectly monotone.** This is a direct embedding-space measurement: BRW is exact
on trees, so tree-like clusters (high q) produce near-zero inconsistency and loopy
clusters (low q) produce larger holonomy.  The metric is zero-assumption — it
requires no homology computation, just the embedding coordinates and the knowledge
of which edges are non-tree.

---

## Finding 4 — SAE feature splitting also tracks loop content (pilot scale)

A sparse autoencoder (ReLU encoder, unit-norm decoder columns, L1 sparsity penalty)
is trained on the pooled BRW node embeddings for each q.  The **feature-splitting
score** is the number of dictionary atoms needed, via greedy argmax cover, to
account for ≥ 90 % of cluster-node memberships — averaged over clusters.

With equal training data (300 clusters/q, dict size 256, 150 epochs, d = 64):

| q  | feature-splitting score |
|----|------------------------|
|  7 | 5.08 ± 1.33            |
|  8 | 5.05 ± 1.51            |
| 12 | 4.93 ± 1.24            |
| 20 | 4.71 ± 1.12            |

**Monotone, in the expected direction:** more loopy clusters (low q) require more
dictionary atoms to cover the same fraction of node memberships — the SAE splits
their "loop-cluster" feature across more atoms.

*Implementation note:* the cluster counts per q must be equalized before training.
Without subsampling, q = 7/8 (more clusters at these small lattice sizes) train a
tighter SAE basis, spuriously lowering their apparent splitting score.  All pilot
results above use 300 clusters per q.

---

## Current pipeline

```
hyperbolic_lattice.py   Mertens-Moore Appendix B vertex labeling for {3,q}
                        (on-the-fly neighbor computation; no pre-allocated
                        exponential-size lattice needed for large ring depths)

generate_clusters.py    Bond percolation at p ≈ 0.75–0.85 p_c; extracts
                        finite clusters by connected components

brw_embedding.py        BRW on BFS spanning tree → ℝ^d node embeddings;
                        computes cycle inconsistency per cluster

sae.py                  PyTorch SAE: ReLU encoder, L1 sparsity, decoder
                        column renormalization after each step

feature_splitting.py    Greedy argmax set-cover score per cluster

sae_experiment.py       Local driver (FULL / FAST configs); see --fast for
                        a smoke-test run in < 10 s

campaign/               GPU-aware runner for lambda01; checkpoints per q so
                        interrupted runs resume; saves SAE weights (.pt)
```

Pilot runs complete locally on CPU in ~10 s at small ring depths.

---

## Planned next steps

### Immediate (campaign on lambda01)
Run `campaign/run_campaign.py` with GPU:
- Ring depths 7–8 (N ~ 28,000–47,000 nodes per lattice)
- BRW dimension d = 128, SAE dict size = 1024, 300 training epochs
- 2,000 balanced clusters per q
- Saves per-q SAE weights for atom-level analysis

### Near-term experiments
- **Vary p relative to p_c.** The pilot uses p ≈ 0.75–0.85 p_c (subcritical,
  finite clusters).  Sweeping p toward p_c and using max-cluster rather than
  all-cluster statistics would probe the critical regime directly.
- **Vary embedding dimension d.** The current pilot uses d = 64.  Checking whether
  feature-splitting trends are stable across d = 32, 64, 128, 256 would establish
  that the result is not an artifact of ambient dimension relative to cluster size.
- **Canonical tree vs. BFS tree.** The Mertens-Moore labeling provides the lattice's
  canonical spanning tree; using it instead of the BFS tree of the percolation
  cluster would test whether cycle inconsistency is measuring cluster topology or
  the mismatch between the cluster and the ambient lattice's tree structure.
- **Absorption and other SAE pathologies.** Feature splitting is one failure mode;
  absorption (a single atom monopolizing a large fraction of argmax assignments) is
  another.  Both could be tabulated across the curvature sweep.
- **Correlation with H1 persistence.** The pilot shows both cycle inconsistency and
  feature splitting are monotone in q; plotting feature splitting directly against
  the measured H1/vertex from the sweep (Finding 1) would test whether the
  relationship is quantitatively linear rather than just ordinal.

### Longer-term
- **Non-tree topology in the embedding.** BRW on the BFS tree destroys cluster loops;
  an embedding that preserves them (e.g., spectral embedding or a spring model that
  includes non-tree edges as constraints) would let the SAE see the loops directly
  and could sharpen the feature-splitting signal.
- **Different SAE architectures.** TopK-SAE (fixed activation count k rather than
  L1 penalty) removes the confound of different effective sparsity levels across
  training runs and may give cleaner feature-splitting estimates.
- **Extend curvature sweep.** Findings 1 and 2 cover q = 6 through 20; q ≥ 7 already
  saturates near the floor.  Interpolating between integer q (using a mixed tiling or
  a continuous family) would fill in the transition region and better constrain the
  fitted intercept.

---

## References

Brill, A. (2024). Neural scaling laws rooted in the data distribution. arXiv:2412.07942.

Brill, A. (2025). Representation learning on a random lattice. arXiv:2504.20197.

Hutchcroft, T. (2019). Percolation on hyperbolic graphs. *GAFA*, 29, 766–810. arXiv:1804.10191.

Mertens, S., & Moore, C. (2017). Percolation thresholds in hyperbolic lattices. *Phys. Rev. E*, 96, 042116. arXiv:1708.05876.

Bobrowski, O., & Skraba, P. (2020). Homological percolation and the Euler characteristic. *Phys. Rev. E*, 101, 032304. arXiv:1910.10146.
