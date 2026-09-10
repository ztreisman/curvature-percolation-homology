# Paper outline: Topological and vector-space embedding of percolated subsets of exponentially growing graphs

Scope decision (2026-09-02): this is its own paper. The geometric/Hessian embedding-capacity
thread (`r(q,d)`, spectral collapse) is a separate paper and not addressed here — see the
git history of this file for the earlier combined outline if needed.

## What changes from the existing draft

`curvature-percolation-homology.tex` / `proposal.tex` already contain most of Part A below
(H1 persistence scaling), well-written, and should be reused close to verbatim — this is a
reframing and extension, not a rewrite from scratch. Two concrete changes:

1. **Reframe**: lead with the embedding question, not with Brill. Brill's data model moves
   from the paper's premise to §6 (an application/discussion section). The current
   abstract/§1 open with "PrincInt's model treats..." — that needs to become the *second*
   move, not the first.
2. **Extend**: the current `.tex` §"Status, and the sketch that remains" describes the
   BRW-embedding + SAE feature-splitting experiment as *not yet executed*. It has since
   been run — README's Findings 3 and 4 report completed results (cycle inconsistency
   monotone in `q`; SAE feature-splitting score monotone in `q` and correlated with both
   `H1` and cycle inconsistency). This becomes new §5 (Part B), replacing that sketch
   paragraph, and changes the framing of the whole paper from "here's a proposal plus a
   sketch of the next step" to "here are two completed, connected measurements."

The throughline that justifies putting A and B in one paper: A measures loop content on
the abstract percolation complex (an intrinsic, embedding-independent quantity); B measures
what happens to that loop content once the cluster is actually embedded into a vector space
and a model is trained on it. That's precisely the "combinatorial embedding vs. vector
embedding" distinction the current draft's Status section already flags as an open question
worth keeping separate — now it's the paper's own two-part structure instead of an
unresolved caveat.

---

## Title

Something like: "How curvature suppresses loops: percolated subsets of hyperbolic tilings,
their homology, and their learned embeddings" — or keep it tighter: "Curvature-tuned
persistent homology and embedding of percolation complexes on `{3,q}` tilings."

Avoid "extending Brill's data model" in the title or subtitle this time — put it in the
abstract's last sentence instead (see below).

## Abstract (content to hit, in order)

1. General framing: a percolation cluster on an exponentially growing graph is itself a
   random subgraph with its own growth/loop statistics; how much of it is representable by
   a tree (i.e., embeddable with zero topological loss) is a question with a precise
   answer, not just an asymptotic one.
2. The `{3,q}` tiling family as an exact curvature dial spanning amenable (`q=6`) to deeply
   hyperbolic (`q=20`), with `λ(q)` as the single growth-rate parameter.
3. Result A: total `H1` persistence per vertex of the bond-percolation complex follows a
   closed-form law in `1/λ(q)`, derived (not just fit) via an exact MST/boundary
   decomposition; finite-size convergence itself splits at the amenable/nonamenable
   boundary.
4. Result B: when percolation clusters are embedded into a vector space via branching
   random walk, the same loop suppression is directly visible in embedding-space holonomy,
   and propagates downstream into a trained sparse autoencoder's feature-splitting
   behavior — a measured link between a topological property of the data-generating
   process and a specific interpretability pathology.
5. Close with the application: this gives an exact, finite, curvature-controlled
   substitute for the "high-dimension implies tree-like" argument used in percolation-based
   data models for interpretability research (Brill 2024, 2025), letting that assumption's
   accuracy be dialed and measured rather than asserted asymptotically.

## 1. Introduction

- Open with the embedding question in general: a percolation cluster is a random
  tree-like-or-not subgraph; representing it faithfully — combinatorially, as a filtered
  complex, or concretely, as vectors in `R^d` — costs something exactly when the cluster
  has loops. Motivate why this matters beyond percolation theory itself: any pipeline that
  embeds hierarchical/branching data into a vector space (this paper's BRW case; more
  generally, any tree- or DAG-structured data fed to a flat encoder) faces the same cost
  when the data isn't perfectly tree-shaped.
- Introduce curvature as the tunable knob: the `{3,q}` family lets this cost be measured
  as an explicit, continuous(-ish) function of one parameter rather than only observed
  qualitatively.
- State the paper's two results (A: intrinsic/topological, B: extrinsic/learned) and that
  they agree — same suppression mechanism, measured two different ways.
- One paragraph at the end pointing forward to §6: this has a direct payoff for
  percolation-based synthetic data models, which currently justify their core
  approximation via an asymptotic high-dimension argument; this paper's `λ(q)` is a
  finite-dimensional, exactly computable stand-in.

## 2. Persistent homology as the diagnostic
*(reuse existing `.tex` §2 near-verbatim)*

- Bond percolation as a single filtration; clique complex construction; GUDHI extracts
  `β0(p), β1(p)`.
- Realization-to-realization averaging (12 draws/lattice).
- Validation: clique complex = true face list, no spurious triangles.
- Keep the connection to Bobrowski & Skraba's homological-percolation work.

## 3. An exact curvature knob: percolation on `{3,q}` tilings
*(reuse existing `.tex` §3, trimmed)*

- `{3,q}` family, `λ(q)`, amenable/nonamenable dichotomy for this family specifically.
- Hutchcroft's theorem: mean-field exponents on nonamenable Gromov-hyperbolic graphs.
- Frame this section's payoff as: curvature gives an *exact*, finite mechanism for the
  same tree-like-at-infinity phenomenon that "high dimension" gives only asymptotically —
  stated here as a general fact about percolation, with the data-model application held
  back for §6.

## 4. Part A — Topological loop content: results and derivation
*(reuse existing `.tex` §4–§"Deriving the scaling law" i.e. current §5 and its subsections,
largely as-is; renumber)*

### 4.1 Method and validation
- Ring-growth construction, Euler characteristic check, flat-lattice `p_c` sanity check.

### 4.2 The scaling law
- Finding 1: `H1/vertex = 0.0381 + 0.1477/λ(q)`, `R²=0.9982`, `q=6..20`.
- Alternative-variable robustness check (`1/λ` beats `1/(q-4)`, `1/q²`, etc.).

### 4.3 Finite-size convergence splits at the amenable boundary
- Finding 2: nonamenable `q` converges within 1–2% almost immediately; `q=6` needs
  `~1000×` the vertices, power-law fit to `N≈3M`.

### 4.4 `q=6`'s position relative to the hyperbolic-only trend
- Leverage discussion, resolved exactly in 4.5.

### 4.5 Deriving the law
- Exact identity `∫β1(p)dp = w_MST − Σu_e + Σmax(triangle edges)`, floating-point-exact
  check against GUDHI.
- Reduction to `E[H1] = E[w_MST] − b/4`; boundary term `b/V→1−1/λ(q)`, proven `1/λ` piece
  with derived coefficient `1/4`.
- Bulk term `μ(q)`, independently measured to `N~2M`, own `1/λ(q)` law; reconstructs
  Finding 1 to four significant figures from two independent mechanisms.
- `q=6` gap resolved exactly via vanishing boundary term at `λ=1`; cross-checked between
  the two computational routes (MST density vs. full persistent homology) to within 0.1%.
- End with what's still open: `μ(q)`'s own `1/λ(q)` linearity is measured, not derived;
  natural next step is a transfer-matrix computation of `E[β0(p)]` ring-by-ring.

## 5. Part B — Vector embedding and its downstream cost (BRW + SAE)
**New section — write from README Findings 3–4, replacing the current `.tex`'s
unexecuted "sketch" paragraph.**

### 5.1 Motivation and method
- State the distinction explicitly, since it's the section's reason for existing: §4
  measures `H1` on the *abstract* percolation complex, independent of any embedding.
  This section asks what happens once a cluster is actually embedded into a vector
  space — the object any downstream model (an SAE, or any learned encoder) actually
  sees.
- BRW embedding: BFS spanning tree of each cluster, i.i.d. Gaussian steps in `R^64`,
  node coordinates as cumulative sums along the tree path. Exact on trees by
  construction; non-tree (cycle-closing) edges pick up nonzero implied holonomy exactly
  when the cluster has a loop.
- Cycle inconsistency metric: RMS norm of holonomy vectors over non-tree edges. Zero
  computation of homology required — direct embedding-space measurement.
- SAE: ReLU encoder, unit-norm decoder columns, L1 sparsity, trained on pooled BRW node
  embeddings per `q`. Feature-splitting score: dictionary atoms needed (greedy argmax
  cover) for ≥90% cluster-membership coverage, averaged per cluster.
- Note the equalized-cluster-count implementation detail (needed to avoid confounding
  splitting score with training-set size).

### 5.2 Results
- Finding 3: cycle inconsistency perfectly monotone in `q` (0.237→0.045 across
  `q=7,8,12,20` at matched `p/p_c`).
- Finding 4: feature-splitting score monotone in `q` (4.837→4.566), significant at
  `n=2000` clusters/`q` (`q=7` vs `q=20`: Welch `p=3.4e-11`); pooled per-cluster
  Spearman `ρ=-0.065, p=5e-9` — small effect, explicitly reported as such, not
  oversold.
- Cross-check: aggregate feature-splitting mean regressed against `H1`/vertex
  (`r=0.93, p=0.07`) and against cycle inconsistency (`r=0.95, p=0.048`) — only 4
  degrees of freedom, marginal `p`-values, but close to linear rather than merely
  ordinal. State the honest limitation: 4 `q`-values is not enough to distinguish
  "linear in `1/λ(q)`" from "some other decreasing function," unlike §4.2's 12-point
  check.
- Figure: `campaign/figure_feature_splitting_campaign.py` output, separating per-cluster
  spread from the precision of the per-`q` mean.

### 5.3 What this section does and doesn't establish
- Be explicit, as the current draft's Status section already is: agreement between
  §4 (combinatorial `H1`) and §5 (embedded/learned feature-splitting) is itself a
  (modest) finding, not a tautology, since embeddings can create or destroy loops
  relative to the abstract complex. Here they track together; that need not hold for
  a different embedding scheme (spectral, spring-model-with-non-tree-edges) or a
  different downstream architecture (TopK-SAE) — flagged as future work.

## 6. Connection to percolation-based data models
*(this is where Brill enters — reframed as application, not premise; adapt current
`.tex` §1's content into this later position)*

- Restate PrincInt's model briefly: data as critical percolation clusters on a
  high-dimensional lattice, Bethe-lattice-exact because high-`d` clusters are tree-like,
  BRW-embedded into feature space (Brill 2024, 2025).
- What §4 adds: an exact, finite, invertible closed-form measurement of tree-likeness as
  a function of curvature, replacing "asymptotically, in high dimension" with "at this
  specific `λ(q)`, this specific fraction of loop content survives." Since the law
  inverts, a target loop-persistence value picks out a `q` (or interpolated tiling)
  directly.
- What §5 adds beyond the original proposal: a second, independent check that the same
  suppression mechanism actually reaches all the way to a trained model's learned
  features, not just the input data's combinatorics — the thing the data model's
  authors ultimately care about.
- State plainly what this does *not* claim: this is not a demonstration that Brill's
  model is wrong or right in high dimension; it's a controllable, low-dimensional analog
  that makes the same tree-likeness assumption falsifiable and quantifiable at finite
  size, which the high-dimension asymptotic argument alone cannot offer.

## 7. Discussion and open questions
- Derive `μ(q)` from first principles (transfer-matrix / Husimi-cactus-style recursion).
- Vary `p` relative to `p_c` (current results are subcritical, `p≈0.75–0.85·p_c`);
  approach `p_c` directly and switch to max-cluster statistics.
- Vary embedding dimension `d` for the BRW step (currently fixed at 64) to check
  feature-splitting trends are stable, not an artifact of ambient dimension relative to
  cluster size.
- Canonical (lattice) tree vs. BFS (cluster) tree for BRW — tests whether cycle
  inconsistency reflects cluster topology or lattice/cluster tree mismatch.
- Non-tree-preserving embeddings (spectral, spring model with non-tree edges as
  constraints) as a sharper test of §5.3's open question.
- TopK-SAE to remove the sparsity-level confound between architectures.
- Extend the curvature sweep's interpolation between integer `q`.

## 8. Pipeline / code appendix
- `hyperbolic_lattice.py`, `percolation_topology.py` (Part A)
- `mu_q_convergence.py`, `q6_convergence.py` (Part A, §4.3/4.5)
- `brw_embedding.py`, `sae.py`, `feature_splitting.py`, `campaign/` (Part B)

---

## Immediate next actions (once this outline is approved)

1. Draft §1 (new intro) and §6 (Brill moved/reframed) as fresh prose — these are the
   only genuinely new writing needed beyond reorganization.
2. Draft §5 (BRW/SAE) from README Findings 3–4 — new prose, existing numbers.
3. Move current `.tex` §2–§5 into §2–§4 here with light edits (mostly section-number and
   cross-reference updates, not content rewrites).
4. Update the abstract last, once §1 and §6 exist, so it accurately reflects the new
   opening/closing frame rather than the old one.
