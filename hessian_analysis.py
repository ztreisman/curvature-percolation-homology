"""
Hessian spectrum analysis of {3,7} unit-distance embeddings.

For each ring depth r, computes the Hessian of the implicit energy minimized
by the physics relaxer at a converged configuration, then classifies its
eigenvalues into rigid-motion zeros, near-zero (soft) modes, and stiff modes.

Energy:
    E = Σ_{edges} (|x_i - x_j| - 1)²  +  repulse_w * Σ_{active non-edges} (repulse_r - |x_i - x_j|)²

where "active non-edges" are non-adjacent pairs within distance repulse_r.

The Maxwell rigidity count predicts (3N - 6 - E_count) floppy modes from the
spring constraints alone.  The repulsion term stiffens modes that would bring
near-touching non-edge pairs closer.  The key signal: what fraction of Maxwell
floppy modes survive as genuine near-zero Hessian eigenvalues?

Usage:
    python hessian_analysis.py [--rings 2 3 4 5] [--out results.json]

Ring 4 and 5 embeddings are loaded from the 237 project library.
Ring 2 and 3 embeddings are generated here via gradient descent.
"""
import sys, os, json, time, glob, argparse
import numpy as np
import torch
import torch.autograd.functional as AF
from scipy.spatial import cKDTree

# Path to the 237-project ring libraries (update if moved)
LIB_DIR = os.path.expanduser('~/Projects/237/mlp_experiments')
RING_LIB = {
    4: os.path.join(LIB_DIR, 'ring4_poincare_lib'),
    5: os.path.join(LIB_DIR, 'ring5_poincare_lib'),
}

sys.path.insert(0, os.path.dirname(__file__))
from mesh_topology import generate_mesh_topology, to_graph


# ── graph generation ──────────────────────────────────────────────────────────

def get_graph(rings, q=7):
    """Return (edges_np, N) for {3,q} ring-depth `rings`."""
    pairs, tris, rc = generate_mesh_topology(rings=rings, q=q)
    G = to_graph(pairs)
    edges = np.array(sorted([tuple(sorted(e)) for e in G.edges()]), dtype=np.int64)
    N = G.number_of_nodes()
    return edges, N


# ── embedding: load or synthesize ────────────────────────────────────────────

def load_best_embedding(rings):
    """Load the best-converged (lowest edge_std) base trial from the library."""
    lib = RING_LIB.get(rings)
    if lib is None or not os.path.isdir(lib):
        return None
    manifest = os.path.join(lib, 'manifest.jsonl')
    if not os.path.exists(manifest):
        return None
    bases = []
    with open(manifest) as f:
        for line in f:
            r = json.loads(line)
            if r.get('kind') == 'base':
                bases.append(r)
    if not bases:
        return None
    best = min(bases, key=lambda r: r['edge_std'])
    path = os.path.join(lib, best['file'])
    d = np.load(path)
    return d['positions'], d['edges'].astype(np.int64), best


def _ring_init_positions(rings, q, edges, N, rng, scale=2.5):
    """
    Initialize positions ring-by-ring: ring k on a circle of radius scale*k
    in the xy-plane with small z-jitter.  scale > 1 ensures non-edge pairs
    start well-separated before optimization shrinks them to unit distance.
    """
    from mesh_topology import ring_counts_seq
    rc = ring_counts_seq(q, rings)
    pos = np.zeros((N, 3))
    pos[0] = [0, 0, rng.uniform(-0.05, 0.05)]
    offset = 1
    for r in range(1, rings + 1):
        count = rc[r]
        for i in range(count):
            angle = 2 * np.pi * i / count
            pos[offset + i] = [scale * r * np.cos(angle),
                                scale * r * np.sin(angle),
                                rng.uniform(-0.05 * r, 0.05 * r)]
        offset += count
    pos += rng.normal(0, 0.02, pos.shape)
    return pos


def synthesize_embedding(rings, q=7, seed=0, init_scale=2.5):
    """
    Generate a converged, collision-free unit-distance embedding.

    Uses a scaled ring-by-ring initialization (vertices on circles of radius
    init_scale * r) to ensure non-edge pairs start well-separated, then
    runs Adam + L-BFGS to minimize the full spring+repulsion energy.

    Returns (positions, edges).
    """
    edges, N = get_graph(rings, q)
    rng = np.random.default_rng(seed)

    # Build complete non-edge list (feasible for small N)
    adj_set = set(map(tuple, np.sort(edges, axis=1)))
    all_non_edges = np.array([(i, j) for i in range(N) for j in range(i+1, N)
                               if (i, j) not in adj_set], dtype=np.int64)

    pos = _ring_init_positions(rings, q, edges, N, rng, scale=init_scale)
    pos_t = torch.tensor(pos, dtype=torch.float64, requires_grad=True)
    edges_t = torch.tensor(edges, dtype=torch.long)
    ne_t = torch.tensor(all_non_edges, dtype=torch.long)

    def full_loss():
        d_e = torch.norm(pos_t[edges_t[:, 0]] - pos_t[edges_t[:, 1]], dim=1)
        spring = ((d_e - 1.0)**2).sum()
        d_r = torch.norm(pos_t[ne_t[:, 0]] - pos_t[ne_t[:, 1]], dim=1)
        repulse = (torch.clamp(1.0 - d_r, min=0.0)**2).sum()
        return spring + repulse

    # Phase 1: Adam for large-scale convergence
    opt1 = torch.optim.Adam([pos_t], lr=0.02)
    for step in range(60000):
        opt1.zero_grad()
        loss = full_loss(); loss.backward(); opt1.step()
        if step % 10000 == 0:
            print(f'    [Adam] step {step:5d}  loss={loss.item():.2e}', flush=True)
        if loss.item() < 1e-10:
            break

    # Phase 2: L-BFGS for tight convergence
    opt2 = torch.optim.LBFGS([pos_t], lr=0.5, max_iter=200,
                               line_search_fn='strong_wolfe')
    for _ in range(30):
        def closure():
            opt2.zero_grad()
            l = full_loss(); l.backward(); return l
        opt2.step(closure)
        if full_loss().item() < 1e-10:
            break

    positions = pos_t.detach().numpy()
    dists = np.linalg.norm(positions[edges[:, 0]] - positions[edges[:, 1]], axis=1)
    spring_err = float(((dists - 1.0)**2).sum())
    dists_ne = np.linalg.norm(positions[all_non_edges[:, 0]] -
                               positions[all_non_edges[:, 1]], axis=1)
    collisions = int((dists_ne < 1.0).sum())
    min_ne = float(dists_ne.min()) if len(dists_ne) > 0 else np.inf
    print(f'    spring={spring_err:.2e}  collisions={collisions}  '
          f'min_non_edge={min_ne:.5f}')
    return positions, edges


def get_embedding(rings, q=7):
    """Return (positions, edges, source_info) for the given ring depth."""
    result = load_best_embedding(rings)
    if result is not None:
        pos, edges, meta = result
        print(f'  Loaded ring {rings} from library: {meta["file"]}  '
              f'edge_std={meta["edge_std"]:.2e}')
        return pos, edges, meta

    print(f'  Synthesizing ring {rings} embedding via gradient descent ...')
    pos, edges = synthesize_embedding(rings, q=q)
    return pos, edges, {'source': 'synthesized', 'rings': rings}


# ── non-edge computation ──────────────────────────────────────────────────────

def build_non_edges(N, edges, positions, cutoff=1.5):
    """
    Find all non-adjacent pairs within `cutoff` distance.
    Uses a KD-tree for efficiency (O(N log N) for sparse cutoff neighborhoods).
    """
    edge_set = set(map(tuple, np.sort(edges, axis=1)))
    tree = cKDTree(positions)
    pairs = tree.query_pairs(cutoff, output_type='ndarray')
    non_edges = []
    for i, j in pairs:
        if (min(i, j), max(i, j)) not in edge_set:
            non_edges.append((i, j))
    return np.array(non_edges, dtype=np.int64) if non_edges else np.empty((0, 2), dtype=np.int64)


# ── energy function ───────────────────────────────────────────────────────────

def make_energy_fn(edges_t, non_edges_t, repulse_r=1.0, repulse_w=1.0):
    """
    Return a function  f: (N*3,) tensor → scalar  for use with torch.autograd.

    Uses torch.clamp for the repulsion term (not boolean masking) so that
    the energy function is everywhere differentiable and torch.autograd.functional
    .hessian computes the correct, positive-semi-definite Hessian contributions.
    Boolean masking inside autodiff produces incorrect Hessians near the boundary.
    """
    def energy(positions_flat):
        pos = positions_flat.reshape(-1, 3)
        # Spring term: penalty for edge length ≠ 1
        pi = pos[edges_t[:, 0]]
        pj = pos[edges_t[:, 1]]
        dists_e = torch.norm(pi - pj, dim=1)
        spring = ((dists_e - 1.0)**2).sum()
        # Repulsion: clamp(repulse_r - d, min=0)^2  — zero for d ≥ repulse_r,
        # smoothly positive for d < repulse_r.  No boolean mask needed.
        if non_edges_t.shape[0] > 0:
            pi = pos[non_edges_t[:, 0]]
            pj = pos[non_edges_t[:, 1]]
            dists_r = torch.norm(pi - pj, dim=1)
            repulse = repulse_w * (torch.clamp(repulse_r - dists_r, min=0.0)**2).sum()
        else:
            repulse = torch.zeros(1, dtype=positions_flat.dtype)
        return spring + repulse
    return energy


# ── Hessian computation ───────────────────────────────────────────────────────

def compute_hessian_eigenvalues(positions, edges, non_edges,
                                repulse_r=1.0, repulse_w=1.0):
    """
    Compute the (N*3 × N*3) Hessian of the energy at `positions` via
    PyTorch's forward-over-reverse autodiff, then return sorted eigenvalues.

    For N=617 (ring 5) this produces a 1851×1851 matrix.  Expect ~1–5 minutes
    on CPU.
    """
    N = len(positions)
    edges_t = torch.tensor(edges, dtype=torch.long)
    non_edges_t = torch.tensor(non_edges, dtype=torch.long)
    pos_flat = torch.tensor(positions.flatten(), dtype=torch.float64, requires_grad=True)

    energy_fn = make_energy_fn(edges_t, non_edges_t, repulse_r, repulse_w)

    print(f'  Computing Hessian ({N*3}×{N*3}) ...', flush=True)
    t0 = time.time()
    H = AF.hessian(energy_fn, pos_flat, vectorize=True)
    H_np = H.detach().numpy()
    print(f'  Hessian done in {time.time()-t0:.1f}s')

    H_sym = (H_np + H_np.T) / 2      # symmetrize for numerical stability
    eigvals = np.linalg.eigvalsh(H_sym)
    return eigvals


# ── spectrum statistics ───────────────────────────────────────────────────────

def maxwell_floppy(N, E):
    """Maxwell rigidity count: expected near-zero eigenvalues from spring constraints."""
    return max(0, 3*N - 6 - E)


def spectrum_stats(eigvals, N, E_count,
                   tol_zero=1e-3, tol_soft=5e-2):
    """
    Classify eigenvalues and compute summary statistics.

    Parameters
    ----------
    eigvals   : sorted eigenvalues (ascending)
    N         : vertex count
    E_count   : edge count
    tol_zero  : |λ| < tol_zero → "zero" (rigid + genuine flat)
    tol_soft  : tol_zero ≤ |λ| < tol_soft → "soft mode"

    Returns dict with counts and the key SLT proxy:
      surviving_fraction = zeros / (maxwell_floppy + 6)
    """
    zeros = int((np.abs(eigvals) < tol_zero).sum())
    soft  = int(((np.abs(eigvals) >= tol_zero) & (np.abs(eigvals) < tol_soft)).sum())
    stiff = int((eigvals >= tol_soft).sum())
    negative = int((eigvals < -tol_zero).sum())
    maxwell = maxwell_floppy(N, E_count)
    expected_zeros = maxwell + 6   # Maxwell floppy + rigid motions
    surviving_frac = zeros / max(expected_zeros, 1)
    return dict(
        n_params   = 3 * N,
        zeros      = zeros,
        soft       = soft,
        stiff      = stiff,
        negative   = negative,
        maxwell    = maxwell,
        expected_zeros = expected_zeros,
        surviving_fraction = surviving_frac,
        min_eigval = float(eigvals[0]),
        max_eigval = float(eigvals[-1]),
        eigval_10  = float(np.percentile(eigvals, 10)),
        eigval_90  = float(np.percentile(eigvals, 90)),
        tol_zero   = tol_zero,
        tol_soft   = tol_soft,
    )


# ── main analysis loop ────────────────────────────────────────────────────────

def run_ring_comparison(rings_list, q=7,
                        repulse_r=1.0, repulse_w=1.0,
                        cutoff=1.5, tol_zero=1e-3, tol_soft=5e-2,
                        out_path='hessian_results.json'):
    results = {}
    all_eigvals = {}

    for rings in rings_list:
        print(f'\n{"="*60}')
        print(f'Ring {rings}  (q={q})')

        pos, edges, meta = get_embedding(rings, q=q)
        N = len(pos)
        E = len(edges)
        maxwell = maxwell_floppy(N, E)
        print(f'  N={N}  E={E}  3N-6={3*N-6}  Maxwell floppy={maxwell}')

        # Verify spring energy at loaded embedding
        diffs = pos[edges[:, 0]] - pos[edges[:, 1]]
        dists = np.linalg.norm(diffs, axis=1)
        spring_E = float(((dists - 1.0)**2).sum())
        print(f'  Spring energy: {spring_E:.2e}  '
              f'edge length range [{dists.min():.6f}, {dists.max():.6f}]')

        # Build non-edges
        non_edges = build_non_edges(N, edges, pos, cutoff=cutoff)
        active = (np.linalg.norm(
            pos[non_edges[:, 0]] - pos[non_edges[:, 1]], axis=1) < repulse_r
        ).sum() if len(non_edges) > 0 else 0
        print(f'  Non-edge pairs within {cutoff}: {len(non_edges)}  '
              f'active repulsion (< {repulse_r}): {active}')

        eigvals = compute_hessian_eigenvalues(
            pos, edges, non_edges, repulse_r=repulse_r, repulse_w=repulse_w)

        stats = spectrum_stats(eigvals, N, E, tol_zero=tol_zero, tol_soft=tol_soft)
        print(f'  Eigenvalue counts:')
        print(f'    zeros (|λ|<{tol_zero:.0e}): {stats["zeros"]}  '
              f'expected (Maxwell+6): {stats["expected_zeros"]}')
        print(f'    soft  (|λ|<{tol_soft:.0e}): {stats["soft"]}')
        print(f'    stiff:                   {stats["stiff"]}')
        print(f'    negative:                {stats["negative"]}')
        print(f'    surviving fraction:      {stats["surviving_fraction"]:.3f}')
        print(f'  min eigenvalue: {stats["min_eigval"]:.4e}')
        print(f'  max eigenvalue: {stats["max_eigval"]:.4e}')

        results[rings] = dict(
            rings=rings, q=q, N=N, E=E,
            spring_energy=spring_E,
            n_non_edges=len(non_edges), n_active_repulsion=int(active),
            **stats,
        )
        all_eigvals[rings] = eigvals.tolist()

    with open(out_path, 'w') as f:
        json.dump({'results': results, 'eigvals': all_eigvals}, f)
    print(f'\nSaved {out_path}')
    return results, all_eigvals


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rings', nargs='+', type=int, default=[2, 3, 4, 5],
                        help='Ring depths to analyze (default: 2 3 4 5)')
    parser.add_argument('--q', type=int, default=7)
    parser.add_argument('--repulse-r', type=float, default=1.0)
    parser.add_argument('--repulse-w', type=float, default=1.0)
    parser.add_argument('--tol-zero', type=float, default=1e-3)
    parser.add_argument('--tol-soft', type=float, default=5e-2)
    parser.add_argument('--out', default='hessian_results.json')
    args = parser.parse_args()

    run_ring_comparison(
        args.rings, q=args.q,
        repulse_r=args.repulse_r, repulse_w=args.repulse_w,
        tol_zero=args.tol_zero, tol_soft=args.tol_soft,
        out_path=args.out,
    )
