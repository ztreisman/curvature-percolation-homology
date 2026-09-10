"""
Hessian spectrum analysis of {3,7} unit-distance embeddings in R^3.

Uses generate_unit_distance_library.py's grow_embedding (ring-by-ring spring
relaxation with repulsion, the same method as the project's embedding pipeline)
for all ring depths, ensuring a consistent comparison.

For rings with a pre-built library (4, 5), the best trial is loaded; otherwise
grow_embedding is run with several seeds and the best result is kept.

Energy at the converged configuration:
    E = Σ_{edges} (|x_i - x_j| - 1)²
      + repulse_w * Σ_{non-edges, d < repulse_r} (repulse_r - |x_i - x_j|)²

Hessian computed via PyTorch forward-over-reverse autodiff.

Usage:
    python hessian_analysis.py [--rings 2 3 4 5] [--repulse-w 1] [--out results.json]
"""
import os
from _paths import HERE, RESULTS, FIGURES
import sys, os, json, time, argparse
import numpy as np
import torch
import torch.autograd.functional as AF
from scipy.spatial import cKDTree

# Path to generate_unit_distance_library.py
GEN_LIB = os.path.expanduser('~/Projects/237/embedding/generate_unit_distance_library.py')
UDG_LIB = os.path.expanduser('~/Projects/237/python/udg_library')


def _import_gen():
    import importlib.util
    spec = importlib.util.spec_from_file_location('gen', GEN_LIB)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ── embedding ─────────────────────────────────────────────────────────────────

def embed_ring(rings, q=7, n_seeds=5):
    """
    Return (positions, edges) for {3,q} ring depth `rings`.

    For rings with a pre-built library (udg_library), loads the best trial.
    Otherwise runs grow_embedding with `n_seeds` seeds and returns the trial
    with the fewest vertex overlaps (tie-broken by lowest edge_std).
    """
    gen = _import_gen()

    # Try loading from pre-built library first
    manifest = os.path.join(UDG_LIB, 'manifest.jsonl')
    if os.path.exists(manifest):
        with open(manifest) as f:
            bases = [json.loads(l) for l in f
                     if json.loads(l).get('kind') == 'base'
                     and json.loads(l).get('rings') == rings
                     and json.loads(l).get('q') == q
                     and json.loads(l).get('dim') == 3]
        if bases:
            best = min(bases, key=lambda b: b['edge_std'])
            d = np.load(os.path.join(UDG_LIB, best['file']))
            pos, edges = d['positions'], d['edges'].astype(np.int64)
            dists = np.linalg.norm(pos[edges[:, 0]] - pos[edges[:, 1]], axis=1)
            tree = cKDTree(pos)
            eset = set(map(tuple, np.sort(edges, axis=1)))
            ov = sum(1 for a, b in tree.query_pairs(1.0)
                     if (min(a, b), max(a, b)) not in eset)
            print(f'  Loaded ring {rings} from library: {best["file"]}  '
                  f'edge_std={best["edge_std"]:.2e}  overlaps={ov}')
            return pos, edges, {'source': 'library', 'file': best['file'],
                                'edge_std': best['edge_std'], 'overlaps': ov}

    # Synthesize with grow_embedding
    adj, ring_of, rings_d, ach = gen.build_3q_disk(q, rings)
    edges = gen.edges_from_adj(adj)
    eset = set(map(tuple, np.sort(edges, axis=1)))

    best_pos, best_ov, best_std = None, np.inf, np.inf
    for seed in range(n_seeds):
        P = gen.grow_embedding(adj, rings_d, ach, dim=3, seed=seed, repulse=True)
        dists = np.linalg.norm(P[edges[:, 0]] - P[edges[:, 1]], axis=1)
        std = float(dists.std())
        tree = cKDTree(P)
        ov = sum(1 for a, b in tree.query_pairs(1.0)
                 if (min(a, b), max(a, b)) not in eset)
        if (ov, std) < (best_ov, best_std):
            best_pos, best_ov, best_std = P, ov, std

    dists = np.linalg.norm(best_pos[edges[:, 0]] - best_pos[edges[:, 1]], axis=1)
    print(f'  Synthesized ring {rings}: edge_std={best_std:.2e}  overlaps={best_ov}')
    return best_pos, edges, {'source': 'synthesized', 'edge_std': best_std, 'overlaps': best_ov}


# ── non-edge computation ──────────────────────────────────────────────────────

def build_non_edges(N, edges, positions, cutoff=1.5):
    edge_set = set(map(tuple, np.sort(edges, axis=1)))
    tree = cKDTree(positions)
    pairs = tree.query_pairs(cutoff, output_type='ndarray')
    ne = [(int(i), int(j)) for i, j in pairs
          if (min(i, j), max(i, j)) not in edge_set]
    return np.array(ne, dtype=np.int64) if ne else np.empty((0, 2), dtype=np.int64)


# ── energy function ───────────────────────────────────────────────────────────

def make_energy_fn(edges_t, non_edges_t, repulse_r=1.0, repulse_w=1.0):
    def energy(positions_flat):
        pos = positions_flat.reshape(-1, 3)
        dists_e = torch.norm(pos[edges_t[:, 0]] - pos[edges_t[:, 1]], dim=1)
        spring = ((dists_e - 1.0) ** 2).sum()
        if non_edges_t.shape[0] > 0:
            dists_r = torch.norm(pos[non_edges_t[:, 0]] - pos[non_edges_t[:, 1]], dim=1)
            repulse = repulse_w * (torch.clamp(repulse_r - dists_r, min=0.0) ** 2).sum()
        else:
            repulse = torch.zeros(1, dtype=positions_flat.dtype)
        return spring + repulse
    return energy


# ── Hessian computation ───────────────────────────────────────────────────────

def compute_hessian_eigenvalues(positions, edges, non_edges,
                                repulse_r=1.0, repulse_w=1.0):
    N = len(positions)
    edges_t = torch.tensor(edges, dtype=torch.long)
    non_edges_t = torch.tensor(non_edges, dtype=torch.long)
    pos_flat = torch.tensor(positions.flatten(), dtype=torch.float64,
                            requires_grad=True)

    energy_fn = make_energy_fn(edges_t, non_edges_t, repulse_r, repulse_w)

    print(f'  Computing Hessian ({N * 3}×{N * 3}) ...', flush=True)
    t0 = time.time()
    H = AF.hessian(energy_fn, pos_flat, vectorize=True)
    H_np = H.detach().numpy()
    print(f'  Hessian done in {time.time() - t0:.1f}s')

    H_sym = (H_np + H_np.T) / 2
    return np.linalg.eigvalsh(H_sym)


# ── spectrum statistics ───────────────────────────────────────────────────────

def maxwell_floppy(N, E):
    return max(0, 3 * N - 6 - E)


def spectrum_stats(eigvals, N, E_count, tol_zero=1e-3, tol_soft=5e-2):
    zeros    = int((np.abs(eigvals) < tol_zero).sum())
    soft     = int(((np.abs(eigvals) >= tol_zero) & (np.abs(eigvals) < tol_soft)).sum())
    stiff    = int((eigvals >= tol_soft).sum())
    negative = int((eigvals < -tol_zero).sum())
    maxwell  = maxwell_floppy(N, E_count)
    expected_zeros = maxwell + 6
    surviving_frac = zeros / max(expected_zeros, 1)
    return dict(
        n_params=3 * N,
        zeros=zeros, soft=soft, stiff=stiff, negative=negative,
        maxwell=maxwell, expected_zeros=expected_zeros,
        surviving_fraction=surviving_frac,
        min_eigval=float(eigvals[0]), max_eigval=float(eigvals[-1]),
        eigval_10=float(np.percentile(eigvals, 10)),
        eigval_90=float(np.percentile(eigvals, 90)),
        tol_zero=tol_zero, tol_soft=tol_soft,
    )


# ── main analysis loop ────────────────────────────────────────────────────────

def run_ring_comparison(rings_list, q=7,
                        repulse_r=1.0, repulse_w=1.0,
                        cutoff=1.5, tol_zero=1e-3, tol_soft=5e-2,
                        n_seeds=5,
                        out_path=os.path.join(HERE, 'hessian_results.json')):
    results = {}
    all_eigvals = {}

    for rings in rings_list:
        print(f'\n{"=" * 60}')
        print(f'Ring {rings}  (q={q})')

        pos, edges, meta = embed_ring(rings, q=q, n_seeds=n_seeds)
        N, E = len(pos), len(edges)
        maxwell = maxwell_floppy(N, E)
        print(f'  N={N}  E={E}  3N-6={3*N-6}  Maxwell floppy={maxwell}')

        dists = np.linalg.norm(pos[edges[:, 0]] - pos[edges[:, 1]], axis=1)
        spring_E = float(((dists - 1.0) ** 2).sum())
        print(f'  Spring energy: {spring_E:.2e}  '
              f'edge length range [{dists.min():.6f}, {dists.max():.6f}]')

        non_edges = build_non_edges(N, edges, pos, cutoff=cutoff)
        active = int((np.linalg.norm(
            pos[non_edges[:, 0]] - pos[non_edges[:, 1]], axis=1) < repulse_r
        ).sum()) if len(non_edges) > 0 else 0
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
            n_non_edges=len(non_edges), n_active_repulsion=active,
            embed_source=meta.get('source', 'unknown'),
            embed_overlaps=meta.get('overlaps', -1),
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
    parser.add_argument('--rings', nargs='+', type=int, default=[2, 3, 4, 5])
    parser.add_argument('--q', type=int, default=7)
    parser.add_argument('--repulse-r', type=float, default=1.0)
    parser.add_argument('--repulse-w', type=float, default=1.0)
    parser.add_argument('--tol-zero', type=float, default=1e-3)
    parser.add_argument('--tol-soft', type=float, default=5e-2)
    parser.add_argument('--n-seeds', type=int, default=5)
    parser.add_argument('--out', default=os.path.join(HERE, 'hessian_results.json'))
    args = parser.parse_args()

    run_ring_comparison(
        args.rings, q=args.q,
        repulse_r=args.repulse_r, repulse_w=args.repulse_w,
        tol_zero=args.tol_zero, tol_soft=args.tol_soft,
        n_seeds=args.n_seeds,
        out_path=args.out,
    )
