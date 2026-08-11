"""
ring6_energy_experiment.py

Tracks spring energy and repulsion energy ring-by-ring as grow_embedding
attempts to build {3,7} embeddings from ring 2 up through ring 6.

For each trial seed, records at each ring depth:
  - N, E
  - spring energy  = Σ_{edges} (|xi - xj| - 1)²
  - repulsion energy = Σ_{non-edge pairs, d<1} (1 - d)²
  - overlap count (non-edge pairs with d < 1.0)
  - min non-edge distance

Ring 6 is where the relaxer empirically fails (0/500 Rust trials).
This experiment asks: does the energy trajectory show a qualitative
change at ring 6, or does it look like a continuous extrapolation?

Usage:
    python ring6_energy_experiment.py [--trials 20] [--out ring6_results.json]
"""
import sys, os, json, time, math, argparse
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize

sys.path.insert(0, os.path.expanduser('~/Projects/237/embedding'))
from generate_unit_distance_library import (
    build_3q_disk, edges_from_adj, edge_stats, grow_embedding
)


# ── Energy functions ──────────────────────────────────────────────────────────

def spring_energy(pos, edges):
    d = np.linalg.norm(pos[edges[:, 0]] - pos[edges[:, 1]], axis=1)
    return float(np.sum((d - 1.0) ** 2))


def repulsion_stats(pos, edges, repulse_r=1.0):
    """Returns (repulsion_energy, overlap_count, min_ne_dist) using KD-tree."""
    N = len(pos)
    tree = cKDTree(pos)
    pairs = tree.query_pairs(max(repulse_r * 1.1, 1.2), output_type='ndarray')
    if len(pairs) == 0:
        return 0.0, 0, np.inf
    eset = set(map(tuple, np.sort(edges, axis=1)))
    ne_mask = np.array([(min(int(a), int(b)), max(int(a), int(b))) not in eset
                        for a, b in pairs])
    if not ne_mask.any():
        return 0.0, 0, np.inf
    ne = pairs[ne_mask]
    dists = np.linalg.norm(pos[ne[:, 0]] - pos[ne[:, 1]], axis=1)
    active = dists < repulse_r
    repulse_e = float(np.sum((repulse_r - dists[active]) ** 2)) if active.any() else 0.0
    n_overlaps = int(active.sum())
    min_ne = float(dists.min()) if len(dists) > 0 else np.inf
    return repulse_e, n_overlaps, min_ne


def ring_by_ring_stats(adj, rings_dict, ach, pos, edges):
    """Compute energy stats at each ring depth using the full embedded positions."""
    stats = {}
    cumulative_verts = [0]  # rings[0] = [center]
    for r in range(ach + 1):
        cumulative_verts.extend(rings_dict[r])

    placed = set()
    for r in range(ach + 1):
        for v in rings_dict[r]:
            placed.add(v)
        placed_arr = np.array(sorted(placed))
        sub_pos = pos[placed_arr]
        # remap edges to sub-indices
        idx = {v: i for i, v in enumerate(placed_arr)}
        sub_edges = np.array([(idx[u], idx[v]) for u in placed_arr
                              for v in adj[u] if v in idx and idx[v] > idx[u]],
                             dtype=np.int64)
        if len(sub_edges) == 0:
            continue
        se = spring_energy(sub_pos, sub_edges)
        re, ov, mn = repulsion_stats(sub_pos, sub_edges)
        dists = np.linalg.norm(sub_pos[sub_edges[:, 0]] - sub_pos[sub_edges[:, 1]], axis=1)
        stats[r] = dict(
            ring=r, N=len(placed_arr), E=len(sub_edges),
            spring_energy=se, repulsion_energy=re,
            n_overlaps=ov, min_ne_dist=mn,
            edge_std=float(dists.std()), edge_max_err=float(np.abs(dists - 1.0).max()),
        )
    return stats


# ── Stage-wise grow + record ──────────────────────────────────────────────────

def grow_and_record(adj, rings_dict, ach, seed, iters_per_stage=400, repulse=True):
    """
    Grow embedding ring by ring, recording energy stats after each ring's relaxation.
    Returns dict keyed by ring depth.
    """
    from generate_unit_distance_library import relax
    rng = np.random.default_rng(seed)
    Vtot = sum(len(rings_dict[k]) for k in range(ach + 1))
    pos = np.zeros((Vtot, 3))

    # ring 0: center
    pos[0] = rng.normal(0, 0.01, 3)
    placed = [0]

    # ring 1: unit circle in xy + small z noise
    q_val = len(rings_dict[1])
    for i, v in enumerate(rings_dict[1]):
        ang = 2 * math.pi * i / q_val
        pos[v] = [math.cos(ang), math.sin(ang), rng.uniform(-0.05, 0.05)]
    placed += rings_dict[1]

    trajectory = {}

    for k in range(2, ach + 1):
        # initialize new ring vertices outward from parents
        for v in rings_dict[k]:
            parents = [u for u in adj[v] if u in set(placed)]
            if parents:
                m = pos[parents].mean(axis=0)
                nm = np.linalg.norm(m)
                outward = m / nm if nm > 1e-6 else rng.normal(0, 1, 3)
                pos[v] = m + 0.9 * outward + rng.normal(0, 0.08, 3)
            else:
                pos[v] = rng.normal(0, k, 3)
            placed.append(v)

        sub = np.array(placed)
        idx = {v: i for i, v in enumerate(sub)}
        E_arr = np.array([(idx[u], idx[v]) for u in sub for v in adj[u]
                          if v in idx and idx[v] > idx[u]], dtype=np.int64)
        L = np.ones(len(E_arr))

        # repulsion pairs
        rp = None
        if repulse:
            tree = cKDTree(pos[sub])
            cand = np.array(sorted(tree.query_pairs(1.0)), dtype=np.int64)
            if len(cand):
                aset = {(min(a, b), max(a, b)) for a, b in E_arr}
                rp_list = [(a, b) for a, b in cand
                           if (min(a, b), max(a, b)) not in aset]
                if rp_list:
                    rp = np.array(rp_list, dtype=np.int64)

        pos[sub] = relax(pos[sub], E_arr, L,
                         iters=iters_per_stage + 50 * k,
                         repulse_pairs=rp, repulse_r=1.0, repulse_w=0.6)

        # record stats at this ring depth
        sub_pos = pos[sub]
        se = spring_energy(sub_pos, E_arr)
        re, ov, mn = repulsion_stats(sub_pos, E_arr)
        dists = np.linalg.norm(sub_pos[E_arr[:, 0]] - sub_pos[E_arr[:, 1]], axis=1)
        trajectory[k] = dict(
            ring=k, N=len(sub), E=len(E_arr),
            spring_energy=se, repulsion_energy=re,
            n_overlaps=ov, min_ne_dist=mn if np.isfinite(mn) else -1,
            edge_std=float(dists.std()),
            edge_max_err=float(np.abs(dists - 1.0).max()),
        )
        print(f'    ring {k}: N={len(sub):4d} E={len(E_arr):5d}  '
              f'spring={se:.3e}  repulsion={re:.3e}  '
              f'overlaps={ov:4d}  edge_std={dists.std():.2e}', flush=True)

    return trajectory


# ── Main ──────────────────────────────────────────────────────────────────────

def run(max_ring=6, n_trials=20, out_path='ring6_results.json'):
    print(f'Building {{3,7}} embedding trajectories to ring {max_ring}, '
          f'{n_trials} trials\n')

    adj, ring_of, rings_dict, ach = build_3q_disk(7, max_ring)
    print(f'Graph: N={sum(len(rings_dict[k]) for k in range(ach+1))}  '
          f'ach={ach}')
    for r in range(ach + 1):
        n = len(rings_dict[r])
        print(f'  ring {r}: {n} vertices')

    all_trials = []
    for trial in range(n_trials):
        t0 = time.time()
        print(f'\n── Trial {trial} ──')
        traj = grow_and_record(adj, rings_dict, ach, seed=trial)
        elapsed = time.time() - t0
        print(f'  done in {elapsed:.1f}s')
        all_trials.append({'trial': trial, 'trajectory': traj})

    # summary table
    print(f'\n{"ring":>5}  {"N":>5}  '
          f'{"spring mean":>12}  {"spring std":>11}  '
          f'{"repulsion mean":>14}  {"overlaps mean":>13}')
    for r in range(2, ach + 1):
        rows = [t['trajectory'].get(r) for t in all_trials
                if r in t['trajectory']]
        if not rows:
            continue
        sp = [x['spring_energy'] for x in rows]
        re = [x['repulsion_energy'] for x in rows]
        ov = [x['n_overlaps'] for x in rows]
        print(f'{r:>5}  {rows[0]["N"]:>5}  '
              f'{np.mean(sp):>12.3e}  {np.std(sp):>11.3e}  '
              f'{np.mean(re):>14.3e}  {np.mean(ov):>13.1f}')

    with open(out_path, 'w') as f:
        json.dump({'max_ring': max_ring, 'n_trials': n_trials,
                   'trials': all_trials}, f)
    print(f'\nSaved {out_path}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--max-ring', type=int, default=6)
    ap.add_argument('--trials', type=int, default=20)
    ap.add_argument('--out', default='ring6_results.json')
    args = ap.parse_args()
    run(max_ring=args.max_ring, n_trials=args.trials, out_path=args.out)
