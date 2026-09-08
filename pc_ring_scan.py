"""
Is the near-absence of H1 loops in the ring-4 critical-percolation figure a
genuine low density at p_c, or just a small-sample artifact of only having
N=232 vertices? Scan beta_1(p_c) of the flag complex across increasing ring
depth and look at beta_1/N.
"""
import time
import numpy as np
from mesh_topology import generate_mesh_topology, to_graph
from percolation_topology import relabel_to_int
import figure_poincare_percolation as fpp

Q = 7
P_C = 0.199351
N_TRIALS = 8
RING_DEPTHS = [4, 5, 6, 7, 8, 9]


def run(rings, n_trials=N_TRIALS):
    pairs, tris, rc = generate_mesh_topology(rings=rings, q=Q)
    G = to_graph(pairs)
    N = G.number_of_nodes()
    betas = []
    n_filled_list = []
    n_occ_list = []
    for seed in range(n_trials):
        H = fpp.percolate(G, P_C, seed)
        basis = fpp.h1_basis_of_flag_complex(H, tris)
        betas.append(len(basis))
        n_filled_list.append(len(fpp.filled_triangles(H, tris)))
        n_occ_list.append(H.number_of_edges())
    betas = np.array(betas)
    return {
        "rings": rings, "N": N,
        "beta1_mean": betas.mean(), "beta1_std": betas.std(),
        "beta1_per_N_mean": (betas / N).mean(),
        "beta1_per_N_std": (betas / N).std(),
        "n_filled_mean": np.mean(n_filled_list),
        "n_occ_mean": np.mean(n_occ_list),
    }


if __name__ == "__main__":
    t0 = time.time()
    for rings in RING_DEPTHS:
        r = run(rings)
        print(f"rings={r['rings']:2d} N={r['N']:6d}  "
              f"occ_edges~{r['n_occ_mean']:.0f}  filled_tri~{r['n_filled_mean']:.1f}  "
              f"beta1={r['beta1_mean']:.2f}+/-{r['beta1_std']:.2f}  "
              f"beta1/N={r['beta1_per_N_mean']:.5f}+/-{r['beta1_per_N_std']:.5f}  "
              f"[{time.time()-t0:.1f}s elapsed]")
