"""
Figure 1: Betti curves for bond percolation on the flat {3,6} triangular
lattice (N=625), 12 independent percolation trials.
Outputs: euclidean_control_betti_curves.png
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from euclidean_lattice import triangular_lattice
from percolation_topology import relabel_to_int, random_edge_filtration, \
    build_filtered_complex, betti_curve, loopiness_stats

G = triangular_lattice(25)
G, _ = relabel_to_int(G)
n = G.number_of_nodes()

ps = np.linspace(0, 1, 200)
n_trials = 12
b0_all = np.zeros((n_trials, len(ps)))
b1_all = np.zeros((n_trials, len(ps)))

for t in range(n_trials):
    filt = random_edge_filtration(G, seed=t)
    st = build_filtered_complex(G, filt, max_dim=2)
    st.compute_persistence()
    b0_all[t] = betti_curve(st, 0, ps)
    b1_all[t] = betti_curve(st, 1, ps)

mean_b1 = b1_all.mean(axis=0) / n
std_b1  = b1_all.std(axis=0)  / n
mean_b0 = b0_all.mean(axis=0) / n

pc_triangular = 2 * np.sin(np.pi / 18)   # exact bond threshold

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(ps, mean_b0, color='#2b6cb0', lw=2, label=r'$\beta_0/N$  (components)')
ax.plot(ps, mean_b1, color='#c05621', lw=2, label=r'$\beta_1/N$  (independent loops)')
ax.fill_between(ps, mean_b1 - std_b1, mean_b1 + std_b1, color='#c05621', alpha=0.2)
ax.axvline(pc_triangular, color='gray', ls='--', lw=1)
ax.text(pc_triangular + 0.01, 0.16, r'$p_c \approx 0.347$', color='gray', fontsize=9)
ax.set_xlabel('occupation probability  $p$')
ax.set_ylabel('Betti number / $N$')
ax.set_title(r'Bond percolation on flat $\{3,6\}$ lattice ($N=625$), 12 trials')
ax.legend(fontsize=9, loc='upper right')
ax.set_ylim(0, max(mean_b0.max(), mean_b1.max()) * 1.1)
fig.tight_layout()
fig.savefig('/mnt/user-data/outputs/euclidean_control_betti_curves.png', dpi=160)
print('saved')
