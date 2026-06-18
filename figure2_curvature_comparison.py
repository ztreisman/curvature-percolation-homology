"""
Figure 2: Beta_1/N curves for q=6,7,8 overlaid, plus total H1 persistence
per vertex vs q for those three cases.
Outputs: curvature_persistence_comparison.png

Requires that the three numpy data files already exist (generated during the
initial three-point sweep). Re-run the sweep cells if starting fresh:
    euclid_b1.npy, euclid_b0.npy, hyp37_b1.npy, hyp37_b0.npy,
    hyp38_b1.npy, hyp38_b0.npy, euclid_ps.npy
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ps = np.load('euclid_ps.npy')

data = {
    r'$q=6$  (flat, Euclidean control)':   ('euclid_b1.npy', 625,  '#2b6cb0'),
    r'$q=7$  ($\{3,7\}$)':                 ('hyp37_b1.npy',  617,  '#c05621'),
    r'$q=8$  (more negatively curved)':    ('hyp38_b1.npy',  609,  '#9b2c2c'),
}

q_vals = [6, 7, 8]
means  = [0.16021647, 0.08872769, 0.07608730]
stds   = [0.00720748, 0.00530705, 0.00215645]
colors = ['#2b6cb0', '#c05621', '#9b2c2c']

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

for label, (fname, n, color) in data.items():
    b1 = np.load(fname)
    mean_b1 = b1.mean(axis=0) / n
    std_b1  = b1.std(axis=0)  / n
    axes[0].plot(ps, mean_b1, color=color, lw=2, label=label)
    axes[0].fill_between(ps, mean_b1 - std_b1, mean_b1 + std_b1,
                          color=color, alpha=0.15)

axes[0].set_xlabel('occupation probability  $p$')
axes[0].set_ylabel(r'$\beta_1 / N$')
axes[0].set_title('Persistent loop density vs $p$, across curvature')
axes[0].legend(fontsize=8.5, loc='upper right')

axes[1].errorbar(q_vals, means, yerr=stds, fmt='-', color='black',
                  ecolor='gray', capsize=4, lw=1.5, zorder=1)
for x, y, c in zip(q_vals, means, colors):
    axes[1].plot(x, y, 'o', color=c, markersize=10, zorder=5)
axes[1].set_xticks(q_vals)
axes[1].set_xlim(5.7, 8.5)
axes[1].set_ylim(0.06, 0.18)
axes[1].set_xlabel(r'vertex degree $q$  in  $\{3,q\}$   (curvature knob)')
axes[1].set_ylabel('total H1 persistence per vertex')
axes[1].set_title('Loop persistence collapses with curvature')
axes[1].annotate('flat', xy=(6, 0.160), xytext=(6.15, 0.172),
                  color='#2b6cb0', fontsize=9)
axes[1].annotate('hyperbolic,\nmore tree-like', xy=(8, 0.076),
                  xytext=(6.3, 0.085), color='#9b2c2c', fontsize=9)

fig.tight_layout()
fig.savefig('curvature_persistence_comparison.png', dpi=160)
print('saved')
