"""
Figure: the bulk MST-weight density mu(q) obeys its own clean 1/lambda(q) law,
and combining it with the (exactly derived) boundary term reconstructs
Finding 1's directly-fitted scaling law almost exactly.

Left panel:  mu(q) vs 1/lambda(q), with fit.
Right panel: the two-mechanism reconstruction overlaid on the direct fit.

Requires mu_q_results.json and sweep_results.json / q6_convergence_results.json
(for the direct Finding-1 fit line).

Usage:
    python figure_mu_decomposition.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

BLUE = '#2a78d6'
BLUE_DARK = '#134a8f'
RED = '#9b2c2c'
GRID = '#e3e2dd'

with open('mu_q_results.json') as f:
    mu_d = json.load(f)
qs = sorted(mu_d, key=int)
q_arr = np.array([int(q) for q in qs], dtype=float)
lam = np.array([mu_d[q]['lam'] for q in qs])
mu = np.array([mu_d[q]['largest_mean'] for q in qs])

X = np.vstack([1 / lam, np.ones_like(lam)]).T
coef, *_ = np.linalg.lstsq(X, mu, rcond=None)
slope_mu, intercept_mu = coef
pred_mu = intercept_mu + slope_mu / lam
r2_mu = 1 - np.sum((mu - pred_mu) ** 2) / np.sum((mu - mu.mean()) ** 2)

# leave-q6-out fit: q=6 is the only lambda=1 (amenable) point and sits at
# extreme leverage (1/lambda=1, far from the hyperbolic cluster in [0.05,0.4])
mask6 = np.array([int(q) == 6 for q in qs])
X_no6 = np.vstack([1 / lam[~mask6], np.ones_like(lam[~mask6])]).T
coef_no6, *_ = np.linalg.lstsq(X_no6, mu[~mask6], rcond=None)
slope_no6, intercept_no6 = coef_no6
pred_no6 = intercept_no6 + slope_no6 / lam[~mask6]
r2_no6 = 1 - np.sum((mu[~mask6] - pred_no6) ** 2) / np.sum((mu[~mask6] - mu[~mask6].mean()) ** 2)
mu6_hyperbolic_pred = intercept_no6 + slope_no6 * 1.0

# combined prediction: H1/V = mu(q) - 1/4 + 1/(4 lambda)
combined_intercept = intercept_mu - 0.25
combined_slope = slope_mu + 0.25

# direct Finding-1 fit (from sweep_results.json + converged q=6 asymptote)
with open('sweep_results.json') as f:
    sweep = json.load(f)
by_q = defaultdict(list)
for r in sweep:
    by_q[r['q']].append(r)
for q in by_q:
    by_q[q].sort(key=lambda r: r['V'])
sweep_qs = sorted(by_q.keys())
best = {q: by_q[q][-1] for q in sweep_qs}
sweep_lam = np.array([best[q]['lambda'] for q in sweep_qs])
sweep_means = np.array([best[q]['mean'] for q in sweep_qs])
sweep_means[sweep_qs.index(6)] = 0.18757
Xd = np.vstack([1 / sweep_lam, np.ones_like(sweep_lam)]).T
direct_coef, *_ = np.linalg.lstsq(Xd, sweep_means, rcond=None)
direct_slope, direct_intercept = direct_coef

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

ax = axes[0]
ax.scatter(1 / lam[~mask6], mu[~mask6], color=RED, zorder=5, s=40, label='q=7..20')
ax.scatter(1 / lam[mask6], mu[mask6], color='#333', zorder=6, s=55, marker='D',
          label='q=6 (amenable, excluded from dashed fit)')
xx = np.linspace(0, 1, 100)
ax.plot(xx, intercept_mu + slope_mu * xx, color=BLUE, lw=1.2, ls=':',
        label=f'all-12 fit: ${intercept_mu:.4f} {slope_mu:+.4f}/\\lambda$  '
              f'($R^2={r2_mu:.4f}$)')
ax.plot(xx, intercept_no6 + slope_no6 * xx, color=BLUE_DARK, lw=1.6, ls='--',
        label=f'q$\\geq$7 only: ${intercept_no6:.4f} {slope_no6:+.4f}/\\lambda$  '
              f'($R^2={r2_no6:.4f}$)')
for q, x, y in zip(qs, 1 / lam, mu):
    ax.annotate(f'q={q}', (x, y), textcoords='offset points', xytext=(4, 4),
               fontsize=8, color='#555')
ax.set_xlabel(r'$1/\lambda(q)$')
ax.set_ylabel(r'$\mu(q) = \mathbb{E}[w_{\mathrm{MST}}]/V$')
ax.set_title('A.  Bulk MST-weight density: q=6 sits off the hyperbolic trend')
ax.legend(fontsize=7)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

ax = axes[1]
lam_grid = np.linspace(1, 17, 300)
ax.plot(lam_grid, direct_intercept + direct_slope / lam_grid, color='#9a9a94',
        lw=2.5, label=f'direct fit: ${direct_intercept:.4f}+{direct_slope:.4f}/\\lambda$')
ax.plot(lam_grid, combined_intercept + combined_slope / lam_grid, color=BLUE_DARK,
        lw=1.3, ls='--',
        label=f'boundary + bulk: ${combined_intercept:.4f}+{combined_slope:.4f}/\\lambda$')
ax.scatter(sweep_lam, sweep_means, color=RED, s=30, zorder=5)
ax.set_xlabel(r'growth rate $\lambda(q)$')
ax.set_ylabel('total H1 persistence per vertex')
ax.set_title('B.  Two mechanisms reconstruct Finding 1')
ax.legend(fontsize=8)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig('figure_mu_decomposition.png', dpi=180)
print(f'mu(q) fit (all 12):    intercept={intercept_mu:.5f} slope={slope_mu:.5f} R2={r2_mu:.6f}')
print(f'mu(q) fit (q>=7 only): intercept={intercept_no6:.5f} slope={slope_no6:.5f} R2={r2_no6:.6f}')
print(f'hyperbolic-only fit extrapolated to q=6: predicted mu(6)={mu6_hyperbolic_pred:.5f}  '
      f'actual mu(6)={mu[mask6][0]:.5f}  gap={mu6_hyperbolic_pred-mu[mask6][0]:+.5f}')
print(f'combined prediction: {combined_intercept:.5f} + {combined_slope:.5f}/lambda')
print(f'direct fit:          {direct_intercept:.5f} + {direct_slope:.5f}/lambda')
print('saved figure_mu_decomposition.png')
