"""
Figure 3: (left) total H1 persistence per vertex vs lambda(q) with 1/lambda
fit; (right) finite-size convergence for q=6 (amenable) vs q=8,14 (nonamenable).
Outputs: curvature_scaling_law.png

Requires sweep_results.json (produced by full_sweep.py).
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

with open('sweep_results.json') as f:
    results = json.load(f)

by_q = defaultdict(list)
for r in results:
    by_q[r['q']].append(r)
for q in by_q:
    by_q[q].sort(key=lambda r: r['V'])

qs      = sorted(by_q.keys())
best    = {q: by_q[q][-1] for q in qs}
lambdas = np.array([best[q]['lambda'] for q in qs])
means   = np.array([best[q]['mean']   for q in qs])

# 1/lambda fit
X = np.vstack([1 / lambdas, np.ones_like(lambdas)]).T
coef, *_ = np.linalg.lstsq(X, means, rcond=None)
slope, intercept = coef
lam_grid  = np.linspace(1, 17, 300)
fit_curve = intercept + slope / lam_grid

# R^2
pred  = intercept + slope / lambdas
ss_res = np.sum((means - pred)**2)
ss_tot = np.sum((means - means.mean())**2)
r2 = 1 - ss_res / ss_tot

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

# --- left panel: scaling law ---
axes[0].scatter(lambdas, means, color='#9b2c2c', zorder=5, s=40)
axes[0].plot(lam_grid, fit_curve, color='#2b6cb0', lw=1.5, ls='--',
             label=f'${intercept:.4f} + {slope:.4f}/\\lambda$  ($R^2={r2:.4f}$)')
for q in qs:
    axes[0].annotate(f'$q={q}$', (best[q]['lambda'], best[q]['mean']),
                      textcoords='offset points', xytext=(4, 4),
                      fontsize=8, color='#555')
axes[0].set_xlabel(r'growth rate $\lambda(q)$  ($=1$ at $q=6$, flat)')
axes[0].set_ylabel('total H1 persistence per vertex')
axes[0].set_title(r'Loop persistence vs curvature, $\{3,q\}$ family')
axes[0].legend(fontsize=9)

# --- right panel: finite-size convergence ---
for q, color, label in [
        (6,  '#2b6cb0', r'$q=6$ (flat, amenable)'),
        (8,  '#c05621', r'$q=8$'),
        (14, '#1d9e75', r'$q=14$'),
]:
    rs = by_q[q]
    Ns = [r['V']    for r in rs]
    ms = [r['mean'] for r in rs]
    axes[1].plot(Ns, ms, 'o-', color=color, label=label, markersize=5)

axes[1].set_xscale('log')
axes[1].set_xlabel('system size $N$ (vertices)')
axes[1].set_ylabel('total H1 persistence per vertex')
axes[1].set_title('Finite-size convergence: amenable vs nonamenable')
axes[1].legend(fontsize=9)

fig.tight_layout()
fig.savefig('/mnt/user-data/outputs/curvature_scaling_law.png', dpi=160)
print(f'saved  slope={slope:.4f}  intercept={intercept:.4f}  R2={r2:.4f}')
