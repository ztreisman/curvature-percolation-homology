"""
Figure 3 (final): total H1 persistence per vertex vs lambda(q), fit to the
corrected law using q=6's converged asymptote (see q6_convergence.py /
figure_q6_convergence.py) rather than its raw, non-converged sweep value.
Overwrites curvature_scaling_law.png in place.

Left panel: the scaling law itself.
Right panel: finite-size convergence, amenable (q=6, now to N~3M) vs
nonamenable (q=8, q=14).

Requires sweep_results.json and q6_convergence_results.json.

Usage:
    python figure3_scaling_law_final.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

Q6_ASYMPTOTE = 0.18757

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
means   = np.array([best[q]['mean'] for q in qs])
means[qs.index(6)] = Q6_ASYMPTOTE

X = np.vstack([1 / lambdas, np.ones_like(lambdas)]).T
coef, *_ = np.linalg.lstsq(X, means, rcond=None)
slope, intercept = coef
pred = intercept + slope / lambdas
ss_res = np.sum((means - pred) ** 2)
ss_tot = np.sum((means - means.mean()) ** 2)
r2 = 1 - ss_res / ss_tot

# leave-q6-out: q=6 is the only lambda=1 point, at extreme leverage relative
# to the hyperbolic cluster (lambda in [2.6,18]) -- check it separately
mask6 = np.array([q == 6 for q in qs])
X_no6 = np.vstack([1 / lambdas[~mask6], np.ones_like(lambdas[~mask6])]).T
coef_no6, *_ = np.linalg.lstsq(X_no6, means[~mask6], rcond=None)
slope_no6, intercept_no6 = coef_no6
pred_no6 = intercept_no6 + slope_no6 / lambdas[~mask6]
r2_no6 = 1 - np.sum((means[~mask6]-pred_no6)**2)/np.sum((means[~mask6]-means[~mask6].mean())**2)
q6_hyperbolic_pred = intercept_no6 + slope_no6 * 1.0

lam_grid = np.linspace(1, 17, 300)
fit_curve = intercept + slope / lam_grid
fit_curve_no6 = intercept_no6 + slope_no6 / lam_grid

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

axes[0].scatter(lambdas[~mask6], means[~mask6], color='#9b2c2c', zorder=5, s=40, label='q=7..20')
axes[0].scatter(lambdas[mask6], means[mask6], color='#333', zorder=6, s=55, marker='D',
                label='q=6 (amenable)')
axes[0].plot(lam_grid, fit_curve, color='#2b6cb0', lw=1.2, ls=':',
             label=f'all-12 fit: ${intercept:.4f} + {slope:.4f}/\\lambda$  ($R^2={r2:.4f}$)')
axes[0].plot(lam_grid, fit_curve_no6, color='#134a8f', lw=1.6, ls='--',
             label=f'q$\\geq$7 only: ${intercept_no6:.4f} + {slope_no6:.4f}/\\lambda$  ($R^2={r2_no6:.4f}$)')
for q in qs:
    axes[0].annotate(f'$q={q}$', (best[q]['lambda'], means[qs.index(q)]),
                      textcoords='offset points', xytext=(4, 4),
                      fontsize=8, color='#555')
axes[0].set_xlabel(r'growth rate $\lambda(q)$  ($=1$ at $q=6$, flat)')
axes[0].set_ylabel('total H1 persistence per vertex')
axes[0].set_title(r'Loop persistence vs curvature: q=6 sits off the hyperbolic trend')
axes[0].legend(fontsize=7)
print(f'hyperbolic-only fit extrapolated to q=6: predicted={q6_hyperbolic_pred:.5f}  '
      f'actual={means[mask6][0]:.5f}  gap={q6_hyperbolic_pred-means[mask6][0]:+.5f}')

for q, color, label in [
        (6,  '#2b6cb0', r'$q=6$ (flat, amenable)'),
        (8,  '#c05621', r'$q=8$'),
        (14, '#1d9e75', r'$q=14$'),
]:
    rs = by_q[q]
    Ns = [r['V']    for r in rs]
    ms = [r['mean'] for r in rs]
    axes[1].plot(Ns, ms, 'o-', color=color, label=label, markersize=5)
axes[1].axhline(Q6_ASYMPTOTE, color='#2b6cb0', lw=1.0, ls=':')

axes[1].set_xscale('log')
axes[1].set_xlabel('system size $N$ (vertices)')
axes[1].set_ylabel('total H1 persistence per vertex')
axes[1].set_title('Finite-size convergence: amenable vs nonamenable')
axes[1].legend(fontsize=9)

fig.tight_layout()
fig.savefig('curvature_scaling_law.png', dpi=160)
print(f'intercept={intercept:.5f}  slope={slope:.5f}  R2={r2:.6f}')
print('saved curvature_scaling_law.png')
