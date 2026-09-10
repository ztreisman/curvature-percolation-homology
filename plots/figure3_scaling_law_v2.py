"""
Figure 3, corrected: re-fits the 1/lambda(q) scaling law from Finding 1 using
the CONVERGED q=6 asymptote (from q6_convergence.py / figure_q6_convergence.py,
A=0.18757+/-0.00034) instead of the original, non-converged N=4921 sweep point
(mean=0.17885). q=6 is the only lambda=1 point, so it has outsized leverage on
the fitted intercept.

Left panel shows both fits overlaid (old vs corrected) so the shift is visible
directly. Right panel is unchanged from figure3_scaling_law.py.

Requires sweep_results.json and q6_convergence_results.json.

Usage:
    python figure3_scaling_law_v2.py
"""
import os
from _paths import HERE, RESULTS, FIGURES
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

Q6_ASYMPTOTE = 0.18757
Q6_ASYMPTOTE_ERR = 0.00034

with open(os.path.join(RESULTS, 'sweep_results.json')) as f:
    results = json.load(f)

by_q = defaultdict(list)
for r in results:
    by_q[r['q']].append(r)
for q in by_q:
    by_q[q].sort(key=lambda r: r['V'])

qs      = sorted(by_q.keys())
best    = {q: by_q[q][-1] for q in qs}
lambdas = np.array([best[q]['lambda'] for q in qs])
means_old = np.array([best[q]['mean'] for q in qs])

means_new = means_old.copy()
idx6 = qs.index(6)
means_new[idx6] = Q6_ASYMPTOTE


def fit(means):
    X = np.vstack([1 / lambdas, np.ones_like(lambdas)]).T
    coef, *_ = np.linalg.lstsq(X, means, rcond=None)
    slope, intercept = coef
    pred = intercept + slope / lambdas
    ss_res = np.sum((means - pred) ** 2)
    ss_tot = np.sum((means - means.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return slope, intercept, r2


slope_old, int_old, r2_old = fit(means_old)
slope_new, int_new, r2_new = fit(means_new)

lam_grid = np.linspace(1, 17, 300)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

# --- left panel: scaling law, old vs corrected ---
axes[0].scatter(lambdas, means_old, color='#9a9a94', zorder=4, s=34,
                 label='original sweep values')
axes[0].scatter([1.0], [Q6_ASYMPTOTE], color='#9b2c2c', zorder=6, s=55,
                 marker='D', label=f'q=6 converged asymptote ({Q6_ASYMPTOTE:.5f})')
axes[0].plot(lam_grid, int_old + slope_old / lam_grid, color='#9a9a94', lw=1.3,
             ls=':', label=f'old fit: {int_old:.4f} + {slope_old:.4f}/λ  '
                            f'($R^2={r2_old:.4f}$)')
axes[0].plot(lam_grid, int_new + slope_new / lam_grid, color='#2b6cb0', lw=1.6,
             ls='--', label=f'corrected: {int_new:.4f} + {slope_new:.4f}/λ  '
                             f'($R^2={r2_new:.4f}$)')
for q in qs:
    m = Q6_ASYMPTOTE if q == 6 else best[q]['mean']
    axes[0].annotate(f'$q={q}$', (best[q]['lambda'], m),
                      textcoords='offset points', xytext=(4, 4),
                      fontsize=8, color='#555')
axes[0].set_xlabel(r'growth rate $\lambda(q)$  ($=1$ at $q=6$, flat)')
axes[0].set_ylabel('total H1 persistence per vertex')
axes[0].set_title(r'Loop persistence vs curvature (q=6 corrected)')
axes[0].legend(fontsize=7.5, loc='upper right')

# --- right panel: finite-size convergence (unchanged) ---
for q, color, label in [
        (6,  '#2b6cb0', r'$q=6$ (flat, amenable)'),
        (8,  '#c05621', r'$q=8$'),
        (14, '#1d9e75', r'$q=14$'),
]:
    rs = by_q[q]
    Ns = [r['V']    for r in rs]
    ms = [r['mean'] for r in rs]
    axes[1].plot(Ns, ms, 'o-', color=color, label=label, markersize=5)
axes[1].axhline(Q6_ASYMPTOTE, color='#9b2c2c', lw=1.0, ls=':',
                 label='q=6 converged asymptote')

axes[1].set_xscale('log')
axes[1].set_xlabel('system size $N$ (vertices)')
axes[1].set_ylabel('total H1 persistence per vertex')
axes[1].set_title('Finite-size convergence: amenable vs nonamenable')
axes[1].legend(fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(FIGURES, 'curvature_scaling_law_corrected.png'), dpi=160)
print(f'OLD fit:       intercept={int_old:.5f}  slope={slope_old:.5f}  R2={r2_old:.6f}')
print(f'CORRECTED fit: intercept={int_new:.5f}  slope={slope_new:.5f}  R2={r2_new:.6f}')
print(f'Intercept shift: {int_new-int_old:+.5f} ({100*(int_new-int_old)/int_old:+.2f}%)')
print(f'Slope shift:     {slope_new-slope_old:+.5f} ({100*(slope_new-slope_old)/slope_old:+.2f}%)')
print('saved curvature_scaling_law_corrected.png')
