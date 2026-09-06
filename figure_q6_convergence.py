"""
Finite-size convergence figure for q=6, combining the original sweep
(N up to 4921, from sweep_results.json) with the lambda01 extension
(N up to ~3M, from q6_convergence_results.json).

Panel A: H1 persistence per vertex vs N (log-x), with the fitted
         approach-to-asymptote curve M(V) = A - B/V^alpha overlaid.
Panel B: |M(V) - A| vs N on log-log axes -- a power-law approach is a
         straight line here, with slope -alpha.

Usage:
    python figure_q6_convergence.py
"""
import json
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE = '#2a78d6'
BLUE_DARK = '#134a8f'
GRID = '#e3e2dd'
TEXT = '#0b0b0b'
TEXT_SECONDARY = '#52514e'

plt.rcParams.update({
    'font.size': 10,
    'axes.edgecolor': GRID,
    'axes.linewidth': 1.0,
    'text.color': TEXT,
    'axes.labelcolor': TEXT,
    'xtick.color': TEXT_SECONDARY,
    'ytick.color': TEXT_SECONDARY,
})


def model(V, A, B, alpha):
    return A - B / V ** alpha


def main(sweep_path='sweep_results.json',
         extension_path='q6_convergence_results.json',
         out_path='figure_q6_convergence.png'):
    old = json.load(open(sweep_path))
    old_q6 = [r for r in old if r['q'] == 6]
    new_q6 = json.load(open(extension_path))

    pts = [(r['V'], r['mean'], r['std'], r['n_trials']) for r in old_q6] + \
          [(r['V'], r['mean'], r['std'], r['n_trials']) for r in new_q6]
    pts.sort()
    V = np.array([p[0] for p in pts], dtype=float)
    M = np.array([p[1] for p in pts])
    S = np.array([p[2] for p in pts])
    N = np.array([p[3] for p in pts])
    sem = S / np.sqrt(N)
    ci95 = 1.96 * sem

    popt, pcov = curve_fit(model, V, M, p0=[0.19, 1.0, 0.3], maxfev=20000)
    A, B, alpha = popt
    perr = np.sqrt(np.diag(pcov))

    # split old vs new for marker styling
    is_new = V > 4921

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    # ── Panel A ──────────────────────────────────────────────────────────
    ax = axes[0]
    vv = np.logspace(np.log10(V.min()), np.log10(V.max()), 200)
    ax.plot(vv, model(vv, *popt), color=BLUE, lw=1.3, ls='--', zorder=1,
           label=f'fit: A - B/V^a,  A={A:.5f}±{perr[0]:.5f}')
    ax.axhline(A, color=TEXT_SECONDARY, lw=0.8, ls=':', zorder=1)
    ax.errorbar(V[~is_new], M[~is_new], yerr=ci95[~is_new], fmt='o',
               color='#9a9a94', ecolor='#9a9a94', capsize=3, markersize=6,
               label='original sweep (N≤4921)', zorder=3)
    ax.errorbar(V[is_new], M[is_new], yerr=ci95[is_new], fmt='o',
               color=BLUE_DARK, ecolor=BLUE_DARK, capsize=3, markersize=6,
               label='lambda01 extension', zorder=4)
    ax.set_xscale('log')
    ax.set_xlabel('N (vertices)')
    ax.set_ylabel(r'$H_1$ persistence per vertex')
    ax.set_title('A.  q=6 finite-size convergence', loc='left', fontsize=10)
    ax.legend(fontsize=7.5, loc='lower right', frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # ── Panel B ──────────────────────────────────────────────────────────
    ax = axes[1]
    gap = A - M
    ax.plot(V, gap, 'o', color=BLUE_DARK, markersize=6, zorder=3)
    vv2 = np.logspace(np.log10(V.min()), np.log10(V.max()), 50)
    ax.plot(vv2, B / vv2 ** alpha, color=BLUE, lw=1.3, ls='--', zorder=1,
           label=f'B/V^a,  a={alpha:.3f}±{perr[2]:.3f}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('N (vertices)')
    ax.set_ylabel(r'$A - $ mean $H_1$/vertex')
    ax.set_title('B.  Power-law approach to asymptote', loc='left', fontsize=10)
    ax.legend(fontsize=8, loc='upper right', frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(color=GRID, linewidth=0.8, which='both', zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f'Saved {out_path}')
    print(f'A={A:.5f}+/-{perr[0]:.5f}  B={B:.4f}+/-{perr[1]:.4f}  '
          f'alpha={alpha:.4f}+/-{perr[2]:.4f}')


if __name__ == '__main__':
    main()
