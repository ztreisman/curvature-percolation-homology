"""
Publication figure for the lambda01 campaign's SAE feature-splitting result.

The naive version of this figure (mean +/- raw per-cluster std) makes a highly
significant, monotone effect look like noise: per-cluster scores are noisy
(std ~1.2-1.4), but at n=2000 clusters/q the standard error of the mean is
~0.03, so the q=7 vs q=20 difference is significant at p=3e-11 (Welch t-test)
despite a small effect size. Plotting raw std conflates "spread of individual
measurements" with "uncertainty in the mean" and visually erases that.

This figure separates the two explicitly:
  (A) per-cluster distributions (violin + jittered strip) -- the honest spread
  (B) mean +/- 95% CI vs q -- the trend, at the precision it was measured
  (C) per-q mean feature-splitting vs H1 persistence/vertex -- the direct
      quantitative link (Pearson r, aggregated over the 4 q values)

Usage:
    python campaign/figure_feature_splitting_campaign.py
"""
import json
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE = '#2a78d6'
BLUE_DARK = '#134a8f'
MUTED = '#9a9a94'
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


def main(summary_path='campaign/results/summary.json',
         out_path='campaign/results/sae_feature_splitting_campaign.png'):
    with open(summary_path) as f:
        d = json.load(f)
    qs = sorted(d, key=int)
    q_int = [int(q) for q in qs]
    scores = {q: np.array(d[q]['scores']) for q in qs}
    means = np.array([scores[q].mean() for q in qs])
    sems = np.array([scores[q].std(ddof=1) / np.sqrt(len(scores[q])) for q in qs])
    ci95 = 1.96 * sems
    ns = [len(scores[q]) for q in qs]
    h1 = np.array([d[q]['h1_mean'] for q in qs])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))

    # ── Panel A: distributions ────────────────────────────────────────────
    ax = axes[0]
    positions = np.arange(len(qs))
    vparts = ax.violinplot([scores[q] for q in qs], positions=positions,
                            widths=0.75, showmeans=False, showmedians=False,
                            showextrema=False)
    for body in vparts['bodies']:
        body.set_facecolor(MUTED)
        body.set_edgecolor('none')
        body.set_alpha(0.5)
    rng = np.random.default_rng(0)
    for i, q in enumerate(qs):
        x = positions[i] + rng.uniform(-0.12, 0.12, size=len(scores[q]))
        ax.scatter(x, scores[q], s=3, color=BLUE, alpha=0.08, linewidths=0)
    ax.scatter(positions, means, s=45, color=BLUE_DARK, zorder=5,
               edgecolors='white', linewidths=1.2)
    ax.set_xticks(positions)
    ax.set_xticklabels(q_int)
    ax.set_xlabel(r'vertex degree $q$ in $\{3,q\}$')
    ax.set_ylabel('atoms for 90% cluster coverage')
    ax.set_title('A.  Per-cluster distribution (n=2000/q)', loc='left', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # ── Panel B: mean +/- 95% CI ──────────────────────────────────────────
    ax = axes[1]
    ax.errorbar(q_int, means, yerr=ci95, fmt='o-', color=BLUE,
               ecolor=BLUE_DARK, capsize=4, lw=1.5, markersize=7,
               markerfacecolor=BLUE_DARK, markeredgecolor='white')
    for x, y, n in zip(q_int, means, ns):
        ax.annotate(f'n={n}', (x, y), textcoords='offset points',
                   xytext=(0, 10), ha='center', fontsize=7.5, color=TEXT_SECONDARY)
    t, p = stats.ttest_ind(scores['7'], scores['20'], equal_var=False)
    ax.annotate(f'q=7 vs q=20\nWelch p={p:.1e}', xy=(0.97, 0.92),
               xycoords='axes fraction', ha='right', va='top', fontsize=8.5,
               color=TEXT_SECONDARY)
    ax.set_xlabel(r'vertex degree $q$ in $\{3,q\}$')
    ax.set_ylabel('mean atoms for 90% coverage (95% CI)')
    ax.set_title('B.  Mean feature-splitting score', loc='left', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # ── Panel C: mean feature-splitting vs H1/vertex ─────────────────────
    ax = axes[2]
    ax.errorbar(h1, means, yerr=ci95, fmt='o', color=BLUE_DARK,
               ecolor=BLUE_DARK, capsize=4, markersize=8)
    r, p = stats.pearsonr(h1, means)
    slope, intercept = np.polyfit(h1, means, 1)
    xx = np.linspace(h1.min() * 0.9, h1.max() * 1.05, 20)
    ax.plot(xx, slope * xx + intercept, color=BLUE, lw=1.2, ls='--', zorder=1)
    for q, x, y in zip(q_int, h1, means):
        ax.annotate(f'q={q}', (x, y), textcoords='offset points',
                   xytext=(6, 4), fontsize=9, color=TEXT_SECONDARY)
    ax.annotate(f'r={r:.2f}  (p={p:.2f}, n=4)', xy=(0.05, 0.06),
               xycoords='axes fraction', ha='left', va='bottom', fontsize=8.5,
               color=TEXT_SECONDARY)
    ax.set_xlabel(r'$H_1$ persistence per vertex')
    ax.set_ylabel('mean feature-splitting score')
    ax.set_title('C.  Feature-splitting vs loop persistence', loc='left', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
