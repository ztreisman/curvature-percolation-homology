"""
Three-panel Hessian spectrum figure for {3,7} embedding capacity wall.

Loads:
  hessian_spring_all.json  — spring-only Hessian for rings 2-5
  hessian_results.json     — full (spring + repulsion) Hessian for rings 2-5

Two distinct findings emerge:

  1. Spring Hessian (repulse_w=0): exactly recovers Maxwell rigidity theory at
     every ring depth.  The number of zero eigenvalues (flat directions of the
     spring energy) equals 3N − 6 − E + 6 for all rings.  The solution manifold
     GROWS with ring depth; this is the standard result.

  2. Full Hessian (repulse_w=1): the library embeddings for rings 4-5 are
     spring-energy minima but are unstable under the full energy.  They
     accumulate vertex overlaps (583 at ring 4, 2909 at ring 5) that give
     rise to negative curvature in the full Hessian.  The number of negative
     eigenvalues (unstable modes) grows dramatically with ring depth — a
     direct signature of the embedding capacity wall.

Usage:
    python figure_hessian_spectrum.py
"""
import os
from _paths import HERE, RESULTS, FIGURES
import json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RING_COLORS = {2: '#2b6cb0', 3: '#276749', 4: '#c05621', 5: '#744210'}


def load(path):
    with open(path) as f:
        data = json.load(f)
    results  = {int(k): v for k, v in data['results'].items()}
    eigvals  = {int(k): np.array(v) for k, v in data['eigvals'].items()}
    return results, eigvals


def panel_spring_spectrum(ax, results_sp, eigvals_sp):
    """Panel 1: log-scale eigenvalue distribution from spring-only Hessian."""
    for r in sorted(eigvals_sp):
        ev    = eigvals_sp[r]
        color = RING_COLORS.get(r, '#666666')
        label = f'ring {r}  (N={results_sp[r]["N"]})'
        pos_ev = ev[ev > 0]
        if len(pos_ev):
            log_min = max(np.log10(pos_ev.min()), -8)
            log_max = np.log10(pos_ev.max())
            bins = np.logspace(log_min, log_max, 70)
            ax.hist(pos_ev, bins=bins, density=True,
                    histtype='step', color=color, lw=1.5, label=label)

    tol = results_sp[next(iter(results_sp))]['tol_zero']
    ax.axvline(tol, color='#c53030', lw=1.0, linestyle=':', alpha=0.8,
               label=f'zero threshold {tol:.0e}')
    ax.set_xscale('log')
    ax.set_xlabel('eigenvalue', fontsize=11)
    ax.set_ylabel('density', fontsize=11)
    ax.set_title('Spring Hessian eigenvalue spectrum\n(all eigenvalues ≥ 0)', fontsize=11)
    ax.legend(fontsize=8, loc='upper left')


def panel_maxwell_zeros(ax, results_sp):
    """Panel 2: Maxwell zero count from spring Hessian — shows solution manifold grows."""
    rs   = sorted(results_sp)
    maxwell_pred = [results_sp[r]['expected_zeros'] for r in rs]
    observed     = [results_sp[r]['zeros'] for r in rs]
    Ns           = [results_sp[r]['N'] for r in rs]

    ax.plot(rs, maxwell_pred, 'o--', color='#718096', lw=1.5, markersize=7,
            label='Maxwell + rigid (3N − 6 − E + 6)')
    ax.plot(rs, observed, 's-', color='#2b6cb0', lw=2, markersize=8,
            label='observed zeros (spring Hessian)')

    for r, m, z in zip(rs, maxwell_pred, observed):
        ax.annotate(str(z), (r, z), textcoords='offset points',
                    xytext=(0, 8), ha='center', fontsize=9, color='#2b6cb0')

    ax.set_xticks(rs)
    ax.set_xticklabels([f'ring {r}\n(N={N})' for r, N in zip(rs, Ns)])
    ax.set_ylabel('eigenvalue count', fontsize=11)
    ax.set_title('Flat directions of spring energy\n(Maxwell rigidity counting)', fontsize=11)
    ax.legend(fontsize=9)


def panel_negative_eigvals(ax, results_full, results_sp):
    """Panel 3: negative eigenvalue count from full Hessian — capacity-wall signal."""
    rs      = sorted(results_full)
    neg     = [results_full[r]['negative'] for r in rs]
    active  = [results_full[r]['n_active_repulsion'] for r in rs]
    Ns      = [results_full[r]['N'] for r in rs]
    colors  = [RING_COLORS.get(r, '#666666') for r in rs]

    ax2 = ax.twinx()
    ax2.bar(rs, active, color='#c6f6d5', alpha=0.5, width=0.4,
            label='active overlaps (d < 1.0, right axis)')
    ax2.set_ylabel('vertex overlap count\n(non-edge pairs within distance 1)',
                   fontsize=10, color='#276749')
    ax2.tick_params(axis='y', labelcolor='#276749')

    ax.plot(rs, neg, 's-', color='#c53030', lw=2, markersize=9, zorder=5,
            label='full-Hessian negative eigenvalues')
    for r, n, c in zip(rs, neg, colors):
        ax.annotate(str(n), (r, n), textcoords='offset points',
                    xytext=(0, 10), ha='center', fontsize=9, color='#c53030')

    ax.set_xticks(rs)
    ax.set_xticklabels([f'ring {r}\n(N={N})' for r, N in zip(rs, Ns)])
    ax.set_ylabel('unstable eigenvalue count\n(full-energy Hessian)', fontsize=10)
    ax.set_title('Capacity-wall signal: unstable modes\n'
                 '(spring minima become repulsion-unstable)', fontsize=11)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=8, loc='upper left')


def make_figure(results_sp, eigvals_sp, results_full,
                out_path=os.path.join(HERE, 'figure_hessian_spectrum.png')):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    panel_spring_spectrum(axes[0], results_sp, eigvals_sp)
    panel_maxwell_zeros(axes[1], results_sp)
    panel_negative_eigvals(axes[2], results_full, results_sp)

    fig.suptitle('{3,7} Hessian spectrum — spring rigidity vs embedding capacity wall',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved {out_path}')


def print_summary(results_sp, results_full):
    print('\nSpring-only Hessian:')
    print(f'{"ring":>5}  {"N":>5}  {"E":>5}  {"Maxwell+6":>9}  '
          f'{"zeros":>6}  {"fraction":>8}')
    for r in sorted(results_sp):
        s = results_sp[r]
        print(f'{r:>5}  {s["N"]:>5}  {s["E"]:>5}  {s["expected_zeros"]:>9}  '
              f'{s["zeros"]:>6}  {s["surviving_fraction"]:>8.3f}')

    print('\nFull Hessian (spring + repulsion):')
    print(f'{"ring":>5}  {"overlaps":>8}  {"negative":>8}  {"zeros":>6}')
    for r in sorted(results_full):
        s = results_full[r]
        print(f'{r:>5}  {s["n_active_repulsion"]:>8}  '
              f'{s["negative"]:>8}  {s["zeros"]:>6}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spring', default=os.path.join(HERE, 'hessian_spring_all.json'))
    parser.add_argument('--full',   default=os.path.join(HERE, 'hessian_results.json'))
    parser.add_argument('--out',    default=os.path.join(HERE, 'figure_hessian_spectrum.png'))
    args = parser.parse_args()

    results_sp,   eigvals_sp   = load(args.spring)
    results_full, eigvals_full = load(args.full)

    print_summary(results_sp, results_full)
    make_figure(results_sp, eigvals_sp, results_full, out_path=args.out)
