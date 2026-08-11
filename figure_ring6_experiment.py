"""
figure_ring6_experiment.py

Three-panel figure from ring6_results.json showing the {3,7} embedding
capacity wall via energy trajectories through ring 6 attempts.
"""
import json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── palette (reference slots 1 & 2, light mode) ──────────────────────────────
C_BLUE   = '#2a78d6'   # categorical slot 1 — converged trials
C_ORANGE = '#eb6834'   # categorical slot 2 — failed trials
C_SURF   = '#fcfcfb'   # chart surface
C_GRID   = '#e1e0d9'   # hairline grid
C_AXIS   = '#c3c2b7'   # axis baseline
C_INK    = '#0b0b0b'   # primary text
C_MUTED  = '#898781'   # tick labels, captions

CONVERGE_TOL = 1e-4    # spring energy threshold for "converged"


def load(path):
    with open(path) as f:
        return json.load(f)


def classify_trials(data):
    """
    For each trial, classify each ring as converged or not.
    Returns dict: trial_idx → {ring: {'spring':..., 'repulsion':..., 'converged':bool}}
    """
    trials = {}
    for t in data['trials']:
        idx = t['trial']
        traj = {}
        for r_str, s in t['trajectory'].items():
            r = int(r_str)
            traj[r] = {
                'spring': s['spring_energy'],
                'repulsion': s['repulsion_energy'],
                'n_overlaps': s['n_overlaps'],
                'edge_std': s['edge_std'],
                'N': s['N'],
                'E': s['E'],
                'converged': s['spring_energy'] < CONVERGE_TOL,
            }
        trials[idx] = traj
    return trials


def panel_spring_trajectories(ax, trials):
    """Spaghetti: spring energy vs ring depth, colored by ring-5 convergence."""
    rings = sorted({r for t in trials.values() for r in t})
    max_ring = max(rings)

    # Classify each trial by ring-5 convergence
    r5_converged = {}
    for idx, traj in trials.items():
        r5_converged[idx] = traj.get(5, {}).get('converged', False)

    # Draw individual trial lines (thin, semi-transparent)
    for idx, traj in trials.items():
        rs = sorted(traj)
        ys = [traj[r]['spring'] for r in rs]
        color = C_BLUE if r5_converged[idx] else C_ORANGE
        ax.plot(rs, ys, color=color, lw=0.8, alpha=0.25, zorder=2)

    # Draw median lines per group per ring
    for converged, color, label in [(True, C_BLUE, 'ring-5 converged'),
                                     (False, C_ORANGE, 'ring-5 failed')]:
        group = [idx for idx, c in r5_converged.items() if c == converged]
        if not group:
            continue
        medians = []
        for r in rings:
            vals = [trials[idx][r]['spring'] for idx in group if r in trials[idx]]
            if vals:
                medians.append((r, np.median(vals)))
        rs_m, ys_m = zip(*medians)
        ax.plot(rs_m, ys_m, color=color, lw=2.0, alpha=1.0, zorder=4,
                label=f'{label} (n={len(group)})')

    ax.set_yscale('log')
    ax.set_xticks(rings)
    ax.set_xlabel('ring depth', fontsize=10, color=C_INK)
    ax.set_ylabel('spring energy', fontsize=10, color=C_INK)
    ax.set_title('Spring energy trajectories\n(each line = one trial, bold = median)',
                 fontsize=10, color=C_INK)
    ax.legend(fontsize=8, framealpha=0.9)
    _style_ax(ax)

    # Capacity wall marker — use axis coords for y so log scale doesn't corrupt bbox
    ax.axvline(5.5, color=C_MUTED, lw=1.0, ls='--', zorder=1, alpha=0.6)
    ax.text(5.55, 0.02, 'capacity\nwall →',
            transform=ax.get_xaxis_transform(),
            color=C_MUTED, fontsize=7.5, va='bottom')


def panel_success_rate(ax, trials):
    """Bar chart: success fraction vs ring depth."""
    rings = sorted({r for t in trials.values() for r in t})
    n_total = len(trials)

    fracs = []
    ns = []
    for r in rings:
        n_conv = sum(1 for t in trials.values()
                     if r in t and t[r]['converged'])
        fracs.append(n_conv / n_total)
        ns.append(trials[list(trials.keys())[0]].get(r, {}).get('N', ''))

    # Color bars by convergence: ≥50% blue, <50% orange, 0% muted
    colors = []
    for f in fracs:
        if f == 0.0:
            colors.append(C_MUTED)
        elif f >= 0.5:
            colors.append(C_BLUE)
        else:
            colors.append(C_ORANGE)

    bars = ax.bar(rings, fracs, color=colors, width=0.55, zorder=2,
                  linewidth=0, edgecolor='none')

    # Direct labels on bars
    for bar, f in zip(bars, fracs):
        label = f'{f:.0%}'
        ypos  = bar.get_height() + 0.02
        ax.text(bar.get_x() + bar.get_width() / 2, ypos, label,
                ha='center', va='bottom', fontsize=9, color=C_INK,
                fontweight='bold')

    ax.set_xticks(rings)
    ax.set_xticklabels([f'ring {r}' for r in rings], fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('fraction converged', fontsize=10, color=C_INK)
    ax.set_title(f'Convergence rate (spring < {CONVERGE_TOL:.0e})\nacross {n_total} trials',
                 fontsize=10, color=C_INK)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    _style_ax(ax)


def panel_repulsion(ax, trials):
    """Median + IQR band of repulsion energy and overlap count vs ring depth."""
    rings = sorted({r for t in trials.values() for r in t})

    medians_rep, q25_rep, q75_rep = [], [], []
    medians_ov,  q25_ov,  q75_ov  = [], [], []
    for r in rings:
        vals_rep = [t[r]['repulsion'] for t in trials.values() if r in t]
        vals_ov  = [t[r]['n_overlaps'] for t in trials.values() if r in t]
        medians_rep.append(np.median(vals_rep)); q25_rep.append(np.percentile(vals_rep, 25)); q75_rep.append(np.percentile(vals_rep, 75))
        medians_ov.append(np.median(vals_ov));  q25_ov.append(np.percentile(vals_ov, 25));  q75_ov.append(np.percentile(vals_ov, 75))

    ax.fill_between(rings, q25_rep, q75_rep, color=C_BLUE, alpha=0.15, zorder=1)
    ax.plot(rings, medians_rep, color=C_BLUE, lw=2.0, marker='o', markersize=6,
            zorder=3, label='repulsion energy (median ± IQR)')

    ax2 = ax.twinx()
    ax2.fill_between(rings, q25_ov, q75_ov, color=C_ORANGE, alpha=0.12, zorder=1)
    ax2.plot(rings, medians_ov, color=C_ORANGE, lw=2.0, marker='s', markersize=6,
             zorder=3, label='vertex overlaps (right)')
    ax2.set_ylabel('vertex overlap count', fontsize=9, color=C_ORANGE)
    ax2.tick_params(axis='y', labelcolor=C_ORANGE)

    ax.set_xticks(rings)
    ax.set_xlabel('ring depth', fontsize=10, color=C_INK)
    ax.set_ylabel('repulsion energy', fontsize=10, color=C_BLUE)
    ax.set_title('Packing frustration vs ring depth\n(median ± IQR across 20 trials)',
                 fontsize=10, color=C_INK)

    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc='upper left')
    _style_ax(ax)


def _style_ax(ax):
    ax.set_facecolor(C_SURF)
    ax.tick_params(colors=C_MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(C_AXIS)
        spine.set_linewidth(0.8)
    ax.grid(axis='y', color=C_GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def make_figure(data, out_path='figure_ring6_experiment.png'):
    trials = classify_trials(data)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.patch.set_facecolor(C_SURF)

    panel_spring_trajectories(axes[0], trials)
    panel_success_rate(axes[1], trials)
    panel_repulsion(axes[2], trials)

    fig.suptitle('{3,7} embedding capacity wall — ring-by-ring energy trajectories '
                 f'({data["n_trials"]} trials)',
                 fontsize=12, color=C_INK)
    fig.subplots_adjust(top=0.88, wspace=0.35)
    fig.savefig(out_path, dpi=150, facecolor=C_SURF)
    print(f'Saved {out_path}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='ring6_results.json')
    ap.add_argument('--out',   default='figure_ring6_experiment.png')
    args = ap.parse_args()
    data = load(args.input)
    make_figure(data, args.out)
