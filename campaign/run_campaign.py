"""
Campaign runner for lambda01.

Checkpoints after each q value: if interrupted, re-running this script
skips any q already in campaign/results/ and continues from where it left off.

Run from project root:
    python campaign/run_campaign.py

Expects GPU (CUDA) available; falls back to CPU with a warning.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, time, pathlib
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from generate_clusters import generate_clusters
from brw_embedding import embed_clusters
from sae import train_sae, get_activations
from feature_splitting import feature_splitting_scores
import campaign.config as cfg

RESULTS_DIR = pathlib.Path(cfg.RESULTS_DIR)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def checkpoint_path(q):
    return RESULTS_DIR / f'q{q:02d}.json'


def load_h1_reference(path='sweep_results.json'):
    try:
        with open(path) as f:
            sweep = json.load(f)
    except FileNotFoundError:
        return {}
    h1 = {}
    for q in cfg.Q_RINGS:
        entries = [r for r in sweep if r['q'] == q]
        if entries:
            best = max(entries, key=lambda r: r['V'])
            h1[q] = {'mean': best['mean'], 'std': best['std'], 'N': best['V']}
    return h1


def run_one_q(q, device, h1_ref):
    rings    = cfg.Q_RINGS[q]
    p        = cfg.P_BY_Q[q]
    n_trials = cfg.N_TRIALS[q]

    print(f"\n{'='*60}")
    print(f"q={q}  rings={rings}  p={p}  n_trials={n_trials}  device={device}")

    t0 = time.time()
    clusters, meta = generate_clusters(
        q, rings, p=p, n_trials=n_trials,
        min_size=cfg.MIN_SIZE, max_size=cfg.MAX_SIZE,
        seed=cfg.SEED)
    print(f"  {meta['n_clusters']} clusters  N={meta['N']}  "
          f"({time.time()-t0:.1f}s)")
    if meta['n_clusters'] < 10:
        print("  *** too few clusters — skipping ***")
        return None

    cap = getattr(cfg, 'MAX_CLUSTERS_PER_Q', None)
    if cap is not None and len(clusters) > cap:
        rng_sub = np.random.default_rng(cfg.SEED + 1)
        idx = rng_sub.choice(len(clusters), cap, replace=False)
        clusters = [clusters[i] for i in idx]
        print(f"  subsampled to {len(clusters)} clusters")

    sizes_list = [c.number_of_nodes() for c in clusters]
    print(f"  sizes: min={min(sizes_list)} mean={np.mean(sizes_list):.1f} "
          f"max={max(sizes_list)}")

    t0 = time.time()
    X, labels, sizes, incon = embed_clusters(
        clusters, d=cfg.BRW_DIM, seed=cfg.SEED)
    print(f"  embeddings X={X.shape}  incon={np.mean(incon):.4f}  "
          f"({time.time()-t0:.1f}s)")

    mu    = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + 1e-8
    X_norm = (X - mu) / sigma

    t0 = time.time()
    model = train_sae(
        X_norm, cfg.DICT_SIZE,
        l1_coef=cfg.L1_COEF, lr=cfg.LR,
        n_epochs=cfg.N_EPOCHS, batch_size=cfg.BATCH_SIZE,
        device=device, verbose=True)
    print(f"  SAE trained  ({time.time()-t0:.1f}s)")

    torch.save(model.state_dict(),
               str(RESULTS_DIR / f'sae_q{q:02d}.pt'))

    H = get_activations(model, X_norm, device=device)
    scores = feature_splitting_scores(
        H, labels, len(clusters), threshold=cfg.COVERAGE)
    used_atoms = int(len(np.unique(H.argmax(axis=1))))

    result = dict(
        q=q, rings=rings, p=p, N=meta['N'],
        n_clusters=len(clusters),
        mean_size=float(np.mean(sizes_list)),
        mean_feature_splitting=float(scores.mean()),
        std_feature_splitting=float(scores.std()),
        mean_cycle_inconsistency=float(np.mean(incon)),
        atoms_used=used_atoms,
        h1_mean=h1_ref.get(q, {}).get('mean'),
        h1_std=h1_ref.get(q, {}).get('std'),
        scores=scores.tolist(),
    )
    print(f"  feature-splitting: {result['mean_feature_splitting']:.3f} ± "
          f"{result['std_feature_splitting']:.3f}  "
          f"atoms_used={used_atoms}/{cfg.DICT_SIZE}")
    return result


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("WARNING: no GPU found; running on CPU (will be slow)")

    h1_ref = load_h1_reference()
    results = {}

    for q in sorted(cfg.Q_RINGS.keys()):
        cp = checkpoint_path(q)
        if cp.exists():
            print(f"q={q}: checkpoint found, loading")
            with open(cp) as f:
                results[q] = json.load(f)
            continue

        r = run_one_q(q, device, h1_ref)
        if r is not None:
            results[q] = r
            with open(cp, 'w') as f:
                json.dump(r, f, indent=2)

    # ── aggregate plot ────────────────────────────────────────────────────────
    qs   = sorted(results.keys())
    fs   = [results[q]['mean_feature_splitting'] for q in qs]
    fs_e = [results[q]['std_feature_splitting'] for q in qs]
    h1   = [results[q].get('h1_mean') for q in qs]
    h1_e = [results[q].get('h1_std') for q in qs]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].errorbar(qs, fs, yerr=fs_e, fmt='o-', color='#2b6cb0',
                     capsize=4, lw=1.5)
    axes[0].set_xlabel(r'vertex degree $q$  in  $\{3,q\}$')
    axes[0].set_ylabel('mean atoms for 90% cluster coverage')
    axes[0].set_title('Feature-splitting score vs curvature  [campaign]')
    axes[0].set_xticks(qs)

    if any(v is not None for v in h1):
        valid = [(h, f, fe, q)
                 for h, f, fe, q in zip(h1, fs, fs_e, qs)
                 if h is not None]
        hv, fv, fev, qv = zip(*valid)
        axes[1].errorbar(hv, fv, yerr=fev, fmt='o', color='#c05621',
                         capsize=4, markersize=8)
        for q, h, f in zip(qv, hv, fv):
            axes[1].annotate(f'q={q}', (h, f),
                             textcoords='offset points', xytext=(5, 3),
                             fontsize=9)
        axes[1].set_xlabel('H1 persistence per vertex (sweep)')
        axes[1].set_ylabel('mean feature-splitting score')
        axes[1].set_title('Feature splitting vs loop persistence  [campaign]')

    fig.tight_layout()
    out = str(RESULTS_DIR / 'sae_feature_splitting_campaign.png')
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")

    summary_path = str(RESULTS_DIR / 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump({str(q): v for q, v in results.items()}, f, indent=2)
    print(f"Saved {summary_path}")


if __name__ == '__main__':
    main()
