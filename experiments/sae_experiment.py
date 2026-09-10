"""
SAE feature-splitting experiment.

For each q: generate subcritical bond-percolation clusters, BRW-embed node
positions, train a sparse autoencoder on the pooled node embeddings, compute
the feature-splitting score per cluster (min atoms for 90% argmax coverage),
and compare the mean score against the H1 persistence per vertex already
measured in sweep_results.json.

Run locally with small parameters to verify the pipeline; see campaign/ for
the large-scale GPU run to bring to the Wednesday call with Brill.

Usage:
    python sae_experiment.py [--fast]

--fast: cuts ring depth, trials, and epochs for a quick smoke-test.
"""
import os
from _paths import HERE, RESULTS, FIGURES
import argparse
import json
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from generate_clusters import generate_clusters
from brw_embedding import embed_clusters
from sae import train_sae, get_activations
from feature_splitting import feature_splitting_scores

# ── configuration ─────────────────────────────────────────────────────────────

FULL = dict(
    # q=8 bumped to rings=5 (N=2281) to match q=7's lattice scale.
    # q=12 and q=20 use higher p (≈85% of p_c) so more clusters form.
    q_rings     = {7: 5, 8: 5, 12: 3, 20: 3},
    p_by_q      = {7: 0.15, 8: 0.12, 12: 0.085, 20: 0.047},
    n_trials    = {7: 60, 8: 40, 12: 60, 20: 40},
    min_size            = 5,
    max_clusters_per_q  = 300,   # balance training data across q values
    brw_dim             = 64,
    dict_size           = 256,
    l1_coef             = 0.02,
    lr                  = 1e-3,
    n_epochs            = 150,
    batch_size          = 512,
    coverage            = 0.90,
    seed                = 42,
)

FAST = dict(
    # q=20 uses more trials to compensate for the tiny (rings=2) lattice
    q_rings     = {7: 4, 8: 4, 12: 3, 20: 2},
    p_by_q      = {7: 0.15, 8: 0.12, 12: 0.075, 20: 0.040},
    n_trials    = {7: 20, 8: 20, 12: 20, 20: 200},
    min_size    = 5,
    brw_dim     = 32,
    dict_size   = 128,
    l1_coef     = 0.02,
    lr          = 1e-3,
    n_epochs    = 50,
    batch_size  = 256,
    coverage    = 0.90,
    seed        = 42,
)

# ── H1 reference from sweep ───────────────────────────────────────────────────

def load_h1_reference(path=os.path.join(RESULTS, 'sweep_results.json')):
    with open(path) as f:
        sweep = json.load(f)
    h1 = {}
    for q in [7, 8, 12, 20]:
        entries = [r for r in sweep if r['q'] == q]
        if entries:
            best = max(entries, key=lambda r: r['V'])
            h1[q] = {'mean': best['mean'], 'std': best['std'], 'N': best['V']}
    return h1

# ── main ──────────────────────────────────────────────────────────────────────

def run_experiment(cfg):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")

    h1_ref = load_h1_reference()
    results = {}
    t_global = time.time()

    for q, rings in sorted(cfg['q_rings'].items()):
        p         = cfg['p_by_q'][q]
        n_trials  = cfg['n_trials'][q]
        print(f"\n{'='*60}")
        print(f"q={q}  rings={rings}  p={p}  n_trials={n_trials}")

        # 1. Generate clusters
        t0 = time.time()
        clusters, meta = generate_clusters(
            q, rings, p=p, n_trials=n_trials,
            min_size=cfg['min_size'], seed=cfg['seed'])
        print(f"  {meta['n_clusters']} clusters  "
              f"(lattice N={meta['N']}, {time.time()-t0:.1f}s)")
        if meta['n_clusters'] < 10:
            print("  *** too few clusters; skipping ***")
            continue

        # Subsample to balance training data across q values.
        # Without this, q=7/8 (many clusters) converge to a tighter SAE basis
        # than q=12/20, conflating SAE convergence quality with topology signal.
        cap = cfg.get('max_clusters_per_q')
        if cap is not None and len(clusters) > cap:
            rng_sub = np.random.default_rng(cfg['seed'] + 1)
            idx = rng_sub.choice(len(clusters), cap, replace=False)
            clusters = [clusters[i] for i in idx]
            print(f"  subsampled to {len(clusters)} clusters")

        sizes = [c.number_of_nodes() for c in clusters]
        print(f"  cluster sizes: min={min(sizes)} mean={np.mean(sizes):.1f} "
              f"max={max(sizes)}")

        # 2. BRW embed
        t0 = time.time()
        X, labels, sizes, inconsistencies = embed_clusters(
            clusters, d=cfg['brw_dim'], seed=cfg['seed'])
        print(f"  embeddings: X={X.shape}  "
              f"cycle_inconsistency={np.mean(inconsistencies):.4f}  "
              f"({time.time()-t0:.1f}s)")

        # 3. Normalize embeddings (per-feature mean/std)
        mu    = X.mean(axis=0, keepdims=True)
        sigma = X.std(axis=0, keepdims=True) + 1e-8
        X_norm = (X - mu) / sigma

        # 4. Train SAE
        t0 = time.time()
        print(f"  training SAE  dict_size={cfg['dict_size']}  "
              f"epochs={cfg['n_epochs']}  ...")
        model = train_sae(
            X_norm, cfg['dict_size'],
            l1_coef=cfg['l1_coef'], lr=cfg['lr'],
            n_epochs=cfg['n_epochs'], batch_size=cfg['batch_size'],
            device=device, verbose=True)
        print(f"  SAE trained in {time.time()-t0:.1f}s")

        # 5. Feature splitting scores
        H = get_activations(model, X_norm, device=device)
        scores = feature_splitting_scores(
            H, labels, len(clusters), threshold=cfg['coverage'])
        mean_score = float(scores.mean())
        std_score  = float(scores.std())
        print(f"  feature-splitting: mean={mean_score:.3f} ± {std_score:.3f}")
        print(f"  scores sample (first 20): {scores[:20].tolist()}")

        # 6. SAE utilization (fraction of atoms ever the top-1)
        used_atoms = len(np.unique(H.argmax(axis=1)))
        print(f"  atoms with any argmax winner: {used_atoms}/{cfg['dict_size']}")

        results[q] = dict(
            q=q, rings=rings, p=p, N=meta['N'],
            n_clusters=len(clusters),
            mean_size=float(np.mean(sizes)),
            mean_feature_splitting=mean_score,
            std_feature_splitting=std_score,
            mean_cycle_inconsistency=float(np.mean(inconsistencies)),
            atoms_used=int(used_atoms),
            h1_mean=h1_ref.get(q, {}).get('mean'),
            h1_std=h1_ref.get(q, {}).get('std'),
        )

    print(f"\nTotal elapsed: {time.time()-t_global:.1f}s")

    # ── save JSON results ──────────────────────────────────────────────────────
    with open(os.path.join(RESULTS, 'sae_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved sae_results.json")

    # ── plot ───────────────────────────────────────────────────────────────────
    qs   = sorted(results.keys())
    fs   = [results[q]['mean_feature_splitting'] for q in qs]
    fs_e = [results[q]['std_feature_splitting'] for q in qs]
    h1   = [results[q]['h1_mean'] for q in qs]
    h1_e = [results[q]['h1_std'] for q in qs]
    incon = [results[q]['mean_cycle_inconsistency'] for q in qs]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: feature splitting vs q
    axes[0].errorbar(qs, fs, yerr=fs_e, fmt='o-', color='#2b6cb0',
                     capsize=4, lw=1.5, label='feature splitting')
    axes[0].set_xlabel(r'vertex degree $q$  in  $\{3,q\}$')
    axes[0].set_ylabel('mean atoms for 90% cluster coverage')
    axes[0].set_title('Feature-splitting score vs curvature')
    axes[0].set_xticks(qs)

    # Panel 2: cycle inconsistency vs q (direct loop content proxy)
    axes[1].plot(qs, incon, 'o-', color='#9b2c2c', lw=1.5)
    axes[1].set_xlabel(r'vertex degree $q$  in  $\{3,q\}$')
    axes[1].set_ylabel('mean BRW cycle inconsistency')
    axes[1].set_title('Loop content of percolation clusters\n'
                      '(cycle inconsistency of BRW embedding)')
    axes[1].set_xticks(qs)

    # Panel 3: feature splitting vs H1 persistence
    if all(v is not None for v in h1):
        axes[2].errorbar(h1, fs,
                         xerr=h1_e, yerr=fs_e,
                         fmt='o', color='#c05621',
                         capsize=4, markersize=8)
        for q, h, f in zip(qs, h1, fs):
            axes[2].annotate(f'q={q}', (h, f),
                             textcoords='offset points', xytext=(5, 3),
                             fontsize=9)
        axes[2].set_xlabel('H1 persistence per vertex (sweep)')
        axes[2].set_ylabel('mean feature-splitting score')
        axes[2].set_title('Feature splitting vs loop persistence')

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, 'sae_feature_splitting.png'), dpi=150)
    print("Saved sae_feature_splitting.png")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast', action='store_true',
                        help='Quick smoke-test with reduced parameters')
    args = parser.parse_args()
    cfg = FAST if args.fast else FULL
    print("Config:", 'FAST' if args.fast else 'FULL')
    run_experiment(cfg)
