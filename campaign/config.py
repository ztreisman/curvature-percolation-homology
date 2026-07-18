"""
Campaign configuration for lambda01 (GPU machine).

Larger lattices, more trials, larger SAE, GPU training.
Uses the same pipeline as sae_experiment.py (local test) but at scale.

Run from the project root:
    python campaign/run_campaign.py

Results are checkpointed to campaign/results/ after each q value so the
run can be interrupted and resumed.
"""

# Lattice sizes: much larger rings than local test.
# q=7  rings=8  -> N  ≈ 26,796 vertices
# q=8  rings=7  -> N  ≈ 28,680 vertices
# q=12 rings=5  -> N  ≈ 47,908 vertices  (large due to exp growth)
# q=20 rings=4  -> N  ≈ 18,621 vertices
Q_RINGS = {7: 8, 8: 7, 12: 5, 20: 4}

# p values: ≈ 0.85 p_c for q=12,20 so enough clusters form at subcritical p
P_BY_Q = {7: 0.15, 8: 0.12, 12: 0.085, 20: 0.047}

# More trials for better statistics
N_TRIALS = {7: 50, 8: 40, 12: 60, 20: 50}

MIN_SIZE            = 5       # include smaller clusters for richer statistics
MAX_SIZE            = None    # no upper limit
MAX_CLUSTERS_PER_Q  = 2000    # balance training data across q values

BRW_DIM    = 128    # embedding dimension (larger than local test)
DICT_SIZE  = 1024   # SAE dictionary size
L1_COEF    = 0.02   # sparsity penalty
LR         = 1e-3   # Adam learning rate
N_EPOCHS   = 300    # training epochs
BATCH_SIZE = 1024   # minibatch size
COVERAGE   = 0.90   # feature-splitting threshold

SEED       = 42

RESULTS_DIR = 'campaign/results'
