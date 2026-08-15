"""How often does an MRR "negative" turn out to be a true edge of the snapshot
being predicted?

The MRR ranker forbids only the EVALUATED split's positives
(`src/metrics/mrr.py`, matching roland `train_utils.py:230-235`), while the
classification batch forbids the target snapshot's WHOLE edge_index
(`federated_orchestrator.py`). So a true edge of the target snapshot sitting in
the train/val subset is an eligible MRR negative. Under `split: [0.8,0.1,0.1]`
that is ~90% of the snapshot's edges.

This is the model-free tier of the diagnostic: no training, no forward pass.
For each evaluated source u it counts the target-snapshot partners of u that are
NOT in the evaluated split, and turns that into the expected number of a K-draw
that lands on one. That expectation is exactly the additive inflation of the
rank denominator, so it bounds the metric's exposure without running anything.

usage: python analysis/probes/mrr_contamination.py [K] [dataset ...]
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import torch

DEFAULT = ["uci", "reddit_body", "as733"]
_a = sys.argv[1:]
if _a and _a[0].isdigit():
    K, TARGETS = int(_a[0]), (_a[1:] or DEFAULT)
else:
    K, TARGETS = None, (_a or DEFAULT)

sys.argv = ["mrr_contamination", "-c", "config/uci_gru.yaml", "--set",
            "model.data_type=feature", "subgraph.num_subgraphs=1", "wandb.mode=disabled"]
from parser import Parser

p = Parser()
cfg0 = p.load_config(p.parse_args())
import src

src.config = cfg0
from registries import datasets
import src.datasets  # noqa: F401
from src.train.federated_orchestrator import _partition_edges_per_snapshot, _pos_for_split

if K is None:
    K = cfg0["experimental"]["rank_eval_multiplier"]

print(f"K={K} negatives per source; 'contam' = expected draws landing on a true "
      f"edge of the target snapshot outside the evaluated split")
print(f"{'dataset':>13s} {'N':>7s} {'pairs':>6s} {'srcs':>8s} {'contam':>8s} "
      f"{'%ofK':>7s} {'>=1':>7s} {'maxc':>6s}")

for name in TARGETS:
    c = p.load_config(p.parse_args(["-c", f"config/{name}_gru.yaml", "--set",
                                    "model.data_type=feature",
                                    "subgraph.num_subgraphs=1", "wandb.mode=disabled"]))
    src.config = c
    snaps = datasets[name](c)
    N, T = snaps[0].num_nodes, len(snaps)
    split_seed = c["dataset"]["split_seed"]
    if split_seed is None:
        split_seed = c["seed"]
    _partition_edges_per_snapshot(snaps, c["dataset"]["split"], split_seed)

    per_source, n_pairs = [], 0
    for t in range(T - 1):
        tgt = snaps[t + 1]
        pos_test = _pos_for_split(tgt, "test")
        if pos_test.size(1) == 0:
            continue
        n_pairs += 1
        # keys of the evaluated split vs the whole target snapshot
        ev = (pos_test[0].numpy().astype(np.int64) * N
              + pos_test[1].numpy().astype(np.int64))
        allk = (tgt.edge_index[0].numpy().astype(np.int64) * N
                + tgt.edge_index[1].numpy().astype(np.int64))
        # distinct target-snapshot pairs that the evaluated split does not forbid
        contam_k = np.unique(allk[~np.isin(allk, ev)])
        csrc = contam_k // N
        counts = np.bincount(csrc, minlength=N)
        srcs = np.unique(pos_test[0].numpy()).astype(np.int64)
        per_source.append(counts[srcs] * (K / N))

    if not per_source:
        print(f"{name:>13s} {'-':>7s} {'0':>6s}")
        continue
    a = np.concatenate(per_source)
    print(f"{name:>13s} {N:>7d} {n_pairs:>6d} {a.size:>8d} {a.mean():>8.3f} "
          f"{100 * a.mean() / K:>6.3f}% {100 * (a >= 1).mean():>6.1f}% {a.max():>6.2f}")

print("\nRead: 'contam' is the expected number of the K negatives that are in fact "
      "true edges of the\ntarget snapshot. It adds directly to the rank denominator, "
      "so compare it against the\nimplied rank 1/MRR (e.g. MRR 0.10 -> rank ~10).")
