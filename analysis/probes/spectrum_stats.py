"""How clustered is the low spectrum of each dataset's cumulative graph?

Eigenvector-level methods (plain Krylov, and eigenvector-by-eigenvector
tracking) need neighbouring eigenvalues to be SEPARATED: an eigenvector's
sensitivity to a perturbation of the matrix scales like 1/gap, and the Krylov
iterations needed to resolve two eigenvalues grow as the relative gap shrinks.
This reports, per dataset, the structural degeneracy (isolated nodes,
components) and the actual gaps of the operator the pipeline consumes —
`calc_eigs_exact_sym`, i.e. after active-subgraph + largest-component
truncation and the near-zero drop.

usage: python analysis/probes/spectrum_stats.py [dataset ...]
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.csgraph import connected_components

K = 60

DEFAULT = ["uci", "bitcoin_alpha", "bitcoin_otc", "as733", "reddit_body", "reddit_title"]
TARGETS = sys.argv[1:] or DEFAULT

sys.argv = ["spectrum_stats", "-c", "config/uci_gru.yaml", "--set",
            "model.data_type=feature", "subgraph.num_subgraphs=1", "wandb.mode=disabled"]
from parser import Parser

p = Parser()
cfg = p.load_config(p.parse_args())
import src

src.config = cfg
from registries import datasets
import src.datasets  # noqa: F401
from src.utils.graph import Graph


def und(ei):
    e = ei.cpu().numpy()
    return {(min(a, b), max(a, b)) for a, b in zip(e[0], e[1]) if a != b}


print(f"{'dataset':>13s} {'t':>5s} {'|E|':>8s} {'iso':>6s} {'comp':>6s} {'giant':>7s} "
      f"{'lam1':>7s} {'lam50':>7s} {'med gap':>9s} {'<1e-3':>6s} {'<1e-2':>6s} {'rel gap':>9s}")
for name in TARGETS:
    c = p.load_config(p.parse_args(["-c", f"config/{name}_gru.yaml", "--set",
                                    "model.data_type=feature",
                                    "subgraph.num_subgraphs=1", "wandb.mode=disabled"]))
    snaps = datasets[name](c)
    N, T = snaps[0].num_nodes, len(snaps)
    cum = set()
    marks = {int(T * 0.5), T - 2}
    for t in range(T - 1):
        cum |= und(snaps[t].edge_index)
        if t not in marks:
            continue
        a = np.array([x for x, _ in cum], dtype=np.int64)
        b = np.array([y for _, y in cum], dtype=np.int64)
        A = sparse.coo_matrix((np.ones(2 * a.size), (np.r_[a, b], np.r_[b, a])),
                              shape=(N, N)).tocsr()
        A.data[:] = 1.0
        deg = np.asarray(A.sum(1)).ravel()
        iso = int((deg == 0).sum())
        ncomp, lab = connected_components(A, directed=False)
        giant = int(np.bincount(lab).max())
        e = torch.tensor(np.stack([np.r_[a, b], np.r_[b, a]]), dtype=torch.long)
        g = Graph(x=torch.ones(N, 1), edge_index=e, node_ids=torch.arange(N))
        w, _ = g.calc_eigs_exact_sym(K)
        w = w.numpy()
        w = w[w > 0]  # drop the zero padding of tiny graphs
        gaps = np.diff(w[:50])
        span = w[min(49, len(w) - 1)] - w[0]
        print(f"{name:>13s} {t:>5d} {len(cum):>8d} {iso:>6d} {ncomp:>6d} {giant:>7d} "
              f"{w[0]:>7.4f} {w[min(49, len(w) - 1)]:>7.4f} {np.median(gaps):>9.2e} "
              f"{int((gaps < 1e-3).sum()):>6d} {int((gaps < 1e-2).sum()):>6d} "
              f"{np.median(gaps) / max(span, 1e-12):>9.2e}", flush=True)
