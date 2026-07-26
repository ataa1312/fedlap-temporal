"""How clustered is the low spectrum of each dataset's cumulative graph?

Eigenvector-level methods (plain Krylov, and eigenvector-by-eigenvector
tracking) need neighbouring eigenvalues to be SEPARATED: an eigenvector's
sensitivity to a perturbation of the matrix scales like 1/gap, and the Krylov
iterations needed to resolve two eigenvalues grow as the relative gap shrinks.
This reports, per dataset, the structural degeneracy (isolated nodes,
components) and the actual gaps of the operator the pipeline consumes —
`calc_eigs_exact_sym`, i.e. after active-subgraph + largest-component
truncation and the near-zero drop.

Report both the INTERNAL crowding of the requested window (which destabilises
individual eigenvectors) and the BOUNDARY gap lambda_k -> lambda_{k+1} (which is
what destabilises the k-dimensional invariant subspace as a whole). The two
windows that matter here: k=50 (`spectral.pe_dim`, the input-PE + probe path)
and k=300 (`spectral.spectral_len`, the f+s smodel / tracking path).

usage: python analysis/probes/spectrum_stats.py [k] [dataset ...]
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

DEFAULT = ["uci", "bitcoin_alpha", "bitcoin_otc", "as733", "reddit_body", "reddit_title"]
_a = sys.argv[1:]
if _a and _a[0].isdigit():
    WIN, TARGETS = int(_a[0]), (_a[1:] or DEFAULT)
else:
    WIN, TARGETS = 50, (_a or DEFAULT)
K = WIN + 10  # solve a few past the window so the boundary gap is measurable

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


print(f"window k={WIN} (pe_dim=50 is the input-PE/probe path; spectral_len=300 is the "
      f"f+s smodel / tracking path)")
print(f"{'dataset':>13s} {'t':>5s} {'|E|':>8s} {'comp':>6s} {'giant':>7s} "
      f"{'lam1':>7s} {'lam_k':>7s} {'med gap':>9s} {'<1e-3':>6s} {'<1e-2':>6s} "
      f"{'rel gap':>9s} {'bnd gap':>9s} {'bnd/med':>8s}")
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
        last = min(WIN, len(w)) - 1
        gaps = np.diff(w[:last + 1])
        span = w[last] - w[0]
        bnd = (w[last + 1] - w[last]) if len(w) > last + 1 else float("nan")
        med = np.median(gaps)
        print(f"{name:>13s} {t:>5d} {len(cum):>8d} {ncomp:>6d} {giant:>7d} "
              f"{w[0]:>7.4f} {w[last]:>7.4f} {med:>9.2e} "
              f"{int((gaps < 1e-3).sum()):>6d} {int((gaps < 1e-2).sum()):>6d} "
              f"{med / max(span, 1e-12):>9.2e} {bnd:>9.2e} {bnd / max(med, 1e-12):>8.2f}",
              flush=True)
