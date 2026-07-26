"""Acceptance test for the Chebyshev-filtered solver, before it touches a model.

Three questions, all answered against the exact solver on the same snapshot:
  1. Does it find the right SUBSPACE?  overlap = ||U_exact^T U_cheb||_F^2 / k
     (1.0 = identical span; this is the quantity that matters under clustering,
     since individual vectors are not identifiable at ~1e-3 gaps).
  2. Are the Ritz values right?  max |lambda_cheb - lambda_exact|.
  3. Does it carry structure?  the §10.6 oracle probe: rank true edges above
     non-edges by cosine of the basis rows (exact ~0.97, Arnoldi ~0.53 on uci).
Plus wall-clock, which is the point of the exercise.

usage: python analysis/probes/cheb_validate.py [dataset] [k] [degrees]
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import torch
import scipy.stats as sst

_a = sys.argv[1:]
DATASET = _a[0] if len(_a) > 0 else "uci"
K = int(_a[1]) if len(_a) > 1 else 50
DEGREES = [int(d) for d in _a[2].split(",")] if len(_a) > 2 else [20, 40, 80]

sys.argv = ["cheb_validate", "-c", f"config/{DATASET}_gru.yaml", "--set",
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


def recon_auc(Q, edges, N, rng, n=4000):
    """Oracle probe: do rows of the basis rank real edges above random pairs?"""
    Qn = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)
    ed = np.array(list(edges), dtype=np.int64)
    idx = rng.choice(len(ed), size=min(n, len(ed)), replace=False)
    pos = (Qn[ed[idx, 0]] * Qn[ed[idx, 1]]).sum(1)
    eset = edges
    neg = []
    while len(neg) < len(pos):
        a, b = int(rng.integers(0, N)), int(rng.integers(0, N))
        if a == b or (min(a, b), max(a, b)) in eset:
            continue
        neg.append((a, b))
    neg = np.array(neg)
    negs = (Qn[neg[:, 0]] * Qn[neg[:, 1]]).sum(1)
    s = np.concatenate([pos, negs])
    r = sst.rankdata(s)
    npos = len(pos)
    return (r[:npos].sum() - npos * (npos + 1) / 2) / (npos * len(negs))


snaps = datasets[DATASET](cfg)
N, T = snaps[0].num_nodes, len(snaps)
rng = np.random.default_rng(0)
cum = set()
marks = {int(T * 0.5), T - 2}
print(f"{DATASET}: N={N} T={T}, k={K}")
print(f"{'t':>5s} {'solver':>16s} {'seconds':>9s} {'subspace':>9s} {'max dlam':>10s} "
      f"{'recon AUC':>10s}")
for t in range(T - 1):
    cum |= und(snaps[t].edge_index)
    if t not in marks:
        continue
    a = np.array([x for x, _ in cum], dtype=np.int64)
    b = np.array([y for _, y in cum], dtype=np.int64)
    e = torch.tensor(np.stack([np.r_[a, b], np.r_[b, a]]), dtype=torch.long)
    g = Graph(x=torch.ones(N, 1), edge_index=e, node_ids=torch.arange(N))

    t0 = time.time()
    w_ex, U_ex = g.calc_eigs_exact_sym(K)
    t_ex = time.time() - t0
    w_ex, U_ex = w_ex.numpy(), U_ex.numpy()
    auc_ex = recon_auc(U_ex, cum, N, np.random.default_rng(1))
    print(f"{t:>5d} {'exact':>16s} {t_ex:>9.1f} {1.0:>9.3f} {0.0:>10.2e} {auc_ex:>10.3f}",
          flush=True)

    # Arnoldi estimate, the basis the tracking path consumes today
    t0 = time.time()
    try:
        D_ar, U_ar, _ = g.calc_eignvalues(estimate=True, log=False, spectral_len=K)
        t_ar = time.time() - t0
        U_ar = U_ar.detach().cpu().numpy()[:, :K]
        auc_ar = recon_auc(U_ar, cum, N, np.random.default_rng(1))
        ov = np.linalg.norm(U_ex.T @ U_ar, "fro") ** 2 / K
        print(f"{t:>5d} {'arnoldi':>16s} {t_ar:>9.1f} {ov:>9.3f} {float('nan'):>10.2e} "
              f"{auc_ar:>10.3f}", flush=True)
    except Exception as exc:
        print(f"{t:>5d} {'arnoldi':>16s} — failed: {exc!r}", flush=True)

    cut = float(w_ex[w_ex > 0][min(K, (w_ex > 0).sum()) - 1])
    for d in DEGREES:
        t0 = time.time()
        w_cb, U_cb = g.calc_eigs_chebyshev(K, cutoff=0.9 * cut, degree=d)
        t_cb = time.time() - t0
        w_cb, U_cb = w_cb.numpy(), U_cb.numpy()
        ov = np.linalg.norm(U_ex.T @ U_cb, "fro") ** 2 / K
        dlam = np.max(np.abs(w_cb - w_ex))
        auc_cb = recon_auc(U_cb, cum, N, np.random.default_rng(1))
        print(f"{t:>5d} {f'cheb d={d}':>16s} {t_cb:>9.1f} {ov:>9.3f} {dlam:>10.2e} "
              f"{auc_cb:>10.3f}", flush=True)
