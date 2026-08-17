"""Is the arnoldi basis near-chance because of the SOLVER, or because of the OPERATOR?

Every published arnoldi-vs-chebyshev comparison changed two things at once. The
exact and chebyshev solvers build L_sym on the giant component; the Krylov path
builds whatever spectral.L_type says (historically 'rw') over ALL nodes. So
"arnoldi scores at chance" could be the Krylov estimate, the random-walk
operator, the missing truncation, or any combination.

Note this is NOT a 2x2: calc_eigs_exact_sym goes through _active_lsym() and
ignores L_type entirely, so there is no "exact on rw" cell to run. The
comparison is three cells:

    exact   (L_sym, giant component)   -- the reference, overlap 1.0 by construction
    arnoldi on L_rw                    -- what every historical null actually used
    arnoldi on L_sym                   -- the operator changed, solver held fixed

Judged on basis quality, not MRR: reconstruction AUC is model-free, whereas MRR
carries a fixed-seed spread of 0.008-0.014 (results.md 12b) that would hide a
real difference.

usage: python analysis/probes/operator_confound.py [dataset] [k]
"""
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import scipy.stats as sst
import torch

_a = sys.argv[1:]
DATASET = _a[0] if _a else "uci"
K = int(_a[1]) if len(_a) > 1 else 50

sys.argv = ["operator_confound", "-c", f"config/{DATASET}_gru.yaml", "--set",
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
    """Do rows of the basis rank real edges above random pairs? Model-free."""
    Qn = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)
    ed = np.array(list(edges), dtype=np.int64)
    idx = rng.choice(len(ed), size=min(n, len(ed)), replace=False)
    pos = (Qn[ed[idx, 0]] * Qn[ed[idx, 1]]).sum(1)
    neg = []
    while len(neg) < len(pos):
        a, b = int(rng.integers(0, N)), int(rng.integers(0, N))
        if a != b and (min(a, b), max(a, b)) not in edges:
            neg.append((a, b))
    neg = np.array(neg)
    negs = (Qn[neg[:, 0]] * Qn[neg[:, 1]]).sum(1)
    s = np.concatenate([pos, negs])
    r = sst.rankdata(s)
    npos = len(pos)
    return (r[:npos].sum() - npos * (npos + 1) / 2) / (npos * len(negs))


snaps = datasets[DATASET](cfg)
N, T = snaps[0].num_nodes, len(snaps)
marks = {int(T * 0.5), T - 2}

print(f"dataset={DATASET}  N={N}  T={T}  k={K}")
print(f"{'t':>5} {'cell':>18} {'sec':>7} {'overlap':>8} {'recon AUC':>10} {'asym?':>6} {'max|Im|':>9}")

cum = set()
for t in range(T - 1):
    cum |= und(snaps[t].edge_index)
    if t not in marks:
        continue
    a = np.array([x for x, _ in cum], dtype=np.int64)
    b = np.array([y for _, y in cum], dtype=np.int64)
    e = torch.tensor(np.stack([np.r_[a, b], np.r_[b, a]]), dtype=torch.long)

    def fresh():
        return Graph(x=torch.ones(N, 1), edge_index=e, node_ids=torch.arange(N))

    # reference: exact L_sym on the giant component (ignores L_type by construction)
    t0 = time.time()
    w_ex, U_ex = fresh().calc_eigs_exact_sym(K)
    t_ex = time.time() - t0
    U_ex = U_ex.numpy()
    auc_ex = recon_auc(U_ex, cum, N, np.random.default_rng(1))
    print(f"{t:>5d} {'exact (L_sym)':>18} {t_ex:>7.1f} {1.000:>8.3f} {auc_ex:>10.3f} "
          f"{'-':>6} {'-':>9}")

    # the Krylov estimate, once per operator
    for lt in ("rw", "sym"):
        cfg["spectral"]["L_type"] = lt
        src.config = cfg
        g = fresh()
        t0 = time.time()
        try:
            _, U_ar, _ = g.calc_eignvalues(estimate=True, log=False, spectral_len=K)
            dt = time.time() - t0
            U_ar = U_ar.detach().cpu().numpy()[:, :K]
            ov = np.linalg.norm(U_ex.T @ U_ar, "fro") ** 2 / K
            auc = recon_auc(U_ar, cum, N, np.random.default_rng(1))
            # is the operator itself symmetric? that is what decides whether the
            # tracker's eigh assumption holds downstream
            L = g.L.to_dense() if hasattr(g.L, "to_dense") else torch.as_tensor(g.L)
            asym = not bool(torch.allclose(L, L.T, atol=1e-6))
            mx = float((L - L.T).abs().max())
            print(f"{t:>5d} {'arnoldi (L_' + lt + ')':>18} {dt:>7.1f} {ov:>8.3f} {auc:>10.3f} "
                  f"{str(asym):>6} {mx:>9.2e}")
        except Exception as exc:
            print(f"{t:>5d} {'arnoldi (L_' + lt + ')':>18} failed: {exc!r}")

print("\noverlap is against the exact L_sym basis; 1.000 means the same subspace.")
print("recon AUC 0.5 = chance. asym? is whether the built operator is non-symmetric,")
print("which is what makes the tracker's Rayleigh-Ritz step invalid downstream.")
