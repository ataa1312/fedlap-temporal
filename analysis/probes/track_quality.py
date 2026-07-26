"""Does the basis need TRACKING, and does warm-starting preserve its quality?

Between consecutive snapshots the graph changes a little. Under a clustered
spectrum (§10.12: ~1e-3 gaps) that small change can rotate individual
eigenvectors arbitrarily while leaving the low-frequency SUBSPACE almost
untouched. This measures which of the two happens, and whether a warm-started
Chebyshev solve lands on the same subspace an independent exact solve finds.

Per consecutive pair (t, t+1), against the exact solver as ground truth:
  drift_sub   overlap(span U_t, span U_{t+1})       — is the subspace stable?
  drift_vec   mean_i |<u_i(t), u_i(t+1)>|           — are the vectors stable?
  proc_gain   ||U_{t+1} - U_t||_F vs after Procrustes alignment — how much of
              the apparent movement is pure gauge (rotation), i.e. removable
  warm_vs_ex  overlap(warm-started cheb at t+1, exact at t+1)  — quality kept?
  cold_vs_ex  overlap(cold cheb at t+1, exact at t+1)
  arno_vs_ex  overlap(arnoldi at t+1, exact at t+1)   — the current tracker input

Drift is only comparable across datasets at the SAME stride: with `stride` left
to spread over the whole run, one reddit step covers 35 snapshots (~25k new
edges) against uci's 2, and the drift numbers are then measuring different
amounts of graph change. Pass `stride`/`start` explicitly to compare
consecutive snapshots on a large dataset without paying for a full pass.

usage: python analysis/probes/track_quality.py [dataset] [k] [max_pairs] [stride] [start]
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

_a = sys.argv[1:]
DATASET = _a[0] if len(_a) > 0 else "uci"
K = int(_a[1]) if len(_a) > 1 else 50
MAX_PAIRS = int(_a[2]) if len(_a) > 2 else 12
STRIDE = int(_a[3]) if len(_a) > 3 else 0      # 0 = spread evenly over the run
START = int(_a[4]) if len(_a) > 4 else 0

sys.argv = ["track_quality", "-c", f"config/{DATASET}_gru.yaml", "--set",
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


def overlap(A, B):
    k = min(A.shape[1], B.shape[1])
    return float(np.linalg.norm(A[:, :k].T @ B[:, :k], "fro") ** 2 / k)


def build(cum, N):
    a = np.array([x for x, _ in cum], dtype=np.int64)
    b = np.array([y for _, y in cum], dtype=np.int64)
    e = torch.tensor(np.stack([np.r_[a, b], np.r_[b, a]]), dtype=torch.long)
    return Graph(x=torch.ones(N, 1), edge_index=e, node_ids=torch.arange(N))


snaps = datasets[DATASET](cfg)
N, T = snaps[0].num_nodes, len(snaps)
step = STRIDE if STRIDE > 0 else max(1, (T - 2) // MAX_PAIRS)
stop = (START + step * (MAX_PAIRS + 1)) if START else (T - 1)
cum = set()
prev = None  # (t, U_exact, lam_k)
print(f"{DATASET}: N={N} T={T} k={K}, every {step} snapshots"
      + (f" from t={START}" if START else ""))
print(f"{'t':>5s} {'dEdges':>7s} {'drift_sub':>10s} {'drift_vec':>10s} {'proc_gain':>10s} "
      f"{'warm_vs_ex':>11s} {'cold_vs_ex':>11s} {'arno_vs_ex':>11s} {'s_warm':>7s} {'s_ex':>7s}")
for t in range(T - 1):
    cum |= und(snaps[t].edge_index)
    if t < START or t > stop:
        continue
    if (t - START) % step or t == 0:
        continue
    g = build(cum, N)
    w_ex, U_ex = g.calc_eigs_exact_sym(K)
    t0 = time.time()
    w_ex2, U_ex2 = g.calc_eigs_exact_sym(K)
    s_ex = time.time() - t0
    w_ex, U_ex = w_ex.numpy(), U_ex.numpy()
    lam_k = float(w_ex[w_ex > 0][-1]) if (w_ex > 0).any() else 0.5

    if prev is not None:
        pt, U_prev, lam_prev = prev
        d_sub = overlap(U_prev, U_ex)
        d_vec = float(np.mean(np.abs((U_prev * U_ex).sum(0))))
        raw = float(np.linalg.norm(U_ex - U_prev))
        # Procrustes: best orthogonal R aligning the new basis to the old one
        M = U_ex.T @ U_prev
        Um, _, Vt = np.linalg.svd(M, full_matrices=False)
        aligned = float(np.linalg.norm(U_ex @ (Um @ Vt) - U_prev))
        proc_gain = aligned / max(raw, 1e-12)

        t0 = time.time()
        _, U_warm = g.calc_eigs_chebyshev(K, cutoff=0.9 * lam_prev, X0=U_prev)
        s_warm = time.time() - t0
        _, U_cold = g.calc_eigs_chebyshev(K, cutoff=0.9 * lam_k)
        try:
            _, U_ar, _ = g.calc_eignvalues(estimate=True, log=False, spectral_len=K)
            a_ov = overlap(U_ar.detach().cpu().numpy()[:, :K], U_ex)
        except Exception:
            a_ov = float("nan")
        print(f"{t:>5d} {len(cum) - pt:>7d} {d_sub:>10.3f} {d_vec:>10.3f} {proc_gain:>10.3f} "
              f"{overlap(U_warm.numpy(), U_ex):>11.3f} {overlap(U_cold.numpy(), U_ex):>11.3f} "
              f"{a_ov:>11.3f} {s_warm:>7.2f} {s_ex:>7.2f}", flush=True)
    prev = (len(cum), U_ex, lam_k)
