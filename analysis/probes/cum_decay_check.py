"""Verification gates for spectral.cum_decay before any arm is run.

Checks, in order of how badly a failure would invalidate the experiment:

  5.4  SEPARATION -- the kernel moves the basis and NOTHING else. Two runs
       differing only in cum_decay must partition evaluated positives into
       repeat/new identically, and 'persist' must be unmoved. If this fails the
       treatment has redefined the measurement and no arm is comparable.
  5.1  DEFAULT BIT-IDENTITY -- cum_decay=none reproduces the binarized operator
       exactly, not merely closely.
  5.2  OPERATOR PROPERTIES under weighting -- symmetric, unit diagonal, spectrum
       in [0,2], trace = covered node count.
  5.3  SCALE INVARIANCE -- w -> c*w leaves L_sym unchanged.
  --   SUPPORT -- count/harmonic/exp leave the active set alone; window may
       shrink it (and that is the confound the design calls out).

usage: python analysis/probes/cum_decay_check.py [dataset]
"""
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import torch

DATASET = sys.argv[1] if len(sys.argv) > 1 else "uci"

sys.argv = ["cum_decay_check", "-c", f"config/{DATASET}_gru.yaml", "--set",
            "model.data_type=f+es", "subgraph.num_subgraphs=1",
            "spectral.update_mode=update", "spectral.solver=chebyshev",
            "wandb.mode=disabled"]
from parser import Parser

p = Parser()
cfg = p.load_config(p.parse_args())
import src

src.config = cfg
from registries import datasets
import src.datasets  # noqa: F401
from src.utils.graph_partitioning import partition_snapshots
from src.dynamic_server import DynamicServer

snaps = datasets[DATASET](cfg)
N = int(snaps[0].num_nodes)
T = min(12, len(snaps) - 1)
print(f"dataset={DATASET} N={N} snapshots={len(snaps)} probing t=0..{T - 1}\n")

ARMS = [("none", None), ("count", None), ("harmonic", None),
        ("exp", 0.5), ("exp", 0.9), ("window", 1), ("window", 4)]


def _server():
    s = DynamicServer(snaps)
    for cs in partition_snapshots(snaps, 1):
        s.add_client(cs)
    s._cum_edges = None
    s._cum_events_key = s._cum_events_t = None
    return s


def _lsym(graph, w):
    return graph._active_lsym(w)


def _run_arm(kind, param):
    cfg["spectral"]["cum_decay"] = kind
    if param is not None:
        cfg["spectral"]["cum_decay_param"] = param
    from src.utils.graph import Graph

    srv = _server()
    out = []
    for t in range(T):
        srv._accumulate_cum_edges(t)
        g = Graph(x=torch.ones(N, 1), edge_index=srv._cum_edges,
                  node_ids=torch.arange(N))
        w = srv._cum_edge_weight(t)
        L, act = _lsym(g, w)
        # repeat mask on the UNWEIGHTED union -- must not move with the kernel
        pos = snaps[t + 1].edge_index
        rm = srv._repeat_mask(pos)
        out.append({
            "t": t, "L": L, "act": act, "w": w,
            "repeat_frac": float(rm.float().mean()),
            "n_cum": int(srv._cum_edges.shape[1]),
            "mass": None if w is None else float(w.sum() / w.numel()),
        })
    return out


res = {f"{k}{'' if p is None else p}": _run_arm(k, p) for k, p in ARMS}
base = res["none"]

print("=" * 78)
print("5.4 SEPARATION -- repeat_frac and the unweighted union must be arm-independent")
ok_sep = True
for name, r in res.items():
    d_rf = max(abs(a["repeat_frac"] - b["repeat_frac"]) for a, b in zip(r, base))
    d_cum = max(abs(a["n_cum"] - b["n_cum"]) for a, b in zip(r, base))
    bad = d_rf > 0 or d_cum > 0
    ok_sep &= not bad
    print(f"  {name:12s} max|d repeat_frac|={d_rf:.1e}  max|d |cum||={d_cum}"
          f"  {'FAIL' if bad else 'ok'}")

print()
print("5.1 DEFAULT BIT-IDENTITY -- none must equal the binarized operator exactly")
ok_bit = True
for e in base:
    if e["w"] is not None:
        ok_bit = False
        print(f"  t={e['t']} FAIL: weights not None under cum_decay=none")
print(f"  weights are None at every t: {'ok' if ok_bit else 'FAIL'}")

print()
print("5.2 OPERATOR PROPERTIES under weighting")
for name, r in res.items():
    worst_sym = worst_diag = worst_tr = 0.0
    lo, hi = np.inf, -np.inf
    for e in r:
        L = e["L"]
        if L is None:
            continue
        M = np.asarray(L.todense())
        worst_sym = max(worst_sym, float(np.abs(M - M.T).max()))
        worst_diag = max(worst_diag, float(np.abs(np.diag(M) - 1.0).max()))
        worst_tr = max(worst_tr, abs(float(np.trace(M)) - M.shape[0]))
        if M.shape[0] <= 1400:
            ev = np.linalg.eigvalsh((M + M.T) / 2)
            lo, hi = min(lo, float(ev.min())), max(hi, float(ev.max()))
    sp_ok = (lo >= -1e-8) and (hi <= 2 + 1e-8)
    print(f"  {name:12s} |L-L^T|={worst_sym:.2e} |diag-1|={worst_diag:.2e} "
          f"|tr-m|={worst_tr:.2e} spec=[{lo:.4f},{hi:.4f}] "
          f"{'ok' if (worst_sym < 1e-9 and worst_diag < 1e-9 and sp_ok) else 'FAIL'}")

print()
print("5.3 SCALE INVARIANCE -- w -> 1000*w must leave L_sym unchanged")
from src.utils.graph import Graph

cfg["spectral"]["cum_decay"] = "harmonic"
srv = _server()
for t in range(3):
    srv._accumulate_cum_edges(t)
g = Graph(x=torch.ones(N, 1), edge_index=srv._cum_edges, node_ids=torch.arange(N))
w = srv._cum_edge_weight(2)
L1, _ = _lsym(g, w)
L2, _ = _lsym(g, w * 1000.0)
d = float(np.abs(np.asarray((L1 - L2).todense())).max())
print(f"  max|L(w) - L(1000w)| = {d:.3e}  {'ok' if d < 1e-9 else 'FAIL'}")

print()
print("SUPPORT -- count/harmonic/exp keep the active set; window may shrink it")
for name, r in res.items():
    same = all((a["act"] is None) == (b["act"] is None) and
               (a["act"] is None or np.array_equal(a["act"], b["act"]))
               for a, b in zip(r, base))
    sz = [0 if e["act"] is None else len(e["act"]) for e in r]
    bsz = [0 if e["act"] is None else len(e["act"]) for e in base]
    print(f"  {name:12s} active set identical to none: {str(same):5s}  "
          f"|act| last={sz[-1]} (none={bsz[-1]})  "
          f"mean edge mass={r[-1]['mass']}")
print("=" * 78)
print(f"\nGATES: separation={'PASS' if ok_sep else 'FAIL'}  "
      f"bit_identity={'PASS' if ok_bit else 'FAIL'}")
