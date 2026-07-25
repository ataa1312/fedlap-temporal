"""Gate 1: decode-time spectral score fusion inside the REAL eval protocol.

Replaces `compute_mrr_from_z` in the eval path with a byte-equivalent copy that
additionally scores the same candidate set with fused scores, so the reported
per-snapshot MRR (rank_eval_multiplier negatives per source, test-split
positives, mrr_method) and the fused MRR come from IDENTICAL pairs.

Two leakage-free fusion weights:
  preq  — logistic over [model score, spectral affinity] fit on PAST snapshots
  val   — logistic fit on the CURRENT snapshot's val-split edges (eval-mode
          scores; the test arm of this pair also uses eval-mode scores)
Each has its own placebo arm (fixed node-row permutation of the same basis).

usage: python analysis/probes/proto_fusion.py [dataset] [config] [Cs] [seeds]
"""
import os
import sys
import time
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import torch

_a = sys.argv[1:]
DATASET = _a[0] if len(_a) > 0 else "uci"
CONFIG = _a[1] if len(_a) > 1 else f"config/{DATASET}_gru.yaml"
CS = [int(c) for c in _a[2].split(",")] if len(_a) > 2 else [1, 3, 7, 9]
SEEDS = [int(s) for s in _a[3].split(",")] if len(_a) > 3 else [1234, 1334, 1434]

K_PE = 50        # exact low-k eigenvectors (matches the §10.7/§10.10 probes)
WARM = 5         # snapshots of history before the prequential fit is used
NEG_FIT = 20     # negatives per source kept from each past snapshot for fitting
VAL_NEG = 50     # negatives per val positive for the val-fitted weight

sys.argv = ["proto_fusion", "-c", CONFIG, "--set", "model.data_type=feature",
            "subgraph.num_subgraphs=1", "wandb.mode=disabled"]
from parser import Parser

p = Parser()
cfg = p.load_config(p.parse_args())
import src

src.config = cfg
from src import device
from registries import datasets
import src.datasets  # noqa: F401
from src.utils.graph import Graph
import src.dynamic_server as ds_mod
from src.metrics.mrr import _sample_filtered_negatives, _rank_and_aggregate
import main as M
from sklearn.linear_model import LogisticRegression

ARMS = ["model", "model_eval", "spec",
        "preq_real", "preq_plac", "val_real", "val_plac"]


def und(ei):
    e = ei.cpu().numpy()
    return {(min(a, b), max(a, b)) for a, b in zip(e[0], e[1]) if a != b}


def zsc(a):
    return (a - a.mean()) / (a.std() + 1e-12)


snaps = datasets[DATASET](cfg)
N, T = snaps[0].num_nodes, len(snaps)
PERM = np.random.default_rng(42).permutation(N)

print(f"{DATASET}: N={N} T={T}; precomputing exact low-{K_PE} bases...", flush=True)
QN, QP = {}, {}
cumset = set()
t0 = time.time()
for t in range(T - 1):
    cumset |= und(snaps[t].edge_index)
    src_l = [a for a, b in cumset]
    dst_l = [b for a, b in cumset]
    e = torch.tensor([src_l + dst_l, dst_l + src_l], dtype=torch.long)
    g = Graph(x=torch.ones(N, 1), edge_index=e, node_ids=torch.arange(N))
    _, U = g.calc_eigs_exact_sym(K_PE)
    Q = U.numpy().astype(np.float32)
    Qn = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)
    QN[t], QP[t] = Qn, Qn[PERM]
print(f"bases done ({time.time() - t0:.0f}s)", flush=True)

CUR = {"t": None, "srv": None}
HIST = []        # prequential store: (zm, z_spec, z_plac, y) from past snapshots
RUNREC = {}      # t -> {arm: mrr}
LAM = []         # (t, lambda_preq, lambda_val) coefficient ratios

_orig_eval = ds_mod.DynamicServer._eval_mrr
_orig_mrr = ds_mod.compute_mrr_from_z


def _eval_hook(self, t, mrr_k, mrr_method):
    CUR["t"], CUR["srv"] = t, self
    return _orig_eval(self, t, mrr_k, mrr_method)


def _eval_scores(model, z, snap, pairs_u, pairs_v, dev):
    """Score pairs with the head in EVAL mode (no BN-stat update, no dropout,
    no RNG draw), restoring the classifier's mode afterwards."""
    temp = snap.clone()
    temp.edge_label_index = torch.stack([pairs_u, pairs_v], dim=0)
    temp.edge_label = torch.zeros(pairs_u.size(0), device=dev)
    inner = getattr(model, "model", model)
    was = inner.training
    model.eval()
    with torch.no_grad():
        pred, _ = model.decode(z, temp)
    model.train(was)
    return pred.detach().cpu().numpy()


def _fit(X, y):
    return LogisticRegression(max_iter=1000).fit(X, y)


def _patched_mrr(z, snap, K, method, dev, model):
    """Copy of compute_mrr_from_z (identical RNG consumption + return value)
    plus the fusion readouts on the same candidate pairs."""
    t, srv = CUR["t"], CUR["srv"]
    pos_mask = snap.edge_label == 1.0
    if pos_mask.sum().item() == 0:
        return float("nan")

    pos_edges = snap.edge_label_index[:, pos_mask]
    u, v_pos = pos_edges[0], pos_edges[1]
    n_nodes = snap.num_nodes

    unique_sources = torch.unique(u)
    n_sources = unique_sources.size(0)
    v_neg = _sample_filtered_negatives(unique_sources, u, v_pos, n_nodes, K, dev)

    src_repeated = unique_sources.unsqueeze(1).expand(-1, K).flatten()
    v_neg_flat = v_neg.flatten()
    all_u = torch.cat([u, src_repeated])
    all_v = torch.cat([v_pos, v_neg_flat])

    temp = snap.clone()
    temp.edge_label_index = torch.stack([all_u, all_v], dim=0)
    temp.edge_label = torch.zeros(all_u.size(0), device=dev)

    scores, _ = model.decode(z, temp)
    mrr_model = _rank_and_aggregate(scores, u, unique_sources, n_sources, K, method)

    try:
        _fusion_readout(t, srv, z, model, snap, scores, u, unique_sources,
                        n_sources, K, method, dev, all_u, all_v)
    except Exception as exc:  # never let the probe break the run
        print(f"  [warn] fusion readout failed at t={t}: {exc!r}", flush=True)
    return mrr_model


def _fusion_readout(t, srv, z, model, snap, scores, u, unique_sources,
                    n_sources, K, method, dev, all_u, all_v):
    if t not in QN:
        return
    n_pos = u.size(0)
    m_tr = scores.detach().cpu().numpy().astype(np.float64)
    uu = all_u.cpu().numpy()
    vv = all_v.cpu().numpy()
    Qn, Qp = QN[t], QP[t]
    s_real = (Qn[uu] * Qn[vv]).sum(1).astype(np.float64)
    s_plac = (Qp[uu] * Qp[vv]).sum(1).astype(np.float64)
    y = np.zeros(len(m_tr))
    y[:n_pos] = 1.0

    def rank(np_scores):
        s = torch.as_tensor(np.ascontiguousarray(np_scores, dtype=np.float32), device=dev)
        return _rank_and_aggregate(s, u, unique_sources, n_sources, K, method)

    rec = {"model": rank(m_tr), "spec": rank(s_real)}

    # --- eval-mode model scores on the same pairs (baseline for the val arm) ---
    m_ev = _eval_scores(model, z, snap, all_u, all_v, dev).astype(np.float64)
    rec["model_eval"] = rank(m_ev)

    zm_tr, zs_r, zs_p = zsc(m_tr), zsc(s_real), zsc(s_plac)

    # --- prequential arm: weights fit on PAST snapshots only ---
    lam_preq = float("nan")
    if len(HIST) >= WARM:
        Xr = np.vstack([np.column_stack([h[0], h[1]]) for h in HIST])
        Xp = np.vstack([np.column_stack([h[0], h[2]]) for h in HIST])
        yy = np.concatenate([h[3] for h in HIST])
        cr, cp = _fit(Xr, yy), _fit(Xp, yy)
        rec["preq_real"] = rank(cr.decision_function(np.column_stack([zm_tr, zs_r])))
        rec["preq_plac"] = rank(cp.decision_function(np.column_stack([zm_tr, zs_p])))
        lam_preq = float(cr.coef_[0][1] / (abs(cr.coef_[0][0]) + 1e-12))

    # --- val arm: weights fit on the CURRENT snapshot's val edges ---
    lam_val = float("nan")
    pv = getattr(srv.global_snaps[t + 1], "pos_val", None)
    if pv is not None and pv.size(1) > 5:
        rng = np.random.default_rng(50000 + t)
        pa = pv[0].cpu().numpy()
        pb = pv[1].cpu().numpy()
        na = np.repeat(pa, VAL_NEG)
        nb = rng.integers(0, N, size=na.size)
        fu = np.concatenate([pa, na])
        fv = np.concatenate([pb, nb])
        fy = np.concatenate([np.ones(pa.size), np.zeros(na.size)])
        m_val = _eval_scores(
            model, z, snap,
            torch.as_tensor(fu, dtype=torch.long, device=dev),
            torch.as_tensor(fv, dtype=torch.long, device=dev), dev,
        ).astype(np.float64)
        sv_r = (Qn[fu] * Qn[fv]).sum(1).astype(np.float64)
        sv_p = (Qp[fu] * Qp[fv]).sum(1).astype(np.float64)
        zm_v = zsc(m_val)
        cr = _fit(np.column_stack([zm_v, zsc(sv_r)]), fy)
        cp = _fit(np.column_stack([zm_v, zsc(sv_p)]), fy)
        zm_ev = zsc(m_ev)
        rec["val_real"] = rank(cr.decision_function(np.column_stack([zm_ev, zs_r])))
        rec["val_plac"] = rank(cp.decision_function(np.column_stack([zm_ev, zs_p])))
        lam_val = float(cr.coef_[0][1] / (abs(cr.coef_[0][0]) + 1e-12))

    LAM.append((t, lam_preq, lam_val))
    RUNREC[t] = rec

    # --- push this snapshot into the prequential history (subsampled negatives) ---
    rng = np.random.default_rng(60000 + t)
    keep = np.arange(n_pos)
    n_neg = len(m_tr) - n_pos
    sub = rng.choice(n_neg, size=min(n_neg, NEG_FIT * n_sources), replace=False) + n_pos
    idx = np.concatenate([keep, sub])
    HIST.append((zm_tr[idx], zs_r[idx], zs_p[idx], y[idx]))


ds_mod.DynamicServer._eval_mrr = _eval_hook
ds_mod.compute_mrr_from_z = _patched_mrr

results = {}
for C in CS:
    cfg["subgraph"]["num_subgraphs"] = C
    for seed in SEEDS:
        cfg["seed"] = seed
        HIST.clear()
        RUNREC.clear()
        LAM.clear()
        t0 = time.time()
        out = M.run_once()
        ts_all = sorted(RUNREC)
        ts_fuse = [t for t in ts_all if "preq_real" in RUNREC[t] and "val_real" in RUNREC[t]]
        chk = float(np.mean([RUNREC[t]["model"] for t in ts_all])) if ts_all else float("nan")
        print(f"C{C} seed {seed}: {len(ts_all)} snaps ({len(ts_fuse)} fused) "
              f"reported_mean_mrr={out['mean_mrr']:.4f} probe_model_mean={chk:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if not ts_fuse:
            continue
        for arm in ARMS:
            vals = [RUNREC[t][arm] for t in ts_fuse if arm in RUNREC[t]]
            if vals:
                results.setdefault((C, arm), []).append(float(np.mean(vals)))
        lam_p = [l[1] for l in LAM if not np.isnan(l[1])]
        lam_v = [l[2] for l in LAM if not np.isnan(l[2])]
        results.setdefault((C, "lam_preq"), []).append(float(np.mean(lam_p)) if lam_p else float("nan"))
        results.setdefault((C, "lam_val"), []).append(float(np.mean(lam_v)) if lam_v else float("nan"))

ds_mod.DynamicServer._eval_mrr = _orig_eval
ds_mod.compute_mrr_from_z = _orig_mrr


def ms(C, k):
    v = results.get((C, k), [])
    return f"{st.mean(v):.4f}±{st.pstdev(v):.4f}" if v else "-"


def dd(C, a, b):
    va, vb = results.get((C, a), []), results.get((C, b), [])
    if not va or not vb:
        return "-"
    d = [x - y for x, y in zip(va, vb)]
    return f"{st.mean(d):+.4f}±{st.pstdev(d):.4f}"


print(f"\n=== IN-PROTOCOL FUSION — {DATASET}, K={cfg['experimental']['rank_eval_multiplier']} "
      f"negatives/source, mrr_method={cfg['metric']['mrr_method']}, test split, "
      f"{len(SEEDS)} seeds ===")
print(f"{'C':>2s} {'MRR model':>15s} {'Δ preq real':>17s} {'Δ preq plac':>17s} "
      f"{'MRR model(eval)':>16s} {'Δ val real':>17s} {'Δ val plac':>17s} "
      f"{'MRR spec':>15s} {'λ preq':>9s} {'λ val':>9s}")
for C in CS:
    lp = results.get((C, "lam_preq"), [])
    lv = results.get((C, "lam_val"), [])
    print(f"{C:>2d} {ms(C, 'model'):>15s} {dd(C, 'preq_real', 'model'):>17s} "
          f"{dd(C, 'preq_plac', 'model'):>17s} {ms(C, 'model_eval'):>16s} "
          f"{dd(C, 'val_real', 'model_eval'):>17s} {dd(C, 'val_plac', 'model_eval'):>17s} "
          f"{ms(C, 'spec'):>15s} "
          f"{(f'{st.mean(lp):+.2f}' if lp else '-'):>9s} "
          f"{(f'{st.mean(lv):+.2f}' if lv else '-'):>9s}")
