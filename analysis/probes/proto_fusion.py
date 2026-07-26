"""Gate 1: decode-time spectral score fusion inside the REAL eval protocol.

Replaces `compute_mrr_from_z` in the eval path with a byte-equivalent copy that
additionally scores the same candidate set with fused scores, so the reported
per-snapshot MRR (rank_eval_multiplier negatives per source, test-split
positives, mrr_method) and the fused MRR come from IDENTICAL pairs.

Fused features, each combined with the model score by its own prequential
logistic (fit on past snapshots only) so the arms are directly comparable:
  spec   — cosine affinity in the exact low-K eigenbasis of the cumulative graph
  plac   — the placebo: same eigenvectors, node rows permuted once (structure gone)
  exists — 1 if the pair is already an edge of that cumulative graph (trivial baseline)
  cn     — log1p(common neighbours) in that graph (trivial baseline)
Plus a `val` arm for spec only: weights fit on the CURRENT snapshot's val-split
edges (leakage-free without history — the recipe an in-model λ would copy).

Every arm is also reported split by whether the positive is a REPEAT of an edge
already in the cumulative graph or a genuinely NEW pair.

`thin` keeps only that fraction of the cumulative edges when solving the
eigenbasis — model and task untouched, so it isolates "does a sparser graph
carry a weaker basis".

usage: python analysis/probes/proto_fusion.py [dataset] [config] [Cs] [seeds] [K] [thin]
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
from scipy import sparse

_a = sys.argv[1:]
DATASET = _a[0] if len(_a) > 0 else "uci"
CONFIG = _a[1] if len(_a) > 1 else f"config/{DATASET}_gru.yaml"
CS = [int(c) for c in _a[2].split(",")] if len(_a) > 2 else [1, 3, 7, 9]
SEEDS = [int(s) for s in _a[3].split(",")] if len(_a) > 3 else [1234, 1334, 1434]
K_PE = int(_a[4]) if len(_a) > 4 else 50  # exact low-k eigenvectors (§10.7/§10.10 used 50)
THIN = float(_a[5]) if len(_a) > 5 else 1.0  # fraction of cumulative edges kept for the basis

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

FEATS = ["spec", "plac", "exists", "cn"]
ARMS = (["model", "model_eval", "rep_frac", "model_rep", "model_new",
         "val_real", "val_plac"]
        + [f"alone_{k}" for k in FEATS]
        + [f"preq_{k}{s}" for k in FEATS for s in ("", "_rep", "_new")])


CACHE = os.environ.get("PROTO_BASIS_CACHE")  # dir of per-snapshot row-normalised bases
if CACHE:
    Path(CACHE).mkdir(parents=True, exist_ok=True)


def und(ei):
    e = ei.cpu().numpy()
    return {(min(a, b), max(a, b)) for a, b in zip(e[0], e[1]) if a != b}


def zsc(a):
    return (a - a.mean()) / (a.std() + 1e-12)


def basis_path(dataset, k, thin, t):
    return Path(CACHE) / f"{dataset}_K{k}_thin{thin}_t{t}.npy"


def solve_basis(N, edges, k, rng=None, thin=1.0):
    """Row-normalised exact low-k eigenvector rows of the cumulative graph."""
    a = np.array([x for x, _ in edges], dtype=np.int64)
    b = np.array([y for _, y in edges], dtype=np.int64)
    if thin < 1.0:
        keep = rng.random(a.size) < thin
        a, b = a[keep], b[keep]
    e = torch.tensor(np.stack([np.r_[a, b], np.r_[b, a]]), dtype=torch.long)
    g = Graph(x=torch.ones(N, 1), edge_index=e, node_ids=torch.arange(N))
    _, U = g.calc_eigs_exact_sym(k)
    Q = U.numpy().astype(np.float32)
    return Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)


snaps = datasets[DATASET](cfg)
N, T = snaps[0].num_nodes, len(snaps)
PERM = np.random.default_rng(42).permutation(N)

print(f"{DATASET}: N={N} T={T}; precomputing exact low-{K_PE} bases (thin={THIN})...", flush=True)
QN, QP, CUMKEY, ADJ = {}, {}, {}, {}
cumset = set()
t0 = time.time()
thin_rng = np.random.default_rng(31337)
for t in range(T - 1):
    cumset |= und(snaps[t].edge_index)
    aa = np.array([a for a, _ in cumset], dtype=np.int64)
    bb = np.array([b for _, b in cumset], dtype=np.int64)
    CUMKEY[t] = np.sort(aa * N + bb)
    A = sparse.coo_matrix((np.ones(2 * aa.size), (np.r_[aa, bb], np.r_[bb, aa])),
                          shape=(N, N)).tocsr()
    A.data[:] = 1.0
    ADJ[t] = A
    cached = basis_path(DATASET, K_PE, THIN, t) if CACHE else None
    if cached is not None and cached.exists():
        Qn = np.load(cached)
    else:
        Qn = solve_basis(N, sorted(cumset), K_PE, thin_rng, THIN)
        if cached is not None:  # write atomically: other hosts read this dir
            tmp = cached.with_suffix(".tmp.npy")
            np.save(tmp, Qn)
            tmp.rename(cached)
    QN[t], QP[t] = Qn, Qn[PERM]
print(f"bases done ({time.time() - t0:.0f}s)", flush=True)

CUR = {"t": None, "srv": None}
HIST = []        # prequential store: dicts of per-pair features from past snapshots
RUNREC = {}      # t -> {arm: mrr}
LAM = []         # (t, lambda_preq_spec, lambda_val_spec)

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


def _pair_features(t, uu, vv):
    """The three graph features on the cumulative graph through t, plus the placebo."""
    Qn, Qp, A = QN[t], QP[t], ADJ[t]
    key = np.minimum(uu, vv).astype(np.int64) * N + np.maximum(uu, vv)
    ck = CUMKEY[t]
    idx = np.clip(np.searchsorted(ck, key), 0, max(len(ck) - 1, 0))
    exists = (ck[idx] == key) if len(ck) else np.zeros(len(key), bool)
    cn = np.asarray(A[uu].multiply(A[vv]).sum(axis=1)).ravel()
    return {"spec": (Qn[uu] * Qn[vv]).sum(1).astype(np.float64),
            "plac": (Qp[uu] * Qp[vv]).sum(1).astype(np.float64),
            "exists": exists.astype(np.float64),
            "cn": np.log1p(cn)}, exists


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
    feats, exists = _pair_features(t, uu, vv)
    y = np.zeros(len(m_tr))
    y[:n_pos] = 1.0
    is_rep = exists[:n_pos]
    u_np = u.cpu().numpy()
    us_np = unique_sources.cpu().numpy()

    def rank(np_scores):
        s = torch.as_tensor(np.ascontiguousarray(np_scores, dtype=np.float32), device=dev)
        return _rank_and_aggregate(s, u, unique_sources, n_sources, K, method)

    def rank_masked(np_scores, mask):
        if method != "max" or not mask.any():
            return float("nan")
        pos = np_scores[:n_pos]
        neg = np_scores[n_pos:].reshape(n_sources, K)
        rr = []
        for i, srcid in enumerate(us_np):
            sel = (u_np == srcid) & mask
            if not sel.any():
                continue
            rr.append(1.0 / ((neg[i] >= pos[sel].max()).sum() + 1))
        return float(np.mean(rr)) if rr else float("nan")

    rec = {"model": rank(m_tr), "rep_frac": float(is_rep.mean()),
           "model_rep": rank_masked(m_tr, is_rep),
           "model_new": rank_masked(m_tr, ~is_rep)}
    for k in FEATS:
        rec[f"alone_{k}"] = rank(feats[k])

    m_ev = _eval_scores(model, z, snap, all_u, all_v, dev).astype(np.float64)
    rec["model_eval"] = rank(m_ev)

    zm_tr = zsc(m_tr)
    zf = {k: zsc(v) for k, v in feats.items()}

    # --- prequential arms: weights fit on PAST snapshots only, one per feature ---
    lam_preq = float("nan")
    if len(HIST) >= WARM:
        yy = np.concatenate([h["y"] for h in HIST])
        for k in FEATS:
            X = np.vstack([np.column_stack([h["zm"], h[k]]) for h in HIST])
            clf = _fit(X, yy)
            f = clf.decision_function(np.column_stack([zm_tr, zf[k]]))
            rec[f"preq_{k}"] = rank(f)
            rec[f"preq_{k}_rep"] = rank_masked(f, is_rep)
            rec[f"preq_{k}_new"] = rank_masked(f, ~is_rep)
            if k == "spec":
                lam_preq = float(clf.coef_[0][1] / (abs(clf.coef_[0][0]) + 1e-12))

    # --- val arm (spec + placebo): weights fit on the CURRENT snapshot's val edges ---
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
        vfeats, _ = _pair_features(t, fu, fv)
        zm_v = zsc(m_val)
        zm_ev = zsc(m_ev)
        for k, arm in (("spec", "val_real"), ("plac", "val_plac")):
            clf = _fit(np.column_stack([zm_v, zsc(vfeats[k])]), fy)
            rec[arm] = rank(clf.decision_function(np.column_stack([zm_ev, zf[k]])))
            if k == "spec":
                lam_val = float(clf.coef_[0][1] / (abs(clf.coef_[0][0]) + 1e-12))

    LAM.append((t, lam_preq, lam_val))
    RUNREC[t] = rec

    # --- push this snapshot into the prequential history (subsampled negatives) ---
    rng = np.random.default_rng(60000 + t)
    n_neg = len(m_tr) - n_pos
    sub = rng.choice(n_neg, size=min(n_neg, NEG_FIT * n_sources), replace=False) + n_pos
    idx = np.concatenate([np.arange(n_pos), sub])
    HIST.append({"zm": zm_tr[idx], "y": y[idx], **{k: zf[k][idx] for k in FEATS}})


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
        ts_fuse = [t for t in ts_all if "preq_spec" in RUNREC[t] and "val_real" in RUNREC[t]]
        chk = float(np.mean([RUNREC[t]["model"] for t in ts_all])) if ts_all else float("nan")
        print(f"C{C} seed {seed}: {len(ts_all)} snaps ({len(ts_fuse)} fused) "
              f"reported_mean_mrr={out['mean_mrr']:.4f} probe_model_mean={chk:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if not ts_fuse:
            continue
        for arm in ARMS:
            vals = [RUNREC[t][arm] for t in ts_fuse if arm in RUNREC[t]]
            vals = [v for v in vals if not np.isnan(v)]
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


def mv(C, k):
    v = results.get((C, k), [])
    return f"{st.mean(v):.3f}" if v else "-"


def dd(C, a, b):
    va, vb = results.get((C, a), []), results.get((C, b), [])
    if not va or not vb:
        return "-"
    d = [x - y for x, y in zip(va, vb)]
    return f"{st.mean(d):+.4f}±{st.pstdev(d):.4f}"


hdr = (f"\n=== IN-PROTOCOL FUSION — {DATASET}, exact Q{K_PE}, thin={THIN}, "
       f"{cfg['experimental']['rank_eval_multiplier']} negatives/source, "
       f"mrr_method={cfg['metric']['mrr_method']}, test split, {len(SEEDS)} seeds ===")
print(hdr)
print(f"{'C':>2s} {'MRR model':>15s} {'Δ preq spec':>17s} {'Δ preq PLACEBO':>17s} "
      f"{'Δ preq exists':>17s} {'Δ preq cn':>17s} {'Δ val spec':>17s} {'Δ val plac':>17s} "
      f"{'λ preq':>8s} {'λ val':>8s}")
for C in CS:
    lp = results.get((C, "lam_preq"), [])
    lv = results.get((C, "lam_val"), [])
    print(f"{C:>2d} {ms(C, 'model'):>15s} {dd(C, 'preq_spec', 'model'):>17s} "
          f"{dd(C, 'preq_plac', 'model'):>17s} {dd(C, 'preq_exists', 'model'):>17s} "
          f"{dd(C, 'preq_cn', 'model'):>17s} {dd(C, 'val_real', 'model_eval'):>17s} "
          f"{dd(C, 'val_plac', 'model_eval'):>17s} "
          f"{(f'{st.mean(lp):+.2f}' if lp else '-'):>8s} "
          f"{(f'{st.mean(lv):+.2f}' if lv else '-'):>8s}")

print("\n--- feature alone (no model), and the repeat/new split of the spec + exists arms ---")
print(f"{'C':>2s} {'rep frac':>9s} | {'spec':>7s} {'plac':>7s} {'exists':>7s} {'cn':>7s} | "
      f"{'model REP':>10s} {'Δ spec REP':>17s} {'Δ exists REP':>17s} | "
      f"{'model NEW':>10s} {'Δ spec NEW':>17s} {'Δ exists NEW':>17s}")
for C in CS:
    print(f"{C:>2d} {mv(C, 'rep_frac'):>9s} | {mv(C, 'alone_spec'):>7s} {mv(C, 'alone_plac'):>7s} "
          f"{mv(C, 'alone_exists'):>7s} {mv(C, 'alone_cn'):>7s} | "
          f"{mv(C, 'model_rep'):>10s} {dd(C, 'preq_spec_rep', 'model_rep'):>17s} "
          f"{dd(C, 'preq_exists_rep', 'model_rep'):>17s} | "
          f"{mv(C, 'model_new'):>10s} {dd(C, 'preq_spec_new', 'model_new'):>17s} "
          f"{dd(C, 'preq_exists_new', 'model_new'):>17s}")
