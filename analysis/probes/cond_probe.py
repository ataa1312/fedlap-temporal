import sys, math, time
sys.path.insert(0, "/Users/ata/Desktop/master-thesis-workspace/master-thesis-codes/codes/fedlap")
import numpy as np
import torch

SEEDS = [1234, 1334, 1434]
K = 50
WARM = 5  # prequential fits start once >= WARM past snapshots exist

sys.argv = ["x", "-c", "config/uci_gru.yaml", "--set", "model.data_type=feature",
            "subgraph.num_subgraphs=1", "wandb.mode=disabled"]
from parser import Parser
p = Parser(); cfg = p.load_config(p.parse_args())
import src; src.config = cfg
from registries import datasets
import src.datasets  # noqa
from src.utils.graph import Graph
import src.dynamic_server as ds_mod
import main as M
import scipy.stats as sst
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def und(ei):
    e = ei.cpu().numpy()
    return {(min(a, b), max(a, b)) for a, b in zip(e[0], e[1]) if a != b}

def auc(pos, neg):
    s = np.concatenate([pos, neg]); r = sst.rankdata(s); npos = len(pos)
    return (r[:npos].sum() - npos * (npos + 1) / 2) / (npos * len(neg))

snaps = datasets["uci"](cfg)
N, T = snaps[0].num_nodes, len(snaps)

# ---- exact basis + candidates per t (seed-independent; computed once) ----
print("precomputing exact bases + candidates...", flush=True)
adj = [set() for _ in range(N)]
cumset = set()
Q_t, cand_t = {}, {}
for t in range(T - 1):
    for a, b in und(snaps[t].edge_index):
        cumset.add((a, b)); adj[a].add(b); adj[b].add(a)
    e = torch.tensor([[a for a, b in cumset] + [b for a, b in cumset],
                      [b for a, b in cumset] + [a for a, b in cumset]], dtype=torch.long)
    g = Graph(x=torch.ones(N, 1), edge_index=e, node_ids=torch.arange(N))
    D, U = g.calc_eigs_exact_sym(K)
    Q_t[t] = U.numpy()
    pos = list(und(snaps[t + 1].edge_index))
    if len(pos) < 10:
        continue
    rng = np.random.default_rng(7000 + t)  # fixed per t -> identical candidates across seeds
    neg = []
    while len(neg) < len(pos):
        a, b = rng.integers(0, N, 2)
        if a == b or (min(a, b), max(a, b)) in cumset:
            continue
        neg.append((int(a), int(b)))
    aa = []
    for pair_list in (pos, neg):
        aa.append(np.array([sum(1.0 / math.log(max(len(adj[c]), 2)) for c in adj[a] & adj[b])
                            for a, b in pair_list]))
    cand_t[t] = (pos, neg, aa[0], aa[1])

# ---- per-seed: capture eval-time z, build features, prequential probes ----
def spec_cos(Q, pairs):
    out = []
    for a, b in pairs:
        qa, qb = Q[a], Q[b]; na, nb = np.linalg.norm(qa), np.linalg.norm(qb)
        out.append(qa @ qb / (na * nb) if na > 0 and nb > 0 else 0.0)
    return np.array(out)

def pair_feats(z, pairs):
    zu = z[[a for a, _ in pairs]]; zv = z[[b for _, b in pairs]]
    nu = np.linalg.norm(zu, axis=1); nv = np.linalg.norm(zv, axis=1)
    dot = (zu * zv).sum(1)
    cos = dot / np.maximum(nu * nv, 1e-12)
    l2 = np.linalg.norm(zu - zv, axis=1)
    return np.stack([cos, dot, l2], 1), np.concatenate([zu, zv, np.abs(zu - zv), zu * zv], 1)

results = {k: [] for k in ["z_log", "zs_log", "z_mlp", "zs_mlp", "spec_alone",
                           "z_log_blind", "zs_log_blind"]}
for seed in SEEDS:
    cfg["seed"] = seed
    zs_capt = []
    orig = ds_mod._stitch_global_z
    def hook(zlist, ids, n, dim, device):
        gz = orig(zlist, ids, n, dim, device)
        zs_capt.append(gz.detach().cpu().numpy())
        return gz
    ds_mod._stitch_global_z = hook
    t0 = time.time()
    M.run_once()
    ds_mod._stitch_global_z = orig
    print(f"seed {seed}: captured z for {len(zs_capt)} snapshots ({time.time()-t0:.0f}s)", flush=True)

    # assemble per-t rows
    rows = {}
    for t, (pos, neg, aap, aan) in cand_t.items():
        if t >= len(zs_capt):
            continue
        z = zs_capt[t]
        Q = Q_t[t]
        fs_p, fr_p = pair_feats(z, pos); fs_n, fr_n = pair_feats(z, neg)
        sp_p, sp_n = spec_cos(Q, pos), spec_cos(Q, neg)
        rows[t] = (fs_p, fs_n, fr_p, fr_n, sp_p, sp_n, aap, aan)

    # prequential: fit on pooled snapshots < t, evaluate on t
    per_t = {k: [] for k in results}
    ts = sorted(rows)
    for i, t in enumerate(ts):
        past = ts[:i]
        if len(past) < WARM:
            continue
        def pool(idx_feats, idx_spec=None):
            X, y = [], []
            for tp in past:
                r = rows[tp]
                fp, fn = r[idx_feats + 0], r[idx_feats + 1]
                if idx_spec is not None:
                    fp = np.column_stack([fp, r[4]]); fn = np.column_stack([fn, r[5]])
                X.append(fp); y.append(np.ones(len(fp)))
                X.append(fn); y.append(np.zeros(len(fn)))
            return np.vstack(X), np.concatenate(y)
        r = rows[t]
        # scalar-feature logistic probe
        Xtr, ytr = pool(0)
        Xtr_s, ytr_s = pool(0, idx_spec=True)
        te_p, te_n = r[0], r[1]
        te_ps, te_ns = np.column_stack([r[0], r[4]]), np.column_stack([r[1], r[5]])
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(Xtr, ytr)
        per_t["z_log"].append(auc(clf.predict_proba(te_p)[:, 1], clf.predict_proba(te_n)[:, 1]))
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(Xtr_s, ytr_s)
        per_t["zs_log"].append(auc(clf.predict_proba(te_ps)[:, 1], clf.predict_proba(te_ns)[:, 1]))
        # AA-blind slice under the same fitted probes
        bp, bn = r[6] == 0, r[7] == 0
        if bp.sum() >= 10 and bn.sum() >= 10:
            clf0 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(Xtr, ytr)
            per_t["z_log_blind"].append(auc(clf0.predict_proba(te_p[bp])[:, 1],
                                            clf0.predict_proba(te_n[bn])[:, 1]))
            clf1 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(Xtr_s, ytr_s)
            per_t["zs_log_blind"].append(auc(clf1.predict_proba(te_ps[bp])[:, 1],
                                             clf1.predict_proba(te_ns[bn])[:, 1]))
        # raw-embedding MLP probe (stronger readout)
        Xtr, ytr = pool(2)
        Xtr_s, ytr_s = pool(2, idx_spec=True)
        te_p, te_n = r[2], r[3]
        te_ps, te_ns = np.column_stack([r[2], r[4]]), np.column_stack([r[3], r[5]])
        mlp = make_pipeline(StandardScaler(), MLPClassifier((64,), max_iter=200, random_state=0)).fit(Xtr, ytr)
        per_t["z_mlp"].append(auc(mlp.predict_proba(te_p)[:, 1], mlp.predict_proba(te_n)[:, 1]))
        mlp = make_pipeline(StandardScaler(), MLPClassifier((64,), max_iter=200, random_state=0)).fit(Xtr_s, ytr_s)
        per_t["zs_mlp"].append(auc(mlp.predict_proba(te_ps)[:, 1], mlp.predict_proba(te_ns)[:, 1]))
        per_t["spec_alone"].append(auc(r[4], r[5]))
    for k in results:
        if per_t[k]:
            results[k].append(np.mean(per_t[k]))
    print(f"seed {seed}: done ({len(per_t['z_log'])} eval snapshots)", flush=True)

import statistics as st
print("\n=== CONDITIONAL-INFORMATION PROBE — uci C1, exact Q50, prequential, mean over t>=WARM ===")
def ms(k):
    v = results[k]
    return f"{st.mean(v):.4f}±{st.pstdev(v):.4f} (n={len(v)})" if v else "–"
print(f"  spectral alone              : {ms('spec_alone')}")
print(f"  logistic  z-features        : {ms('z_log')}")
print(f"  logistic  z-features + spec : {ms('zs_log')}")
print(f"  MLP probe raw z pairs       : {ms('z_mlp')}")
print(f"  MLP probe raw z pairs + spec: {ms('zs_mlp')}")
print(f"  AA-blind  logistic z        : {ms('z_log_blind')}")
print(f"  AA-blind  logistic z + spec : {ms('zs_log_blind')}")
for a, b, lbl in [("zs_log", "z_log", "Δ logistic"), ("zs_mlp", "z_mlp", "Δ MLP"),
                  ("zs_log_blind", "z_log_blind", "Δ AA-blind")]:
    if results[a] and results[b]:
        d = [x - y for x, y in zip(results[a], results[b])]
        print(f"  {lbl:26s}: {st.mean(d):+.4f}±{st.pstdev(d):.4f}  per-seed {[f'{x:+.4f}' for x in d]}")
