import os, sys, math, time
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT)); os.chdir(_ROOT)
import numpy as np
import torch

SEEDS = [1234, 1334, 1434]
K = 50
WARM = 5

sys.argv = ["x", "-c", "config/uci_gru.yaml", "--set", "model.data_type=feature",
            "subgraph.num_subgraphs=1", "wandb.mode=disabled"]
from parser import Parser
p = Parser(); cfg = p.load_config(p.parse_args())
import src; src.config = cfg
from src import device
from registries import datasets
import src.datasets  # noqa
from src.utils.graph import Graph
import src.dynamic_server as ds_mod
from src.train.federated_orchestrator import _stitch_global_z as orig_stitch
import main as M
import scipy.stats as sst
from sklearn.linear_model import LogisticRegression
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
    pos = list(und(snaps[t + 1].edge_index))
    if len(pos) < 10:
        continue
    rng = np.random.default_rng(7000 + t)
    neg = []
    while len(neg) < len(pos):
        a, b = rng.integers(0, N, 2)
        if a == b or (min(a, b), max(a, b)) in cumset:
            continue
        neg.append((int(a), int(b)))
    aa_p = np.array([sum(1.0 / math.log(max(len(adj[c]), 2)) for c in adj[a] & adj[b]) for a, b in pos])
    aa_n = np.array([sum(1.0 / math.log(max(len(adj[c]), 2)) for c in adj[a] & adj[b]) for a, b in neg])
    Qn = U.numpy()
    def sc(pairs):
        out = []
        for a, b in pairs:
            qa, qb = Qn[a], Qn[b]; na, nb = np.linalg.norm(qa), np.linalg.norm(qb)
            out.append(qa @ qb / (na * nb) if na > 0 and nb > 0 else 0.0)
        return np.array(out)
    cand_t[t] = (pos, neg, aa_p, aa_n, sc(pos), sc(neg))

results = {k: [] for k in ["model", "model_spec", "model_blind", "model_spec_blind"]}
for seed in SEEDS:
    cfg["seed"] = seed
    score_capt = {}
    orig_eval = ds_mod.DynamicServer._eval_mrr
    def eval_hook(self, t, mrr_k, mrr_method):
        res = orig_eval(self, t, mrr_k, mrr_method)
        if t in cand_t:
            zs, ids = [], []
            for cl in self.clients:
                z, nid = cl.encode(t)
                zs.append(z); ids.append(nid)
            gz = orig_stitch(zs, ids, self.global_snaps[0].num_nodes, zs[0].shape[1], device)
            pos, neg = cand_t[t][0], cand_t[t][1]
            pairs = pos + neg
            snap = self.global_snaps[t].clone().to(device)
            snap.edge_label_index = torch.tensor(
                [[a for a, _ in pairs], [b for _, b in pairs]], dtype=torch.long, device=device)
            snap.edge_label = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))]).to(device)
            was = self.classifier.model.training
            self.classifier.eval()
            with torch.no_grad():
                pred, _ = self.classifier.decode(gz, snap)
            self.classifier.train(was)
            score_capt[t] = pred.detach().cpu().numpy()
        return res
    ds_mod.DynamicServer._eval_mrr = eval_hook
    t0 = time.time()
    M.run_once()
    ds_mod.DynamicServer._eval_mrr = orig_eval
    print(f"seed {seed}: captured model scores for {len(score_capt)} snapshots ({time.time()-t0:.0f}s)", flush=True)

    per_t = {k: [] for k in results}
    ts = sorted(set(score_capt) & set(cand_t))
    for i, t in enumerate(ts):
        past = ts[:i]
        if len(past) < WARM:
            continue
        pos, neg, aa_p, aa_n, sp_p, sp_n = cand_t[t]
        ms = score_capt[t]
        ms_p, ms_n = ms[:len(pos)], ms[len(pos):]
        per_t["model"].append(auc(ms_p, ms_n))
        def pool(with_spec):
            X, y = [], []
            for tp in past:
                po, ne, _, _, spp, spn = cand_t[tp]
                m = score_capt[tp]
                mp, mn = m[:len(po)], m[len(po):]
                Xp = np.column_stack([mp, spp]) if with_spec else mp[:, None]
                Xn = np.column_stack([mn, spn]) if with_spec else mn[:, None]
                X += [Xp, Xn]; y += [np.ones(len(po)), np.zeros(len(ne))]
            return np.vstack(X), np.concatenate(y)
        Xtr, ytr = pool(True)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(Xtr, ytr)
        te_p = np.column_stack([ms_p, sp_p]); te_n = np.column_stack([ms_n, sp_n])
        per_t["model_spec"].append(auc(clf.predict_proba(te_p)[:, 1], clf.predict_proba(te_n)[:, 1]))
        bp, bn = aa_p == 0, aa_n == 0
        if bp.sum() >= 10 and bn.sum() >= 10:
            per_t["model_blind"].append(auc(ms_p[bp], ms_n[bn]))
            per_t["model_spec_blind"].append(
                auc(clf.predict_proba(te_p[bp])[:, 1], clf.predict_proba(te_n[bn])[:, 1]))
    for k in results:
        if per_t[k]:
            results[k].append(np.mean(per_t[k]))
    print(f"seed {seed}: done", flush=True)

import statistics as st
print("\n=== DECISIVE PROBE — model's OWN trained scores ± exact-spectral affinity (uci C1) ===")
def ms_(k):
    v = results[k]
    return f"{st.mean(v):.4f}±{st.pstdev(v):.4f} (n={len(v)})" if v else "–"
print(f"  model score alone            : {ms_('model')}")
print(f"  model score + spec (logistic): {ms_('model_spec')}")
print(f"  AA-blind: model alone        : {ms_('model_blind')}")
print(f"  AA-blind: model + spec       : {ms_('model_spec_blind')}")
for a, b, lbl in [("model_spec", "model", "Δ overall"), ("model_spec_blind", "model_blind", "Δ AA-blind")]:
    if results[a] and results[b]:
        d = [x - y for x, y in zip(results[a], results[b])]
        print(f"  {lbl:28s}: {st.mean(d):+.4f}±{st.pstdev(d):.4f}  per-seed {[f'{x:+.4f}' for x in d]}")
