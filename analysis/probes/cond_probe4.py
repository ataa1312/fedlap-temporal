import os, sys, math, time
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT)); os.chdir(_ROOT)
import numpy as np
import torch

SEEDS = [1234, 1334, 1434]
CS = [1, 3, 7, 9]
K = 50
WARM = 5
NEG_RANK = 200  # ranking negatives per positive (corrupted target endpoint)

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

print("precomputing bases, 1:1 pools, and ranking sets...", flush=True)
adj = [set() for _ in range(N)]
cumset = set()
cand_t, rank_t = {}, {}
for t in range(T - 1):
    for a, b in und(snaps[t].edge_index):
        cumset.add((a, b)); adj[a].add(b); adj[b].add(a)
    e = torch.tensor([[a for a, b in cumset] + [b for a, b in cumset],
                      [b for a, b in cumset] + [a for a, b in cumset]], dtype=torch.long)
    g = Graph(x=torch.ones(N, 1), edge_index=e, node_ids=torch.arange(N))
    D, U = g.calc_eigs_exact_sym(K)
    Qn = U.numpy()
    Qnorm = Qn / np.maximum(np.linalg.norm(Qn, axis=1, keepdims=True), 1e-12)
    pos = list(und(snaps[t + 1].edge_index))
    if len(pos) < 10:
        continue
    tomorrow = set(pos)
    rng = np.random.default_rng(7000 + t)
    # 1:1 pool (for prequential fusion fitting; same construction as probe3)
    neg = []
    while len(neg) < len(pos):
        a, b = rng.integers(0, N, 2)
        if a == b or (min(a, b), max(a, b)) in cumset:
            continue
        neg.append((int(a), int(b)))
    def spec(pairs):
        ua = Qnorm[[a for a, _ in pairs]]; ub = Qnorm[[b for _, b in pairs]]
        return (ua * ub).sum(1)
    cand_t[t] = (pos, neg, spec(pos), spec(neg))
    # ranking sets: for each positive (u,v), NEG_RANK corruptions (u, w)
    rng2 = np.random.default_rng(9000 + t)
    rank_pairs = []
    for (u, v) in pos:
        rank_pairs.append((u, v))
        w = rng2.integers(0, N, NEG_RANK + 40)
        picked = 0
        for cand in w:
            cand = int(cand)
            if cand == u or cand == v or (min(u, cand), max(u, cand)) in tomorrow:
                continue
            rank_pairs.append((u, cand))
            picked += 1
            if picked == NEG_RANK:
                break
        while picked < NEG_RANK:  # pad in the unlikely exhaustion case
            rank_pairs.append((u, int(rng2.integers(0, N))))
            picked += 1
    rank_t[t] = (len(pos), rank_pairs, spec(rank_pairs))

def mrr_from_scores(npos, scores):
    block = 1 + NEG_RANK
    rr = []
    for i in range(npos):
        s = scores[i * block:(i + 1) * block]
        rank = 1 + (s[1:] > s[0]).sum() + 0.5 * (s[1:] == s[0]).sum()
        rr.append(1.0 / rank)
    return float(np.mean(rr))

results = {}
for C in CS:
    cfg["subgraph"]["num_subgraphs"] = C
    for seed in SEEDS:
        cfg["seed"] = seed
        capt = {}
        orig_eval = ds_mod.DynamicServer._eval_mrr
        def eval_hook(self, t, mrr_k, mrr_method):
            res = orig_eval(self, t, mrr_k, mrr_method)
            if t in cand_t:
                zs, ids = [], []
                for cl in self.clients:
                    z, nid = cl.encode(t)
                    zs.append(z); ids.append(nid)
                gz = orig_stitch(zs, ids, self.global_snaps[0].num_nodes, zs[0].shape[1], device)
                def score(pairs):
                    snap = self.global_snaps[t].clone().to(device)
                    snap.edge_label_index = torch.tensor(
                        [[a for a, _ in pairs], [b for _, b in pairs]],
                        dtype=torch.long, device=device)
                    snap.edge_label = torch.zeros(len(pairs), device=device)
                    was = self.classifier.model.training
                    self.classifier.eval()
                    with torch.no_grad():
                        pred, _ = self.classifier.decode(gz, snap)
                    self.classifier.train(was)
                    return pred.detach().cpu().numpy()
                pos, neg, _, _ = cand_t[t]
                capt[t] = (score(pos + neg), score(rank_t[t][1]))
            return res
        ds_mod.DynamicServer._eval_mrr = eval_hook
        t0 = time.time()
        M.run_once()
        ds_mod.DynamicServer._eval_mrr = orig_eval
        print(f"C{C} seed {seed}: {len(capt)} snaps ({time.time()-t0:.0f}s)", flush=True)

        per_t = {k: [] for k in ["mrr_model", "mrr_fused", "auc_model", "auc_fused"]}
        ts = sorted(set(capt) & set(cand_t))
        for i, t in enumerate(ts):
            past = ts[:i]
            if len(past) < WARM:
                continue
            pos, neg, sp_p, sp_n = cand_t[t]
            ms = capt[t][0]
            ms_p, ms_n = ms[:len(pos)], ms[len(pos):]
            per_t["auc_model"].append(auc(ms_p, ms_n))
            X, y = [], []
            for tp in past:
                po, ne, spp, spn = cand_t[tp]
                m = capt[tp][0]
                X += [np.column_stack([m[:len(po)], spp]), np.column_stack([m[len(po):], spn])]
                y += [np.ones(len(po)), np.zeros(len(ne))]
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(
                np.vstack(X), np.concatenate(y))
            fu = lambda m_, s_: clf.predict_proba(np.column_stack([m_, s_]))[:, 1]
            per_t["auc_fused"].append(auc(fu(ms_p, sp_p), fu(ms_n, sp_n)))
            npos, rpairs, rspec = rank_t[t]
            rms = capt[t][1]
            per_t["mrr_model"].append(mrr_from_scores(npos, rms))
            per_t["mrr_fused"].append(mrr_from_scores(npos, fu(rms, rspec)))
        for k in per_t:
            if per_t[k]:
                results.setdefault((C, k), []).append(float(np.mean(per_t[k])))

import statistics as st
print("\n=== MRR-STYLE READOUT — uci, 200 ranking negatives/positive, prequential fusion ===")
print(f"{'C':>2s} {'MRR model':>15s} {'MRR fused':>15s} {'Δ MRR':>17s} {'AUC model':>15s} {'Δ AUC (sanity)':>17s}")
for C in CS:
    def ms_(k):
        v = results.get((C, k), [])
        return f"{st.mean(v):.4f}±{st.pstdev(v):.4f}" if v else "–"
    def dd(a, b):
        va, vb = results.get((C, a), []), results.get((C, b), [])
        if not va or not vb:
            return "–"
        d = [x - y for x, y in zip(va, vb)]
        return f"{st.mean(d):+.4f}±{st.pstdev(d):.4f}"
    print(f"{C:>2d} {ms_('mrr_model'):>15s} {ms_('mrr_fused'):>15s} {dd('mrr_fused','mrr_model'):>17s} "
          f"{ms_('auc_model'):>15s} {dd('auc_fused','auc_model'):>17s}")
