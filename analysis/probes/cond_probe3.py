"""usage: python analysis/probes/cond_probe3.py [dataset] [config] [Cs] [seeds]"""
import os, sys, math, time
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT)); os.chdir(_ROOT)
import numpy as np
import torch

_a = sys.argv[1:]
DATASET = _a[0] if len(_a) > 0 else "uci"
CONFIG = _a[1] if len(_a) > 1 else f"config/{DATASET}_gru.yaml"
CS = [int(c) for c in _a[2].split(",")] if len(_a) > 2 else [1, 3, 7, 9]
SEEDS = [int(s) for s in _a[3].split(",")] if len(_a) > 3 else [1234, 1334, 1434]
K = 50
WARM = 5

sys.argv = ["x", "-c", CONFIG, "--set", "model.data_type=feature",
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

snaps = datasets[DATASET](cfg)
N, T = snaps[0].num_nodes, len(snaps)

# ---- exact global bases + candidate pairs per t (C- and seed-independent) ----
print("precomputing exact bases + candidates...", flush=True)
adj = [set() for _ in range(N)]
cumset = set()
cand_t = {}
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

results = {}
cuts = {}  # (C) -> list over seeds of (mean cut fraction, mean lost edges per snapshot)
for C in CS:
    cfg["subgraph"]["num_subgraphs"] = C
    for seed in SEEDS:
        cfg["seed"] = seed
        score_capt = {}
        cut_frac_t, cut_abs_t = [], []
        orig_eval = ds_mod.DynamicServer._eval_mrr
        def eval_hook(self, t, mrr_k, mrr_method):
            res = orig_eval(self, t, mrr_k, mrr_method)
            # lost-MP edges: endpoints of snapshot t's edges in different clients
            owner = np.full(N, -1, dtype=np.int64)
            for ci, cl in enumerate(self.clients):
                owner[cl.snaps[t].node_ids.cpu().numpy()] = ci
            ei = self.global_snaps[t].edge_index.cpu().numpy()
            seen = set()
            total = cut = 0
            for a, b in zip(ei[0], ei[1]):
                if a == b:
                    continue
                key = (min(a, b), max(a, b))
                if key in seen:
                    continue
                seen.add(key)
                total += 1
                if owner[a] != owner[b]:
                    cut += 1
            if total:
                cut_frac_t.append(cut / total); cut_abs_t.append(cut)
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
        cuts.setdefault(C, []).append((float(np.mean(cut_frac_t)), float(np.mean(cut_abs_t))))
        print(f"C{C} seed {seed}: {len(score_capt)} snaps captured, "
              f"cut={np.mean(cut_frac_t):.3f} ({time.time()-t0:.0f}s)", flush=True)

        per_t = {k: [] for k in ["model", "model_spec", "model_blind", "model_spec_blind"]}
        ts = sorted(set(score_capt) & set(cand_t))
        for i, t in enumerate(ts):
            past = ts[:i]
            if len(past) < WARM:
                continue
            pos, neg, aa_p, aa_n, sp_p, sp_n = cand_t[t]
            ms = score_capt[t]
            ms_p, ms_n = ms[:len(pos)], ms[len(pos):]
            per_t["model"].append(auc(ms_p, ms_n))
            X, y = [], []
            for tp in past:
                po, ne, _, _, spp, spn = cand_t[tp]
                m = score_capt[tp]
                X += [np.column_stack([m[:len(po)], spp]), np.column_stack([m[len(po):], spn])]
                y += [np.ones(len(po)), np.zeros(len(ne))]
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(
                np.vstack(X), np.concatenate(y))
            te_p = np.column_stack([ms_p, sp_p]); te_n = np.column_stack([ms_n, sp_n])
            per_t["model_spec"].append(auc(clf.predict_proba(te_p)[:, 1], clf.predict_proba(te_n)[:, 1]))
            bp, bn = aa_p == 0, aa_n == 0
            if bp.sum() >= 10 and bn.sum() >= 10:
                per_t["model_blind"].append(auc(ms_p[bp], ms_n[bn]))
                per_t["model_spec_blind"].append(
                    auc(clf.predict_proba(te_p[bp])[:, 1], clf.predict_proba(te_n[bn])[:, 1]))
        for k in per_t:
            if per_t[k]:
                results.setdefault((C, k), []).append(float(np.mean(per_t[k])))

import statistics as st
print(f"\n=== FEDERATED CONDITIONAL-INFORMATION PROBE — {DATASET}, exact Q{K}, model-score baseline ===")
print(f"{'C':>2s} {'cut frac':>13s} {'lost e/snap':>11s} {'model alone':>15s} {'model+spec':>15s} "
      f"{'Δ overall':>17s} {'Δ AA-blind':>17s}")
for C in CS:
    cf = [x[0] for x in cuts[C]]; ca = [x[1] for x in cuts[C]]
    def ms_(k):
        v = results.get((C, k), [])
        return f"{st.mean(v):.4f}±{st.pstdev(v):.4f}" if v else "–"
    def dd(a, b):
        va, vb = results.get((C, a), []), results.get((C, b), [])
        if not va or not vb:
            return "–"
        d = [x - y for x, y in zip(va, vb)]
        return f"{st.mean(d):+.4f}±{st.pstdev(d):.4f}"
    print(f"{C:>2d} {st.mean(cf):>6.3f}±{st.pstdev(cf):.3f} {st.mean(ca):>11.1f} "
          f"{ms_('model'):>15s} {ms_('model_spec'):>15s} {dd('model_spec','model'):>17s} "
          f"{dd('model_spec_blind','model_blind'):>17s}")
