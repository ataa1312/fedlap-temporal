import os, sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT)); os.chdir(_ROOT)
import math, torch
sys.argv = ["x", "-c", "config/uci_gru.yaml", "--set", "model.data_type=f+s",
            "spectral.update_mode=recompute", "subgraph.num_subgraphs=1", "wandb.mode=disabled"]
from parser import Parser
p = Parser(); cfg = p.load_config(p.parse_args())
import src; src.config = cfg
from registries import datasets
import src.datasets  # noqa
from src.utils.graph import Graph

torch.manual_seed(0)
snaps = datasets["uci"](cfg)
N = snaps[0].num_nodes
T = len(snaps)

def undirected_pairs(edge_index):
    e = edge_index.cpu().numpy()
    return {(min(a, b), max(a, b)) for a, b in zip(e[0], e[1]) if a != b}

def auc(pos, neg):
    import numpy as np
    s = np.concatenate([pos, neg]); y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = s.argsort(); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # midranks for ties
    import scipy.stats as st
    ranks = st.rankdata(s)
    return (ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

import numpy as np
adj = [set() for _ in range(N)]          # cumulative adjacency
deg = np.zeros(N)
cum_edges = None
Q_keep = None
rows = []
rng = np.random.default_rng(1234)

for t in range(T - 1):
    e = snaps[t].edge_index.cpu()
    e2 = torch.cat([e, e.flip(0)], dim=1)
    cum_edges = e2 if cum_edges is None else torch.unique(torch.cat([cum_edges, e2], dim=1), dim=1)
    for a, b in undirected_pairs(e):
        adj[a].add(b); adj[b].add(a)
    deg = np.array([len(s_) for s_ in adj], dtype=float)

    g = Graph(x=torch.ones(N, 1), edge_index=cum_edges, node_ids=torch.arange(N))
    D, U, _ = g.calc_eignvalues(estimate=True, spectral_len=300, log=False)
    Qr = U.detach().float().cpu().numpy()
    Dr = D.detach().float().cpu().numpy()
    if Q_keep is None:
        Q_keep = Qr.copy()

    pos_pairs = list(undirected_pairs(snaps[t + 1].edge_index))
    if len(pos_pairs) < 10:
        continue
    npairs = len(pos_pairs)
    neg_pairs = []
    posset = set(pos_pairs)
    while len(neg_pairs) < npairs:
        a, b = rng.integers(0, N, 2)
        if a == b: continue
        key = (min(a, b), max(a, b))
        if key in posset: continue
        neg_pairs.append((int(a), int(b)))

    def scores(pairs):
        aa, dg, sk, sr, sl, prev = [], [], [], [], [], []
        order = np.argsort(Dr)          # low eigenvalues first
        low = order[:50]
        def cos(Q, a, b, cols=None):
            qa = Q[a] if cols is None else Q[a][cols]
            qb = Q[b] if cols is None else Q[b][cols]
            na, nb = np.linalg.norm(qa), np.linalg.norm(qb)
            return float(qa @ qb / (na * nb)) if na > 0 and nb > 0 else 0.0
        for a, b in pairs:
            common = adj[a] & adj[b]
            aa.append(sum(1.0 / math.log(max(len(adj[w]), 2)) for w in common))
            dg.append(deg[a] * deg[b])
            sk.append(cos(Q_keep, a, b))
            sr.append(cos(Qr, a, b))
            sl.append(cos(Qr, a, b, low))
            prev.append(1.0 if b in adj[a] else 0.0)
        return map(np.array, (aa, dg, sk, sr, sl, prev))

    Paa, Pdg, Psk, Psr, Psl, Pprev = scores(pos_pairs)
    Naa, Ndg, Nsk, Nsr, Nsl, Nprev = scores(neg_pairs)

    row = dict(t=t, n=npairs,
               AA=auc(Paa, Naa), DEG=auc(Pdg, Ndg), PREV=auc(Pprev, Nprev),
               SPEC_keep=auc(Psk, Nsk), SPEC_rec=auc(Psr, Nsr), SPEC_low50=auc(Psl, Nsl))
    # complementary test: candidates where AA is blind (no common neighbors)
    pm = Paa == 0; nm = Naa == 0
    if pm.sum() >= 10 and nm.sum() >= 10:
        row["frac_pos_AAblind"] = float(pm.mean())
        row["SPEC_rec@AAblind"] = auc(Psr[pm], Nsr[nm])
        row["DEG@AAblind"] = auc(Pdg[pm], Ndg[nm])
    rows.append(row)

import statistics as st2
keys = ["AA", "DEG", "PREV", "SPEC_keep", "SPEC_rec", "SPEC_low50", "SPEC_rec@AAblind", "DEG@AAblind", "frac_pos_AAblind"]
print(f"\n=== ORACLE PROBE uci ({len(rows)} snapshots, random negatives, mean AUC) ===")
for k in keys:
    v = [r[k] for r in rows if k in r]
    if v: print(f"  {k:18s} {st2.mean(v):.3f}   (n={len(v)})")
