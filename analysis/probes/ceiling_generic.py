"""Is the C-growing "conditional ceiling" (§10.10) specific to the spectral basis,
or does ANY C-independent side feature show it because the baseline falls?

`_spectral_step` (src/dynamic_server.py:786-796) solves the basis on the GLOBAL
cumulative union and only afterwards slices rows per client; the partition never
enters the solve. cond_probe3.py mirrors that: it precomputes the bases and the
candidate pairs ONCE, outside the C loop. So the injected feature is bit-identical
at C=1 and C=9 and carries no C-dependent information. Under that reading the
ceiling Delta = AUC(base+spec) - AUC(base) can only grow with C because AUC(base)
falls, i.e. because the side feature stops being redundant with what the baseline
still sees. If so, any informative C-independent feature reproduces the curve.

Two model-free experiments (NO training anywhere):

  A. SHARDING. cond_probe3's trained backbone is replaced by a model-free
     surrogate: a prequential logistic over structural features of the VISIBLE
     graph, i.e. the cumulative union minus every cross-client edge. That is
     exactly the edge set message passing still has under a random C-way node
     partition (src/utils/graph_partitioning.py:145-162 keeps intra_edges only).
     The surrogate degrades with C for the same reason the backbone does. Side
     features (all computed on the GLOBAL cumulative union, hence C-independent):
     spec, exists, cn, aa, degprod, plus a permuted-basis placebo and pure noise.

  B. NOISE LADDER, C=1, ZERO edges severed. The same C=1 baseline score is
     degraded by additive Gaussian noise down through the AUC levels the real
     backbone hits at C3/C7/C9. If Delta grows the same way with no partition at
     all, "the ceiling grows with sharding" carries no federated content.

Table A2 additionally splits the ceiling on §16's axis: positives that already sit
in the cumulative union (repeat) versus positives that do not (new), with the
negative set held identical across both subsets.

Candidate SUPPORTED if the non-spectral side features track spec's curve in A and
if B reproduces the curve without severing anything. REFUTED if only the
globally-computed eigenbasis grows with C.

Candidate-set caveat, inherited verbatim from cond_probe3: negatives are sampled
to be NON-edges of the cumulative union, so `exists` is 1 on every repeat positive
and 0 on every negative. At C=1 the visible-graph baseline already contains that
bit, so base-AUC on the repeat subset is pinned at 1.0 and the C=1 repeat cell has
no headroom by construction. The C>1 cells are the informative ones.

usage: python analysis/probes/ceiling_generic.py [dataset] [Cs] [seeds] [stride] [full|noexists]
       CEILGEN_CACHE=<dir> caches the per-snapshot bases + pair features, so
       re-runs of the readout cost seconds instead of an eigendecomposition sweep.
"""
import os
import sys
import math
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

import pickle

import numpy as np
import scipy.stats as sst
import torch

_a = sys.argv[1:]
DATASET = _a[0] if len(_a) > 0 else "uci"
CS = [int(c) for c in _a[1].split(",")] if len(_a) > 1 else [1, 3, 7, 9]
SEEDS = [int(s) for s in _a[2].split(",")] if len(_a) > 2 else [1234, 1334, 1434]
STRIDE = int(_a[3]) if len(_a) > 3 else 1
# 'noexists' drops the visible-graph adjacency bit from the baseline, so the
# baseline cannot see "this pair is already an edge here". Robustness check: the
# spec-vs-exists ordering must not be an artefact of the baseline's feature set.
BASE_MODE = _a[4] if len(_a) > 4 else "full"
BCOLS = slice(0, 5) if BASE_MODE == "full" else slice(1, 5)
K = 50
WARM = 5
MIN_POS = 10
MAX_POS = 1500  # candidate positives per snapshot (as733 snapshots carry ~20k edges)
PWIN = 25       # prequential fit uses at most this many past snapshots
SIGMAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 8.0]

sys.argv = ["ceiling_generic", "-c", f"config/{DATASET}_gru.yaml", "--set",
            "model.data_type=feature", "subgraph.num_subgraphs=1", "wandb.mode=disabled"]
from parser import Parser

p = Parser()
cfg = p.load_config(p.parse_args())
import src

src.config = cfg
from registries import datasets
import src.datasets  # noqa: F401
from src.utils.graph import Graph
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def und(ei):
    e = ei.cpu().numpy()
    return {(min(a, b), max(a, b)) for a, b in zip(e[0], e[1]) if a != b}


def auc(pos, neg):
    s = np.concatenate([pos, neg])
    r = sst.rankdata(s)
    npos = len(pos)
    return (r[:npos].sum() - npos * (npos + 1) / 2) / (npos * len(neg))


def fit_apply(Xtr, ytr, Xte_p, Xte_n):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(Xtr, ytr)
    return (clf.predict_proba(Xte_p)[:, 1], clf.predict_proba(Xte_n)[:, 1])


def struct_feats(pairs, adj, deg):
    """[exists, common-neighbours, Adamic-Adar, log1p(d_u*d_v), log1p(d_u+d_v)]"""
    out = np.zeros((len(pairs), 5))
    for i, (a, b) in enumerate(pairs):
        na, nb = adj[a], adj[b]
        common = na & nb
        out[i, 0] = 1.0 if b in na else 0.0
        out[i, 1] = len(common)
        out[i, 2] = sum(1.0 / math.log(max(len(adj[c]), 2)) for c in common)
        out[i, 3] = math.log1p(deg[a] * deg[b])
        out[i, 4] = math.log1p(deg[a] + deg[b])
    return out


snaps = datasets[DATASET](cfg)
N, T = snaps[0].num_nodes, len(snaps)
PERM = np.random.default_rng(42).permutation(N)

# ---------------------------------------------------------------- pass 1: global
# Bases, candidate pairs and every SIDE feature are computed on the global
# cumulative union only. Nothing here sees the partition -> C-independent by
# construction, exactly as _spectral_step and cond_probe3.py are.
CACHE = Path(os.environ.get("CEILGEN_CACHE", "analysis/probes/.cache")) \
    / f"ceilgen_{DATASET}_k{K}_s{STRIDE}_m{MAX_POS}.pkl"
CACHE.parent.mkdir(parents=True, exist_ok=True)
print(f"[{DATASET}] N={N} T={T} — precomputing global bases + candidates...", flush=True)
t0 = time.time()
adj_g = [set() for _ in range(N)]
cumset = set()
new_edges, cand = {}, {}
_hit = pickle.loads(CACHE.read_bytes()) if CACHE.exists() else None
for t in range(T - 1):
    ne = und(snaps[t].edge_index)
    new_edges[t] = ne
    for a, b in ne:
        cumset.add((a, b))
        adj_g[a].add(b)
        adj_g[b].add(a)
    if _hit is not None:
        continue
    if t % STRIDE:
        continue
    pos = sorted(und(snaps[t + 1].edge_index))
    if len(pos) < MIN_POS or len(cumset) < 2:
        continue
    rng = np.random.default_rng(7000 + t)
    if len(pos) > MAX_POS:
        pos = [pos[i] for i in rng.choice(len(pos), MAX_POS, replace=False)]
    neg = []
    while len(neg) < len(pos):
        a, b = rng.integers(0, N, 2)
        if a == b or (min(a, b), max(a, b)) in cumset:
            continue
        neg.append((int(a), int(b)))
    e = torch.tensor([[a for a, b in cumset] + [b for a, b in cumset],
                      [b for a, b in cumset] + [a for a, b in cumset]], dtype=torch.long)
    g = Graph(x=torch.ones(N, 1), edge_index=e, node_ids=torch.arange(N))
    _, U = g.calc_eigs_exact_sym(K)
    Qn = U.numpy()
    nrm = np.linalg.norm(Qn, axis=1)
    Qp = Qn[PERM]
    nrp = nrm[PERM]

    def cosaff(pairs, Q, nr):
        out = np.zeros(len(pairs))
        for i, (a, b) in enumerate(pairs):
            if nr[a] > 0 and nr[b] > 0:
                out[i] = Q[a] @ Q[b] / (nr[a] * nr[b])
        return out

    deg_g = np.array([len(s) for s in adj_g])
    side = {}
    for nm, pr in (("p", pos), ("n", neg)):
        S = struct_feats(pr, adj_g, deg_g)
        side[nm] = {
            "spec": cosaff(pr, Qn, nrm),
            "exists": S[:, 0],
            "cn": S[:, 1],
            "aa": S[:, 2],
            "degprod": S[:, 3],
            "specperm": cosaff(pr, Qp, nrp),
        }
    rr = np.random.default_rng(9000 + t)
    side["p"]["noise"] = rr.standard_normal(len(pos))
    side["n"]["noise"] = rr.standard_normal(len(neg))
    cand[t] = (pos, neg, side)
if _hit is not None:
    cand = _hit
    print(f"  cache hit {CACHE}", flush=True)
else:
    CACHE.write_bytes(pickle.dumps(cand))
print(f"  {len(cand)} eval snapshots, {time.time()-t0:.0f}s", flush=True)

SIDE = ["spec", "exists", "cn", "aa", "degprod", "specperm", "noise"]
TS = sorted(cand)

# ------------------------------------------------- pass 2: A, sharded baseline
res = {}   # (C, arm) -> list over seeds of mean AUC
cuts = {}  # C -> list over seeds of (mean cut fraction, mean lost edges/snap)
base_scores = {}  # (C, seed) -> {t: (score_pos, score_neg)}  (for experiment B)
for C in CS:
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        owner = rng.choice(C, N)  # mirrors random_assign (graph_partitioning.py:267)
        adj_v = [set() for _ in range(N)]
        cf_t, ca_t = [], []
        feats = {}
        for t in range(T - 1):
            for a, b in new_edges[t]:
                if owner[a] == owner[b]:
                    adj_v[a].add(b)
                    adj_v[b].add(a)
            tot = len(new_edges[t])
            cut = sum(1 for a, b in new_edges[t] if owner[a] != owner[b])
            if tot:
                cf_t.append(cut / tot)
                ca_t.append(cut)
            if t not in cand:
                continue
            pos, neg, _ = cand[t]
            deg_v = np.array([len(s) for s in adj_v])
            feats[t] = (struct_feats(pos, adj_v, deg_v)[:, BCOLS],
                        struct_feats(neg, adj_v, deg_v)[:, BCOLS])
        cuts.setdefault(C, []).append((float(np.mean(cf_t)), float(np.mean(ca_t))))

        per_arm = {a: [] for a in ["base"] + SIDE}
        per_arm.update({a + "|rep": [] for a in ["base"] + SIDE})
        per_arm.update({a + "|new": [] for a in ["base"] + SIDE})
        repfrac = []
        bs = {}
        for i, t in enumerate(TS):
            past = TS[max(0, i - PWIN):i]
            if len(past) < WARM:
                continue
            pos, neg, side = cand[t]
            # §16's split: positives are split on cumulative-union membership,
            # the negative set is identical for both subsets.
            rm = side["p"]["exists"] > 0
            repfrac.append(float(rm.mean()))
            Bp, Bn = feats[t]

            def stack(cols):
                Xtr, ytr = [], []
                for tp in past:
                    po, ne, sd = cand[tp]
                    bp, bn = feats[tp]
                    ap = np.column_stack([bp] + [sd["p"][c] for c in cols])
                    an = np.column_stack([bn] + [sd["n"][c] for c in cols])
                    Xtr += [ap, an]
                    ytr += [np.ones(len(po)), np.zeros(len(ne))]
                tp_ = np.column_stack([Bp] + [side["p"][c] for c in cols])
                tn_ = np.column_stack([Bn] + [side["n"][c] for c in cols])
                return np.vstack(Xtr), np.concatenate(ytr), tp_, tn_

            def record(arm, sp, sn):
                per_arm[arm].append(auc(sp, sn))
                if rm.sum() >= MIN_POS:
                    per_arm[arm + "|rep"].append(auc(sp[rm], sn))
                if (~rm).sum() >= MIN_POS:
                    per_arm[arm + "|new"].append(auc(sp[~rm], sn))

            Xtr, ytr, tp_, tn_ = stack([])
            sp, sn = fit_apply(Xtr, ytr, tp_, tn_)
            record("base", sp, sn)
            bs[t] = (sp, sn)
            for c in SIDE:
                Xtr, ytr, tp_, tn_ = stack([c])
                a_p, a_n = fit_apply(Xtr, ytr, tp_, tn_)
                record(c, a_p, a_n)
        base_scores[(C, seed)] = bs
        for a in per_arm:
            if per_arm[a]:
                res.setdefault((C, a), []).append(float(np.mean(per_arm[a])))
        res.setdefault((C, "repfrac"), []).append(float(np.mean(repfrac)))
        print(f"  A: C{C} seed {seed} cut={np.mean(cf_t):.3f} "
              f"base={res[(C,'base')][-1]:.4f}", flush=True)

# ---------------------------------- pass 3: B, noise ladder at C=1 (no severing)
resB = {}
C0 = CS[0]
for seed in SEEDS:
    bs = base_scores[(C0, seed)]
    for sg in SIGMAS:
        nz = {}
        for t in bs:
            r = np.random.default_rng(int(1e6 + 1000 * sg) + t + 7 * seed)
            sp, sn = bs[t]
            z = np.concatenate([sp, sn])
            z = (z - z.mean()) / (z.std() + 1e-12)
            z = z + sg * r.standard_normal(len(z))
            nz[t] = (z[:len(sp)], z[len(sp):])
        keys = sorted(nz)
        per = {a: [] for a in ["base"] + SIDE}
        for i, t in enumerate(keys):
            past = keys[max(0, i - PWIN):i]
            if len(past) < WARM:
                continue
            _, _, side = cand[t]
            np_, nn_ = nz[t]
            per["base"].append(auc(np_, nn_))
            for c in SIDE:
                Xtr, ytr = [], []
                for tp in past:
                    po, ne, sd = cand[tp]
                    zp, zn = nz[tp]
                    Xtr += [np.column_stack([zp, sd["p"][c]]),
                            np.column_stack([zn, sd["n"][c]])]
                    ytr += [np.ones(len(po)), np.zeros(len(ne))]
                a_p, a_n = fit_apply(
                    np.vstack(Xtr), np.concatenate(ytr),
                    np.column_stack([np_, side["p"][c]]),
                    np.column_stack([nn_, side["n"][c]]))
                per[c].append(auc(a_p, a_n))
        for a in per:
            if per[a]:
                resB.setdefault((sg, a), []).append(float(np.mean(per[a])))
    print(f"  B: seed {seed} done", flush=True)

# ------------------------------------------------------------------- reporting
import statistics as st


def ms(d, key):
    v = d.get(key, [])
    return f"{st.mean(v):.4f}±{st.pstdev(v):.4f}" if v else "–"


def dd(d, a, b):
    va, vb = d.get(a, []), d.get(b, [])
    if not va or not vb:
        return "–"
    x = [p - q for p, q in zip(va, vb)]
    return f"{st.mean(x):+.4f}±{st.pstdev(x):.4f}"


hdr = f"{'C':>2s} {'cut':>12s} {'lost/snap':>9s} {'base alone':>15s}"
print(f"\n=== A. SHARDED MODEL-FREE BASELINE — {DATASET}, exact Q{K}, "
      f"{len(TS)} snaps, {len(SEEDS)} partition seeds, base={BASE_MODE} ===")
print("Delta = AUC(base + side feature) - AUC(base).  Every side feature is "
      "computed on the\nGLOBAL cumulative union and is therefore identical at every C.")
print(hdr + "".join(f"{('D '+c):>17s}" for c in SIDE))
for C in CS:
    cf = [x[0] for x in cuts[C]]
    ca = [x[1] for x in cuts[C]]
    row = (f"{C:>2d} {st.mean(cf):>6.3f}±{st.pstdev(cf):.3f} {st.mean(ca):>9.1f} "
           f"{ms(res,(C,'base')):>15s}")
    row += "".join(f"{dd(res,(C,c),(C,'base')):>17s}" for c in SIDE)
    print(row)

print(f"\n=== A2. THE SAME CEILING SPLIT REPEAT / NEW (§16 axis) — {DATASET} ===")
print("Positives split on cumulative-union membership; negatives identical in both.")
SPL = ["spec", "exists", "cn"]
print(f"{'C':>2s} {'rep frac':>8s} {'base REP':>15s} {'base NEW':>15s}"
      + "".join(f"{('D '+c+' REP'):>17s}{('D '+c+' NEW'):>17s}" for c in SPL))
for C in CS:
    row = (f"{C:>2d} {st.mean(res[(C,'repfrac')]):>8.3f} "
           f"{ms(res,(C,'base|rep')):>15s} {ms(res,(C,'base|new')):>15s}")
    for c in SPL:
        row += (f"{dd(res,(C,c+'|rep'),(C,'base|rep')):>17s}"
                f"{dd(res,(C,c+'|new'),(C,'base|new')):>17s}")
    print(row)

print(f"\n=== B. NOISE LADDER AT C={C0} — zero edges severed, {len(SEEDS)} seeds ===")
print("The C=1 baseline score is degraded by additive Gaussian noise. No partition.")
print(f"{'sigma':>6s} {'base alone':>15s}" + "".join(f"{('D '+c):>17s}" for c in SIDE))
for sg in SIGMAS:
    row = f"{sg:>6.1f} {ms(resB,(sg,'base')):>15s}"
    row += "".join(f"{dd(resB,(sg,c),(sg,'base')):>17s}" for c in SIDE)
    print(row)

def delta(d, key, c):
    return st.mean([x - y for x, y in zip(d[(key, c)], d[(key, "base")])])


print("\n=== C. Delta AS A FUNCTION OF BASELINE AUC (both experiments) ===")
SHOW = ["spec", "exists", "cn", "degprod"]
print(f"{'source':>16s} {'base AUC':>10s}" + "".join(f"{('D '+c):>11s}" for c in SHOW))
for C in CS:
    print(f"{('shard C'+str(C)):>16s} {st.mean(res[(C,'base')]):>10.4f}"
          + "".join(f"{delta(res,C,c):>+11.4f}" for c in SHOW))
for sg in SIGMAS:
    print(f"{('noise s='+str(sg)):>16s} {st.mean(resB[(sg,'base')]):>10.4f}"
          + "".join(f"{delta(resB,sg,c):>+11.4f}" for c in SHOW))

# Matched-baseline contrast: what the noise ladder predicts for a baseline as weak
# as sharding's, interpolated on baseline AUC. Any excess is sharding-SPECIFIC and
# cannot be explained by "the baseline simply got worse".
xb = [st.mean(resB[(sg, "base")]) for sg in SIGMAS][::-1]
print("\n--- matched-baseline contrast (noise ladder interpolated to the sharded "
      "baseline AUC) ---")
print(f"{'C':>2s} {'base AUC':>9s}" + "".join(
    f"{('D '+c+' shard'):>16s}{('noise-pred'):>12s}{('excess'):>9s}" for c in SHOW))
for C in CS:
    b = st.mean(res[(C, "base")])
    row = f"{C:>2d} {b:>9.4f}"
    for c in SHOW:
        yb = [delta(resB, sg, c) for sg in SIGMAS][::-1]
        pred = float(np.interp(b, xb, yb))
        row += f"{delta(res,C,c):>+16.4f}{pred:>+12.4f}{delta(res,C,c)-pred:>+9.4f}"
    print(row)
