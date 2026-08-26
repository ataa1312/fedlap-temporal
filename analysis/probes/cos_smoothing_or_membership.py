"""Is `cos(u,v)` a SMOOTHING score or a MEMBERSHIP score?

The objection under test: the lowest eigenvectors of L_sym are a low-pass filter,
so `cos` should behave like the multi-hop proximity that message passing already
computes -- and a term that only re-supplies smoothing cannot give the SAME gain
at C=1 (nothing severed) as at C=9. §16 measured the repeat-subset gain as
+0.0909 at C1 vs +0.0567 at C9, i.e. LARGER where nothing is cut, so the
objection is well posed.

The competing account: `cos` is computed on the CUMULATIVE union of every edge up
to t (`dynamic_server._spectral_step`, :784-789), while the encoder message-passes
over snapshot t only. So the deficit it fills would be TEMPORAL, not spatial, and
`cos` would be a soft indicator of "uv is already an edge of the union" rather
than a proximity measure.

The two accounts make opposite predictions once MEMBERSHIP IS HELD FIXED:

    smoothing   -> cos keeps ranking skill WITHIN the already-an-edge group and
                   WITHIN the not-yet-an-edge group, like CN/Adamic-Adar do
    membership  -> cos's skill collapses to chance inside both groups; all of it
                   lived in the between-group contrast

Everything here is model-free: no training, no forward pass. `cos` is computed
verbatim as `fed_dynamic_classifier.edge_score` does it
(`src/GNN/fed_dynamic_classifier.py:378-396`) -- row-normalise U, elementwise
product, sum over the k columns -- at the model's real `k = spectral.pe_dim = 50`
(NOT spectral_len=300: a previous probe used 300 and its conclusion reversed when
corrected), on the exact sym-Laplacian basis of the cumulative union.

Why the exact solver is the right stand-in for the chebyshev one the §16 runs used:
`cos` is algebraically `P_uv / sqrt(P_uu P_vv)` with `P = U U^T`, so it depends on
the SUBSPACE only and is invariant to any orthogonal rotation of the k columns
(verified numerically to 7e-16). The chebyshev solver lands on the exact subspace
with overlap 1.000 (§15/§10.12b), so it computes the same `cos` -- and the same
argument makes sign-canonicalisation and the procrustes step irrelevant here.

Scores compared on the SAME pairs
  cos     the model's spectral affinity, k=50
  mem     1 if uv is an edge of the cumulative union up to t, else 0  (pure memory)
  CNcum   common neighbours on the cumulative union   (2-hop smoothing, full history)
  AAcum   Adamic-Adar on the cumulative union         (2-hop smoothing, degree-weighted)
  CNt     common neighbours on SNAPSHOT t only        (what the encoder can see)
  PA      preferential attachment, deg_u*deg_v on the union -- popularity with NO
          pairwise structure at all; the null any residual "skill" has to clear
  RECENT  index of the last snapshot in which uv appeared (-1 if never)
  COUNT   how many past snapshots uv appeared in
          RECENT/COUNT are the POSITIVE CONTROL for the matched-membership test:
          they are memory WITH gradation, so if they rank well inside the
          already-an-edge group then that group is not degenerate and cos's
          chance-level score there is a fact about cos, not about the test.

Pair pools
  POSrep  evaluated test positive at t+1, already an edge of the union   (member)
  POSnew  evaluated test positive at t+1, not yet an edge of the union   (non-member)
  NEGuni  uniform random pair excluding all of t+1's edges -- the protocol's own
          negative distribution (`_attach_future_link_pred_labels`, mrr.py)
  NEGmem  random pair that IS an edge of the union but does NOT recur at t+1
          -- the matched control the protocol never draws enough of

Pooling. Every AUC is built from per-positive win rates against that SNAPSHOT's
own negative pool, so the pooled number is a legitimate AUC even though scores
are not comparable across snapshots (each snapshot has its own basis). Low-
recurrence datasets (bitcoin_otc: ~8% repeat) have 1-2 repeat positives per
snapshot, where a per-snapshot AUC is meaningless -- read POOLED there and treat
the per-snapshot spread as the sampling check.

usage: python analysis/probes/cos_smoothing_or_membership.py [dataset ...]
                 [--marks 12] [--k 50] [--neg 8000] [--srck 50] [--maxpos 800]
                 [--minpos 8] [--minedges 10] [--solver exact|cheb] [--rng 0]
                 [--posplit test|all]
       --marks 0 uses every snapshot (needed for bitcoin_otc's repeat columns).

reproduced with: uci --marks 12 | bitcoin_otc --marks 0 | as733 --marks 40
"""
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import random as pyrandom

import numpy as np
import scipy.sparse as sparse
import scipy.stats as sst
import torch

DEFAULT = ["uci", "bitcoin_otc", "as733"]
SCORES = ["cos", "mem", "CNcum", "AAcum", "CNt", "PA", "RECENT", "COUNT"]
COLS = ["all", "rep", "new", "mMEM", "mNON", "detect"]


def _flag(name, cast, default):
    if name in sys.argv:
        i = sys.argv.index(name)
        v = cast(sys.argv[i + 1])
        del sys.argv[i : i + 2]
        return v
    return default


MARKS = _flag("--marks", int, 12)
K = _flag("--k", int, 50)
NNEG = _flag("--neg", int, 8000)
SRCK = _flag("--srck", int, 50)
MAXPOS = _flag("--maxpos", int, 800)
MINPOS = _flag("--minpos", int, 8)
MINEDGES = _flag("--minedges", int, 10)
SOLVER = _flag("--solver", str, "exact")
RNG_SEED = _flag("--rng", int, 0)
POSSPLIT = _flag("--posplit", str, "test")   # 'all' = every edge of t+1, for power
                                             # on low-recurrence graphs; 'test' is
                                             # the set the real eval actually ranks
TARGETS = [a for a in sys.argv[1:] if not a.startswith("-")] or DEFAULT

sys.argv = ["cos_smoothing_or_membership", "-c", "config/uci_gru.yaml", "--set",
            "model.data_type=feature", "subgraph.num_subgraphs=1", "wandb.mode=disabled"]
from parser import Parser

p = Parser()
cfg0 = p.load_config(p.parse_args())
import src

src.config = cfg0
from registries import datasets
import src.datasets  # noqa: F401
from src.train.federated_orchestrator import _partition_edges_per_snapshot, _pos_for_split
from src.utils.graph import Graph


def _seed(s):
    """main.py::_seed -- keeps the edge split identical to a real run."""
    pyrandom.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def winrates(pos, neg):
    """Per-positive Mann-Whitney win rate against `neg`; ties count half. Ties
    matter: a node outside the giant component has an identically-zero basis row,
    so its cos is exactly 0.0, as is CN for most random pairs. The mean of this
    is the ROC-AUC, and the per-element form is what makes pooling across
    snapshots legitimate."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return np.empty(0)
    ns = np.sort(neg)
    lo = np.searchsorted(ns, pos, "left")
    hi = np.searchsorted(ns, pos, "right")
    return (lo + 0.5 * (hi - lo)) / ns.size


def cond_winrates(pos_s, neg_s):
    """Source-conditioned win rates: each positive is ranked only against
    negatives that SHARE ITS SOURCE, which is exactly MRR's geometry
    (mrr.py:194-206)."""
    if pos_s.size == 0 or neg_s.size == 0:
        return np.empty(0)
    w = (neg_s < pos_s[:, None]).sum(1) + 0.5 * (neg_s == pos_s[:, None]).sum(1)
    return w / neg_s.shape[1]


def ukey(u, v, N):
    u = np.asarray(u, dtype=np.int64)
    v = np.asarray(v, dtype=np.int64)
    return np.minimum(u, v) * N + np.maximum(u, v)


def ms(vals):
    a = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if a.size == 0:
        return "n/a"
    if a.size == 1:
        return f"{a[0]:.3f}(n1)"
    return f"{a.mean():.3f}±{a.std(ddof=1):.3f}"


def pooled(chunks):
    a = [c for c in chunks if c.size]
    if not a:
        return float("nan"), 0
    a = np.concatenate(a)
    return float(a.mean()), a.size


def partial_r(x, y, z):
    """Correlation of x with y after both are residualised on z, on RANKS.

    CN/AA/PA are heavy-tailed counts and Pearson on raw values understates them;
    ranks make (f) the same kind of statistic as the AUCs in (a)-(e). Ranking a
    binary variable is monotone, so `y` and `z` are unaffected."""
    x = sst.rankdata(x)
    rxy = np.corrcoef(x, y)[0, 1]
    rxz = np.corrcoef(x, z)[0, 1]
    ryz = np.corrcoef(y, z)[0, 1]
    d = np.sqrt(max(1 - rxz**2, 1e-12) * max(1 - ryz**2, 1e-12))
    return (rxy - rxz * ryz) / d, rxy, rxz


def reject_sample(N, srcs, rng, forbid, extra_forbid=None, rounds=40):
    """A destination for each entry of `srcs`: uniform, no self-loops, avoiding
    the `forbid` key array and optionally `extra_forbid`."""
    w = rng.integers(0, N, srcs.size)
    for _ in range(rounds):
        bad = w == srcs
        k = ukey(srcs, w, N)
        bad |= np.isin(k, forbid)
        if extra_forbid is not None:
            bad |= np.isin(k, extra_forbid)
        if not bad.any():
            break
        w[bad] = rng.integers(0, N, int(bad.sum()))
    return w


def table(head, cols, get):
    print(f"  {'score':>6} | " + " ".join(f"{c:>14}" for c in cols) + f"   {head}")
    for s in SCORES:
        print(f"  {s:>6} | " + " ".join(f"{get(s, c):>14}" for c in cols))


print(f"cos as in fed_dynamic_classifier.edge_score, k={K} ({SOLVER} sym-Laplacian "
      f"basis of the cumulative union), up to {MARKS} snapshots per dataset")
print("AUC 0.500 = chance. Cells are POOLED over positives; per-snapshot mean±std "
      "follows in its own table.")
print("The 'MATCHED ON MEMBERSHIP' block is the decisive test.\n")

for name in TARGETS:
    cfg = p.load_config(p.parse_args(
        ["-c", f"config/{name}_gru.yaml", "--set", "model.data_type=feature",
         "subgraph.num_subgraphs=1", "wandb.mode=disabled"]))
    src.config = cfg

    _seed(cfg["seed"])
    snaps = datasets[name](cfg)
    N, T = snaps[0].num_nodes, len(snaps)
    split_seed = cfg["dataset"]["split_seed"]
    if split_seed is None:
        split_seed = cfg["seed"]
    _partition_edges_per_snapshot(snaps, cfg["dataset"]["split"], split_seed)

    n_tasks = T - 1
    marks = (set(range(1, n_tasks)) if MARKS <= 0 else
             set(np.unique(np.linspace(1, n_tasks - 1, MARKS).astype(int)).tolist()))

    gp = {s: {c: [] for c in COLS} for s in SCORES}     # global, per-positive rates
    cp = {s: {c: [] for c in COLS} for s in SCORES}     # source-conditioned
    gs = {s: {c: [] for c in COLS} for s in SCORES}     # global, per-snapshot AUC
    cs = {s: {c: [] for c in COLS} for s in SCORES}     # cond,   per-snapshot AUC
    qq = {s: [] for s in SCORES}      # per-snapshot balanced 2x2, pooled at the end
    corr = {s: {c: [] for c in ("r_mem", "r_lab", "part")} for s in SCORES}
    meta = {c: [] for c in ("npos", "repfrac", "zero", "nmemneg", "srcmem", "nrep")}
    rows = []

    cum = None
    hk = np.empty(0, dtype=np.int64)      # sorted keys of the union
    hlast = np.empty(0, dtype=np.int64)   # last snapshot each key was seen in
    hcnt = np.empty(0, dtype=np.int64)    # how many snapshots each key appeared in
    for t in range(n_tasks):
        e = snaps[t].edge_index.cpu()
        e = torch.cat([e, e.flip(0)], dim=1)
        if cum is not None:
            e = torch.cat([cum, e], dim=1)
        cum = torch.unique(e, dim=1)

        # per-edge history, kept as sorted arrays so lookups stay vectorised
        sk0 = snaps[t].edge_index.cpu().numpy()
        sk = np.unique(ukey(sk0[0], sk0[1], N))
        pos_ = np.searchsorted(hk, sk)
        hit = np.zeros(sk.size, dtype=bool)
        if hk.size:
            ok_ = pos_ < hk.size
            hit[ok_] = hk[pos_[ok_]] == sk[ok_]
        hlast[pos_[hit]] = t
        hcnt[pos_[hit]] += 1
        fresh = sk[~hit]
        if fresh.size:
            hk = np.concatenate([hk, fresh])
            hlast = np.concatenate([hlast, np.full(fresh.size, t)])
            hcnt = np.concatenate([hcnt, np.ones(fresh.size, dtype=np.int64)])
            o = np.argsort(hk)
            hk, hlast, hcnt = hk[o], hlast[o], hcnt[o]

        if t not in marks:
            continue

        pos = (snaps[t + 1].edge_index if POSSPLIT == "all"
               else _pos_for_split(snaps[t + 1], "test")).cpu().numpy().astype(np.int64)
        if POSSPLIT == "all":                       # both directions are stored
            uu = np.unique(ukey(pos[0], pos[1], N))
            pos = np.stack([uu // N, uu % N])
        if pos.shape[1] < MINEDGES:
            continue
        rng = np.random.default_rng(RNG_SEED + t)

        # ---- the basis, exactly as _spectral_step serves it for f+es ---- #
        g = Graph(x=torch.ones(N, 1), edge_index=cum, node_ids=torch.arange(N))
        _, act = g._active_lsym()
        if act is None or act.size <= K + 2:
            continue
        if SOLVER == "exact":
            _, U = g.calc_eigs_exact_sym(K)
        else:
            _, U = g.calc_eigs_chebyshev(K)
        U = U.numpy()
        Un = U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)

        # ---- adjacency of the cumulative union and of snapshot t ---- #
        ce = cum.numpy()
        A = sparse.coo_matrix((np.ones(ce.shape[1]), (ce[0], ce[1])), shape=(N, N)).tocsr()
        A = ((A + A.T) > 0).astype(np.float64)
        A.setdiag(0)
        A.eliminate_zeros()
        deg = np.asarray(A.sum(1)).ravel()
        aa_w = np.where(deg > 1, 1.0 / np.log(np.maximum(deg, 1.0000001)), 0.0)
        se = snaps[t].edge_index.cpu().numpy()
        As = sparse.coo_matrix((np.ones(se.shape[1]), (se[0], se[1])), shape=(N, N)).tocsr()
        As = ((As + As.T) > 0).astype(np.float64)
        As.setdiag(0)
        As.eliminate_zeros()

        cumkeys = np.unique(ukey(ce[0], ce[1], N))
        tom = snaps[t + 1].edge_index.cpu().numpy()
        forbid = np.unique(ukey(tom[0], tom[1], N))     # the protocol's forbidden set

        def hist(k):
            i = np.searchsorted(hk, k)
            j = np.minimum(i, max(hk.size - 1, 0))
            f = (i < hk.size) & (hk[j] == k) if hk.size else np.zeros(k.size, bool)
            return (np.where(f, hlast[j], -1).astype(np.float64),
                    np.where(f, hcnt[j], 0).astype(np.float64))

        def score(u, v):
            u = np.asarray(u, dtype=np.int64)
            v = np.asarray(v, dtype=np.int64)
            M = A[u].multiply(A[v])
            Ms = As[u].multiply(As[v])
            last, cnt = hist(ukey(u, v, N))
            return {
                "RECENT": last,
                "COUNT": cnt,
                "cos": (Un[u] * Un[v]).sum(1),
                "mem": np.isin(ukey(u, v, N), cumkeys).astype(np.float64),
                "CNcum": np.asarray(M.sum(1)).ravel(),
                "AAcum": np.asarray(M @ aa_w).ravel(),
                "CNt": np.asarray(Ms.sum(1)).ravel(),
                "PA": deg[u] * deg[v],
            }

        # ---------------- global pools ---------------- #
        pu, pv = pos[0], pos[1]
        prep = np.isin(ukey(pu, pv, N), cumkeys)
        s_pos = score(pu, pv)

        nu = rng.integers(0, N, NNEG)
        nv = reject_sample(N, nu, rng, forbid)
        keep = nu != nv
        nu, nv = nu[keep], nv[keep]
        nmem = np.isin(ukey(nu, nv, N), cumkeys)
        s_neg = score(nu, nv)
        non = np.where(~nmem)[0]
        rng.shuffle(non)
        ref, cell = non[: non.size // 2], non[non.size // 2 :]   # ref pool vs NEGnon cell

        # NEGmem: edges of the union that do NOT recur at t+1 -- the matched control
        cand = cumkeys[~np.isin(cumkeys, forbid)]
        if cand.size:
            pick = rng.choice(cand.size, size=min(NNEG, cand.size), replace=False)
            mu, mv = cand[pick] // N, cand[pick] % N
            s_mneg = score(mu, mv)
        else:
            mu = np.array([], dtype=np.int64)
            s_mneg = {s: np.empty(0) for s in SCORES}

        in_gc = np.zeros(N, dtype=bool)
        in_gc[act] = True
        meta["npos"].append(pos.shape[1])
        meta["nrep"].append(int(prep.sum()))
        meta["repfrac"].append(float(prep.mean()))
        meta["zero"].append(float((~(in_gc[pu] & in_gc[pv])).mean()))
        meta["nmemneg"].append(int(mu.size))

        def put(store_p, store_s, s, c, w, n_min=MINPOS):
            store_p[s][c].append(w)
            if w.size >= n_min:
                store_s[s][c].append(float(w.mean()))

        for s in SCORES:
            put(gp, gs, s, "all", winrates(s_pos[s], s_neg[s]))
            put(gp, gs, s, "rep", winrates(s_pos[s][prep], s_neg[s]))
            put(gp, gs, s, "new", winrates(s_pos[s][~prep], s_neg[s]))
            # (e) matched on membership
            put(gp, gs, s, "mMEM", winrates(s_pos[s][prep], s_mneg[s]))
            put(gp, gs, s, "mNON", winrates(s_pos[s][~prep], s_neg[s][~nmem]))
            # (b') no labels involved: does the score itself say "already an edge"?
            put(gp, gs, s, "detect", winrates(s_mneg[s], s_neg[s][cell]))
            # (f) balanced 2x2, BUILT WITHIN THIS SNAPSHOT then pooled: balancing
            # only after pooling would let the four cells draw different snapshot
            # mixtures and read between-snapshot variation as a partial effect.
            # Scores are not comparable across snapshots (each has its own basis),
            # so what gets pooled is each pair's win rate against a HELD-OUT
            # non-member reference pool from its own snapshot.
            cells = (winrates(s_pos[s][prep], s_neg[s][ref]),
                     winrates(s_pos[s][~prep], s_neg[s][ref]),
                     winrates(s_mneg[s], s_neg[s][ref]),
                     winrates(s_neg[s][cell], s_neg[s][ref]))
            m = min(c.size for c in cells)
            if m >= 5:
                for c, lb, mb in zip(cells, (1, 1, 0, 0), (1, 0, 1, 0)):
                    idx = rng.choice(c.size, m, replace=False)
                    qq[s].append((c[idx], np.full(m, float(lb)), np.full(m, float(mb))))

        # ---------------- (f) correlations on the protocol pool ---------------- #
        for s in SCORES:
            x = np.concatenate([s_pos[s], s_neg[s]])
            lab = np.concatenate([np.ones(s_pos[s].size), np.zeros(s_neg[s].size)])
            mem = np.concatenate([prep.astype(float), nmem.astype(float)])
            if x.std() > 0 and mem.std() > 0:
                pr, rxy, rxz = partial_r(x, lab, mem)
                corr[s]["r_lab"].append(rxy)
                corr[s]["r_mem"].append(rxz)
                corr[s]["part"].append(pr)

        # ---------------- source-conditioned (MRR geometry) ---------------- #
        sel = np.arange(pos.shape[1])
        if sel.size > MAXPOS:
            sel = rng.choice(sel, MAXPOS, replace=False)
        su, srep = pu[sel], prep[sel]
        sp = {s: v[sel] for s, v in s_pos.items()}

        rep_src = np.repeat(su, SRCK)
        s_uni = score(rep_src, reject_sample(N, rep_src, rng, forbid))
        s_non = score(rep_src, reject_sample(N, rep_src, rng, forbid, extra_forbid=cumkeys))

        # per-source member negatives: past partners of u that do NOT recur at t+1
        ok, w_mem = [], []
        for i, uu in enumerate(su):
            nb = A.indices[A.indptr[uu] : A.indptr[uu + 1]]
            nb = nb[nb != uu]
            nb = nb[~np.isin(ukey(np.full(nb.size, uu), nb, N), forbid)]
            if nb.size == 0:
                continue
            ok.append(i)
            w_mem.append(rng.choice(nb, SRCK, replace=nb.size < SRCK))
        ok = np.asarray(ok, dtype=np.int64)
        meta["srcmem"].append(float(ok.size / max(su.size, 1)))
        s_mem = (score(np.repeat(su[ok], SRCK), np.stack(w_mem).ravel())
                 if ok.size else None)

        for s in SCORES:
            U_ = s_uni[s].reshape(su.size, SRCK)
            NO = s_non[s].reshape(su.size, SRCK)
            put(cp, cs, s, "all", cond_winrates(sp[s], U_))
            put(cp, cs, s, "rep", cond_winrates(sp[s][srep], U_[srep]))
            put(cp, cs, s, "new", cond_winrates(sp[s][~srep], U_[~srep]))
            put(cp, cs, s, "mNON", cond_winrates(sp[s][~srep], NO[~srep]))
            if s_mem is not None:
                m = srep[ok]
                M_ = s_mem[s].reshape(ok.size, SRCK)
                put(cp, cs, s, "mMEM", cond_winrates(sp[s][ok][m], M_[m]))

        rows.append((t, pos.shape[1], int(prep.sum()),
                     pooled(gp["cos"]["all"][-1:])[0], pooled(gp["cos"]["rep"][-1:])[0],
                     pooled(gp["cos"]["new"][-1:])[0], pooled(gp["cos"]["mMEM"][-1:])[0],
                     pooled(gp["cos"]["mNON"][-1:])[0]))

    n = len(meta["npos"])
    if n == 0:
        print(f"=== {name}: no usable snapshot\n")
        continue

    # ---------------- (f) pooled balanced 2x2 ---------------- #
    bal = {}
    for s in SCORES:
        if not qq[s]:
            bal[s] = None
            continue
        bx = np.concatenate([a for a, _, _ in qq[s]])
        bl = np.concatenate([a for _, a, _ in qq[s]])
        bm = np.concatenate([a for _, _, a in qq[s]])
        bal[s] = partial_r(bx, bl, bm) + (bx.size // 4,) if bx.std() > 0 else None

    print(f"=== {name}  N={N}  T={T}  tasks={n_tasks}  snapshots used={n}  "
          f"positives={POSSPLIT}"
          f"{'   [n<3 -- single-snapshot numbers]' if n < 3 else ''}")
    print(f"    positives/snapshot {np.mean(meta['npos']):.0f} "
          f"(repeat {np.mean(meta['nrep']):.1f}, fraction {np.mean(meta['repfrac']):.3f}), "
          f"zero-basis-row positives {np.mean(meta['zero']):.3f}, "
          f"NEGmem pool {np.mean(meta['nmemneg']):.0f}, "
          f"sources with a member-negative {np.mean(meta['srcmem']):.3f}")

    def gpool(s, c):
        v, k = pooled(gp[s][c])
        return f"{v:.3f}" if np.isfinite(v) else "n/a"

    def cpool(s, c):
        v, k = pooled(cp[s][c])
        return f"{v:.3f}" if np.isfinite(v) else "n/a"

    npair = {c: pooled(gp["cos"][c])[1] for c in COLS}
    print(f"\n  POOLED AUC. positives pooled per column: "
          + ", ".join(f"{c}={npair[c]}" for c in COLS))
    print(f"\n  (a-d) vs the protocol's own negatives | (e) matched | (b') detection")
    table("GLOBAL pool", ["all", "rep", "new", "mMEM", "mNON", "detect"], gpool)
    print()
    table("SOURCE-CONDITIONED (MRR geometry)",
          ["all", "rep", "new", "mMEM", "mNON"], cpool)

    print(f"\n  same numbers as per-snapshot mean±std (cells with >={MINPOS} "
          f"positives only; n varies by column)")
    table("GLOBAL", ["all", "rep", "new", "mMEM", "mNON", "detect"],
          lambda s, c: ms(gs[s][c]))
    print()
    table("SOURCE-CONDITIONED", ["all", "rep", "new", "mMEM", "mNON"],
          lambda s, c: ms(cs[s][c]))

    print(f"\n  (e') share of each score's REPEAT-column skill that SURVIVES "
          f"matching on membership,\n       (AUC_mMEM - 0.5) / (AUC_rep - 0.5); "
          f"~0 means the skill WAS membership")
    print(f"  {'score':>6} | {'global':>14} {'source-cond':>14}")
    for s in SCORES:
        out = []
        for st in (gp, cp):
            r = pooled(st[s]["rep"])[0] - 0.5
            m = pooled(st[s]["mMEM"])[0] - 0.5
            out.append(f"{m / r:>13.1%}" if np.isfinite(r) and abs(r) > 0.01 else
                       f"{'n/a':>13}")
        print(f"  {s:>6} | {out[0]:>14} {out[1]:>14}")

    print(f"\n  (f) rank correlations. 'prot' = protocol pool, per-snapshot "
          f"mean±std. 'bal' = pooled balanced 2x2")
    print(f"  {'score':>6} | {'prot r(.,mem)':>14} {'prot r(.,lab)':>14} "
          f"{'prot r|mem':>14} | {'bal r(.,mem)':>14} {'bal r(.,lab)':>14} "
          f"{'bal r|mem':>14}")
    for s in SCORES:
        b = bal[s]
        bs = ((f"{b[2]:>14.3f} {b[1]:>14.3f} {b[0]:>14.3f}") if b else
              f"{'n/a':>14} {'n/a':>14} {'n/a':>14}")
        print(f"  {s:>6} | {ms(corr[s]['r_mem']):>14} {ms(corr[s]['r_lab']):>14} "
              f"{ms(corr[s]['part']):>14} | {bs}")
    if bal["cos"]:
        print(f"        (balanced per snapshot then pooled: {bal['cos'][3]} pairs per "
              f"cell, so membership and label are orthogonal by construction)")

    print(f"\n  per-snapshot detail for cos (global pool)")
    print(f"  {'t':>6} {'npos':>7} {'nrep':>6} {'all':>7} {'repeat':>7} {'new':>7} "
          f"{'mMEMBER':>8} {'mNONMEM':>8}")
    for r in rows:
        f = lambda x: f"{x:.3f}" if np.isfinite(x) else "  n/a"
        print(f"  {r[0]:>6d} {r[1]:>7d} {r[2]:>6d} {f(r[3]):>7} {f(r[4]):>7} "
              f"{f(r[5]):>7} {f(r[6]):>8} {f(r[7]):>8}")
    print()

print("Reading: if cos is a low-pass/diffusion score it keeps skill in the two "
      "matched columns\n(mMEM, mNON), as CNcum/AAcum do. If it is a memory of the "
      "cumulative union, those columns\nfall to 0.5 while 'detect' stays high -- "
      "i.e. cos was tracking 'mem', which is 0.5 in the\nmatched columns BY "
      "CONSTRUCTION. PA is the popularity null: residual skill below PA is not\n"
      "evidence of smoothing.")
