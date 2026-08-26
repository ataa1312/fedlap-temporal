"""Is COLD START the binding constraint on new pairs?

The candidate: `Graph._active_lsym` keeps only nodes with cumulative degree > 0
and both solvers zero-pad the rest (graph.py:490, 548-549, 622-623), so a pair
with a never-yet-seen endpoint reaches `DynamicSInvariant.edge_score` as the
zero vector -- cos = 0, phi = 0, lev = log1p(0) = 0 -- and the MLP emits one
constant for every such pair. Repeat positives are cumulative edges, hence
always covered; new positives frequently are not. If that is what caps the
new-pair number, the spectral signal must be present on the COVERED new subset
and absent only on the uncovered one. The competing "recurrence" account
predicts no signal on covered new pairs either.

The test proposed upstream -- add a `covered` mask to
`dynamic_server._eval_mrr` and re-decode banked `f+es` checkpoints -- is NOT
model-free (it needs a trained decoder) and no checkpoints are banked in this
checkout. Substitution: score the pairs with the spectral quantity ITSELF,
inside the real MRR protocol. `results.md` 10.16 established that the whole
measured effect reduces to the unfiltered affinity cos(u,v) = <U_u/|U_u|,
U_v/|U_v|> (fed_dynamic_classifier.py:385-392), so "does the spectral term have
anything to offer this subset" is exactly "does that scalar rank this subset's
positives above the protocol's negatives". Everything downstream -- MRR
negatives, the filtered sampler, the per-source max aggregation, the
repeat/new mask -- is the pipeline's own code, called directly.

Three controls make the reading unambiguous:

  random      uniform scores through the same aggregation: the MRR chance floor
              for this K, which is NOT 0 and NOT 0.5.
  cov-neg     negatives redrawn from covered nodes only. Against the default
              uniform pool a covered positive beats every zero-row negative for
              free, which inflates every "covered" cell; this removes that.
  AA          Adamic-Adar on the same cumulative union: a genuinely structural,
              non-spectral readout. Separates "this subset is unpredictable
              from cumulative structure" from "the SPECTRAL readout throws the
              structure away" (cf. 10.12c, "the binding constraint is the
              READOUT").

usage: python analysis/probes/coldstart_coverage.py [dataset] [k] [stride]
"""
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import scipy.sparse as sparse
import scipy.stats as sst
import torch

_a = sys.argv[1:]
DATASET = _a[0] if _a else "uci"
K = int(_a[1]) if len(_a) > 1 else 300
STRIDE = int(_a[2]) if len(_a) > 2 else 1

sys.argv = ["coldstart_coverage", "-c", f"config/{DATASET}_gru.yaml", "--set",
            "model.data_type=feature", "subgraph.num_subgraphs=1", "wandb.mode=disabled"]
from parser import Parser

p = Parser()
cfg = p.load_config(p.parse_args())
import src

src.config = cfg
from registries import datasets
import src.datasets  # noqa: F401
from src.utils.graph import Graph
from src.dynamic_server import DynamicServer
from src.metrics.mrr import _sample_filtered_negatives, _rank_and_aggregate
from src.train.federated_orchestrator import _partition_edges_per_snapshot

MRR_K = cfg["experimental"]["rank_eval_multiplier"]
METHOD = cfg["metric"]["mrr_method"]
SEED = cfg["seed"]


class _MaskShim:
    """Duck-typed `self` for DynamicServer._repeat_mask -- it reads exactly two
    attributes, so the real method runs unmodified."""

    def __init__(self, snaps):
        self.global_snaps = snaps
        self._cum_edges = None


def auc(pos, neg):
    if len(pos) < 5 or len(neg) < 5:
        return np.nan
    s = np.concatenate([pos, neg])
    r = sst.rankdata(s)
    n = len(pos)
    return (r[:n].sum() - n * (n + 1) / 2) / (n * len(neg))


def pct(pos, neg):
    """Per-positive AUC contribution: the fraction of `neg` it beats (ties at
    0.5). Its mean over a cell IS that cell's AUC, so pooling these across
    snapshots gives a positive-count-weighted AUC and sidesteps the tiny
    per-snapshot cells on sparse datasets -- without pooling raw scores, which
    are not comparable across snapshots (different bases)."""
    if len(pos) == 0 or len(neg) < 5:
        return np.empty(0)
    ns = np.sort(neg)
    lo = np.searchsorted(ns, pos, "left")
    hi = np.searchsorted(ns, pos, "right")
    return (lo + hi) / (2.0 * len(ns))


def degree_matched_pct(pos_s, pos_tgt_deg, neg_s, neg_tgt_deg):
    """`pct` computed inside log2 target-degree buckets, so a cell whose targets
    are systematically more (or less) popular than the uniform negative pool is
    not scored on that difference. Buckets with <5 negatives are dropped."""
    pb = np.minimum(np.log2(np.maximum(pos_tgt_deg, 1)).astype(int), 8)
    nb = np.minimum(np.log2(np.maximum(neg_tgt_deg, 1)).astype(int), 8)
    out = []
    for b in np.unique(pb):
        m, mn = pb == b, nb == b
        if mn.sum() >= 5:
            out.append(pct(pos_s[m], neg_s[mn]))
    return np.concatenate(out) if out else np.empty(0)


def covered_negatives(sources, pos_u, pos_v, cov_idx, N, k, rng):
    """K negatives per source drawn from COVERED nodes only, with the same
    positive-collision resampling as _sample_filtered_negatives."""
    key = np.unique(pos_u.numpy().astype(np.int64) * N + pos_v.numpy().astype(np.int64))
    out = np.empty((len(sources), k), dtype=np.int64)
    for i, s in enumerate(sources.tolist()):
        draw = cov_idx[rng.integers(0, len(cov_idx), k)]
        for _ in range(8):
            bad = np.isin(s * N + draw, key) | (draw == s)
            if not bad.any():
                break
            draw[bad] = cov_idx[rng.integers(0, len(cov_idx), int(bad.sum()))]
        out[i] = draw
    return torch.from_numpy(out)


def mrr_cell(score_pos, score_neg, u, sources, keep):
    """The pipeline's own per-source aggregation, on arbitrary scores."""
    if keep is not None and not bool(keep.any()):
        return np.nan
    s = torch.cat([score_pos, score_neg.reshape(-1)])
    return _rank_and_aggregate(s, u, sources, sources.numel(), MRR_K, METHOD, keep=keep)


snaps = datasets[DATASET](cfg)
N, T = snaps[0].num_nodes, len(snaps)
_partition_edges_per_snapshot(snaps, cfg["dataset"]["split"], SEED)
shim = _MaskShim(snaps)

acc = {}
def add(key, val):
    if val == val:
        acc.setdefault(key, []).append(float(val))

pool = {}
def extend(key, arr):
    if len(arr):
        pool.setdefault(key, []).append(np.asarray(arr, dtype=float))

# (cos, lev) percentile coordinates per cell, for the optimal-readout ceiling
ceil_pool: dict[str, list] = {}


def ceiling_auc(recs, cols=(0, 1), bins=10, degree_matched=False):
    """Best AUC ANY function of the feature block can reach on this cell.

    A below-chance AUC is not "no signal" -- the smodel's MLP can invert a
    feature, and can be non-monotone. So bound what is actually there: bin the
    2-D (cos, lev) percentile plane, estimate the positive/negative density
    ratio (the Neyman-Pearson optimal discriminator) on one half of the
    snapshots, score the other half, and swap. Cross-fitting by snapshot means
    the number cannot be inflated by fitting the bins to the eval points.
    """
    if len(recs) < 4:
        return np.nan, 0
    cols = list(cols)
    idx = np.arange(len(recs))
    aucs, ns = [], 0
    for fit_par in (0, 1):
        fit = [i for i in idx if i % 2 == fit_par]
        ev = [i for i in idx if i % 2 != fit_par]
        P = np.concatenate([recs[i][0][:, cols] for i in fit])
        Ng = np.concatenate([recs[i][2][:, cols] for i in fit])
        if len(P) < 20 or not ev:
            continue
        edges = np.linspace(0, 1, bins + 1)
        edges[-1] += 1e-9
        hp = np.histogramdd(P, bins=[edges] * len(cols))[0] + 0.5
        hn = np.histogramdd(Ng, bins=[edges] * len(cols))[0] + 0.5
        lr = np.log(hp / hp.sum()) - np.log(hn / hn.sum())

        def sc(X):
            ix = tuple(np.clip((X[:, c] * bins).astype(int), 0, bins - 1)
                       for c in range(len(cols)))
            return lr[ix]

        u_, n_ = [], 0
        for i in ev:
            pXY, pdg, nXY, ndg = recs[i]
            if len(pXY) < 1:
                continue
            sp, sn = sc(pXY[:, cols]), sc(nXY[:, cols])
            v = degree_matched_pct(sp, pdg, sn, ndg) if degree_matched else pct(sp, sn)
            if len(v):
                u_.append(v)
                n_ += len(v)
        if u_:
            aucs.append(float(np.concatenate(u_).mean()))
            ns += n_
    return (float(np.mean(aucs)) if aucs else np.nan), ns

DEG_BUCKETS = [(1, 1), (2, 4), (5, 16), (17, 10 ** 9)]

print(f"# {DATASET}  N={N}  T={T}  k={K}  MRR_K={MRR_K}  method={METHOD}  "
      f"stride={STRIDE}  solver=calc_eigs_exact_sym")
print(f"{'t':>4} {'sec':>6} {'cov%':>6} {'pos':>6} {'rep%':>6} "
      f"{'newCov%':>8} {'m_rep':>7} {'m_newC':>7} {'m_newU':>7} {'m_rand':>7} "
      f"{'A_repC':>7} {'A_newC':>7} {'A_AAnewC':>9}")

rng = np.random.default_rng(20250820)
for t in range(T - 1):
    # cumulative undirected union up to t, exactly as _spectral_step accumulates it
    e = snaps[t].edge_index.cpu()
    e = torch.cat([e, e.flip(0)], dim=1)
    if shim._cum_edges is not None:
        e = torch.cat([shim._cum_edges, e], dim=1)
    shim._cum_edges = torch.unique(e, dim=1)
    if t % STRIDE or t == 0:
        continue
    pos = snaps[t + 1].pos_test
    if pos.size(1) < 5:  # sparse datasets have ~13 test positives per snapshot
        continue

    t0 = time.time()
    g = Graph(x=torch.ones(N, 1), edge_index=shim._cum_edges,
              node_ids=torch.arange(N))
    _, U = g.calc_eigs_exact_sym(K)
    dt = time.time() - t0
    U = U.numpy()
    nrm = np.linalg.norm(U, axis=1)
    covered = nrm > 0
    if covered.sum() < 50:
        continue
    Un = U / np.maximum(nrm, 1e-12)[:, None]
    cov_idx = np.where(covered)[0]

    # cumulative adjacency (binary, no self-loops) -- for AA and for degrees
    ce = shim._cum_edges.numpy()
    A = sparse.coo_matrix((np.ones(ce.shape[1]), (ce[0], ce[1])), shape=(N, N)).tocsr()
    A = ((A + A.T) > 0).astype(np.float64)
    A.setdiag(0)
    A.eliminate_zeros()
    cdeg = np.asarray(A.sum(1)).ravel()
    wlog = np.zeros(N)
    m = cdeg > 1
    wlog[m] = 1.0 / np.log(cdeg[m])
    Aw = A @ sparse.diags(wlog)

    u, v = pos[0], pos[1]
    rmask = DynamicServer._repeat_mask(shim, pos)          # real code
    sources = torch.unique(u)
    v_neg = _sample_filtered_negatives(sources, u, v, N, MRR_K, "cpu", None)  # real code
    v_negC = covered_negatives(sources, u, v, cov_idx, N, MRR_K, rng)

    un, vn = u.numpy(), v.numpy()
    cov_pair = torch.from_numpy(covered[un] & covered[vn])
    rep = rmask
    new = ~rmask

    # ---- scores: the affinity the smodel consumes (fed_dynamic_classifier:385-392)
    def aff(a, b):
        return torch.from_numpy((Un[a] * Un[b]).sum(1)).float()

    sp_pos = aff(un, vn)
    sp_neg = aff(np.repeat(sources.numpy(), MRR_K), v_neg.numpy().ravel())
    sp_negC = aff(np.repeat(sources.numpy(), MRR_K), v_negC.numpy().ravel())

    rnd_pos = torch.rand(pos.size(1))
    rnd_neg = torch.rand(sources.numel() * MRR_K)

    # ---- Adamic-Adar on the same union, one sparse row-op per source
    src_np = sources.numpy()
    aa_rows = {int(s): np.asarray((Aw[int(s)] @ A.T).todense()).ravel() for s in src_np}
    aa_pos = torch.from_numpy(
        np.array([aa_rows[int(a)][int(b)] for a, b in zip(un, vn)])).float()
    aa_negC = torch.from_numpy(
        np.concatenate([aa_rows[int(s)][v_negC[i].numpy()]
                        for i, s in enumerate(src_np)])).float()

    # ---- MRR cells, real aggregation
    m_rep = mrr_cell(sp_pos, sp_neg, u, sources, rep)
    m_newC = mrr_cell(sp_pos, sp_neg, u, sources, new & cov_pair)
    m_newU = mrr_cell(sp_pos, sp_neg, u, sources, new & ~cov_pair)
    m_rand = mrr_cell(rnd_pos, rnd_neg, u, sources, None)
    add("mrr_all", mrr_cell(sp_pos, sp_neg, u, sources, None))
    add("mrr_repeat", m_rep)
    add("mrr_new", mrr_cell(sp_pos, sp_neg, u, sources, new))
    add("mrr_new_cov", m_newC)
    add("mrr_new_unc", m_newU)
    add("mrr_random", m_rand)
    add("mrr_rand_new_cov", mrr_cell(rnd_pos, rnd_neg, u, sources, new & cov_pair))
    # same cells against the covered-only negative pool
    add("mrrC_repeat", mrr_cell(sp_pos, sp_negC, u, sources, rep))
    add("mrrC_new_cov", mrr_cell(sp_pos, sp_negC, u, sources, new & cov_pair))
    add("mrrC_random", mrr_cell(rnd_pos, rnd_neg, u, sources, None))
    add("mrrC_AA_new_cov", mrr_cell(aa_pos, aa_negC, u, sources, new & cov_pair))
    add("mrrC_AA_repeat", mrr_cell(aa_pos, aa_negC, u, sources, rep))

    # ---- AUC cells (covered positives vs covered negatives only)
    P, NG, NGC = sp_pos.numpy(), sp_neg.numpy(), sp_negC.numpy()
    a_repC = auc(P[(rep & cov_pair).numpy()], NGC)
    a_newC = auc(P[(new & cov_pair).numpy()], NGC)
    a_aa = auc(aa_pos.numpy()[(new & cov_pair).numpy()], aa_negC.numpy())
    add("auc_rep_cov", a_repC)
    add("auc_new_cov", a_newC)
    add("auc_new_unc", auc(P[(new & ~cov_pair).numpy()], NG))
    add("auc_AA_new_cov", a_aa)
    add("auc_AA_rep_cov", auc(aa_pos.numpy()[(rep & cov_pair).numpy()], aa_negC.numpy()))

    # ---- what the affinity actually looks like on each cell, and the degree
    # profile of each cell (a below-chance AUC needs this to be readable)
    negC_t = v_negC.numpy().ravel()
    add("aff_rep_cov", P[(rep & cov_pair).numpy()].mean())
    add("aff_new_cov", P[(new & cov_pair).numpy()].mean())
    add("aff_negC", NGC.mean())
    add("deg_tgt_rep_cov", cdeg[vn[(rep & cov_pair).numpy()]].mean())
    add("deg_tgt_new_cov", cdeg[vn[(new & cov_pair).numpy()]].mean())
    add("deg_tgt_negC", cdeg[negC_t].mean())

    # ---- pooled (positive-weighted) AUC + the degree-matched control
    mrc, mnc, mnu = (rep & cov_pair).numpy(), (new & cov_pair).numpy(), (new & ~cov_pair).numpy()
    src_of_pos = un
    negC_src = np.repeat(src_np, MRR_K)
    extend("P_rep_cov", pct(P[mrc], NGC))
    extend("P_new_cov", pct(P[mnc], NGC))
    extend("P_new_unc", pct(P[mnu], NG))
    extend("P_AA_rep_cov", pct(aa_pos.numpy()[mrc], aa_negC.numpy()))
    extend("P_AA_new_cov", pct(aa_pos.numpy()[mnc], aa_negC.numpy()))
    extend("Pdm_rep_cov",
           degree_matched_pct(P[mrc], cdeg[vn[mrc]], NGC, cdeg[negC_t]))
    extend("Pdm_new_cov",
           degree_matched_pct(P[mnc], cdeg[vn[mnc]], NGC, cdeg[negC_t]))
    extend("Pdm_AA_new_cov",
           degree_matched_pct(aa_pos.numpy()[mnc], cdeg[vn[mnc]],
                              aa_negC.numpy(), cdeg[negC_t]))
    extend("Pdm_rand_new_cov",
           degree_matched_pct(np.random.default_rng(t).random(mnc.sum()), cdeg[vn[mnc]],
                              np.random.default_rng(t + 7).random(len(NGC)), cdeg[negC_t]))

    # ---- (cos, lev) percentile coordinates for the optimal-readout ceiling.
    # lev = log1p(|U_u||U_v| * scale) is monotone in the product of row norms,
    # so the percentile transform makes n_scale irrelevant.
    lev_pos = nrm[un] * nrm[vn]
    lev_negC = nrm[np.repeat(src_np, MRR_K)] * nrm[negC_t]
    sub = rng.permutation(len(NGC))[:6000]
    negXY = np.stack([pct(NGC[sub], NGC), pct(lev_negC[sub], lev_negC),
                      np.random.default_rng(t + 99).random(len(sub))], 1)
    ndg = cdeg[negC_t][sub]
    for cell, msk in (("rep_cov", mrc), ("new_cov", mnc)):
        if msk.sum() >= 3:
            pXY = np.stack([pct(P[msk], NGC), pct(lev_pos[msk], lev_negC),
                            np.random.default_rng(t + 5).random(int(msk.sum()))], 1)
            ceil_pool.setdefault(cell, []).append((pXY, cdeg[vn[msk]], negXY, ndg))

    # ---- coverage census
    add("cov_frac_nodes", covered.mean())
    add("cov_frac_negpool", covered[v_neg.numpy().ravel()].mean())
    add("rep_frac", rep.float().mean().item())
    add("cov_frac_rep", cov_pair[rep].float().mean().item() if rep.any() else np.nan)
    add("cov_frac_new", cov_pair[new].float().mean().item() if new.any() else np.nan)
    # sources whose ENTIRE new-positive set is uncovered: nothing spectral can help
    lost = 0
    tot = 0
    for s in sources.tolist():
        sel = (u == s) & new
        if bool(sel.any()):
            tot += 1
            lost += int(not bool(cov_pair[sel].any()))
    add("frac_new_sources_all_uncovered", lost / tot if tot else np.nan)

    # ---- graded coverage: is it a cliff (cold start) or a slope?
    mind = np.minimum(cdeg[un], cdeg[vn])
    for lo, hi in DEG_BUCKETS:
        sel = ((new & cov_pair).numpy()) & (mind >= lo) & (mind <= hi)
        add(f"auc_new_cov_mindeg_{lo}", auc(P[sel], NGC))
        add(f"n_new_cov_mindeg_{lo}", sel.sum())

    print(f"{t:>4d} {dt:>6.1f} {100*covered.mean():>6.1f} {pos.size(1):>6d} "
          f"{100*rep.float().mean():>6.1f} {100*cov_pair[new].float().mean():>8.1f} "
          f"{m_rep:>7.4f} {m_newC:>7.4f} {m_newU:>7.4f} {m_rand:>7.4f} "
          f"{a_repC:>7.3f} {a_newC:>7.3f} {a_aa:>9.3f}")

print(f"\n=== {DATASET}: mean over {len(acc.get('mrr_all', []))} evaluated transitions ===")
order = [
    "cov_frac_nodes", "cov_frac_negpool", "rep_frac", "cov_frac_rep", "cov_frac_new",
    "frac_new_sources_all_uncovered",
    "mrr_random", "mrr_all", "mrr_repeat", "mrr_new", "mrr_new_cov", "mrr_new_unc",
    "mrr_rand_new_cov",
    "mrrC_random", "mrrC_repeat", "mrrC_new_cov", "mrrC_AA_repeat", "mrrC_AA_new_cov",
    "auc_rep_cov", "auc_new_cov", "auc_new_unc", "auc_AA_rep_cov", "auc_AA_new_cov",
    "aff_rep_cov", "aff_new_cov", "aff_negC",
    "deg_tgt_rep_cov", "deg_tgt_new_cov", "deg_tgt_negC",
]
for lo, _ in DEG_BUCKETS:
    order += [f"auc_new_cov_mindeg_{lo}", f"n_new_cov_mindeg_{lo}"]
for key in order:
    if key in acc:
        arr = np.array(acc[key])
        print(f"  {key:32s} {arr.mean():+.4f}  (sd {arr.std():.4f}, n={len(arr)})")

print(f"\n--- pooled AUC (positive-weighted over all evaluated transitions) ---")
for key in ["P_rep_cov", "P_new_cov", "P_new_unc", "P_AA_rep_cov", "P_AA_new_cov",
            "Pdm_rep_cov", "Pdm_new_cov", "Pdm_AA_new_cov", "Pdm_rand_new_cov"]:
    if key in pool:
        a = np.concatenate(pool[key])
        se = a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else np.nan
        print(f"  {key:20s} AUC {a.mean():.4f} +- {se:.4f} (se)   n_pos={len(a)}")

print("\n--- optimal-readout ceiling, cross-fitted by snapshot (0.5 = the block "
      "holds nothing about this cell, in EITHER sign) ---")
VARIANTS = [("cos+lev", (0, 1), False), ("cos only", (0,), False),
            ("lev only", (1,), False), ("random ctl", (2,), False),
            ("cos+lev degmatched", (0, 1), True),
            ("lev only degmatched", (1,), True),
            ("random ctl degmatched", (2,), True)]
for cell in ("rep_cov", "new_cov"):
    if cell not in ceil_pool:
        continue
    for label, cols, dm in VARIANTS:
        a, n = ceiling_auc(ceil_pool[cell], cols, degree_matched=dm)
        print(f"  ceiling[{cell:8s}] {label:22s} AUC {a:.4f}   n_eval={n}")

print("""
reading:
  mrr_*        MRR of the SPECTRAL AFFINITY ALONE through the pipeline's own
               per-source aggregation. mrr_random is the chance floor.
  mrrC_*       same, negatives redrawn from covered nodes only (removes the
               free win a covered positive gets over a zero-row negative).
  auc_*_cov    covered positives vs covered negatives; 0.5 = chance.
  AA           Adamic-Adar on the same cumulative union: is the subset
               predictable from structure at all?
  If cold start is the binding constraint, mrr_new_cov / auc_new_cov must sit
  clearly above chance. If they sit at chance while auc_AA_new_cov does not,
  the constraint is the spectral readout, not the zero rows.""")
