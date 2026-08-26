"""Can the ZERO-FEATURE / GIANT-COMPONENT truncation explain the null on new pairs?

`Graph._active_lsym` (src/utils/graph.py:472-502) builds L_sym from the ADJACENCY
of the cumulative union alone, keeps only nodes with degree > 0, then keeps only
the largest connected component of that. Every dropped node is zero-padded back
(graph.py:548-549, 622-623), so its basis row is identically zero and every
spectral quantity the readout computes for a pair touching it — cos, the
heat-kernel bank, the leverage term — is exactly 0. Such a pair gets a learned
CONSTANT and cannot be ranked at all.

If new pairs disproportionately touch those nodes, that alone would explain why
the spectral term restores ~114% of sharding's cost on repeat pairs and ~0% on
new pairs (results.md §16). If new pairs are covered as well as repeat pairs, the
explanation is refuted and the null has to come from the affinity itself.

Everything here is model-free: no training, no forward pass, no eigensolve for
the coverage table (the zero set is decided by connectivity alone). The optional
--marks pass does solve for a basis, purely to ask whether the pairs that ARE
covered are ranked any better than chance.

Definitions used below
  active   node with degree > 0 in the cumulative undirected union up to t
  GC       largest connected component of the active subgraph -- exactly `act`
           as returned by _active_lsym, i.e. the rows that are NOT zero
  zero     an evaluated positive with >= 1 endpoint outside GC
  never    zero because an endpoint has never appeared in the union (deg 0)
  offGC    zero only because an endpoint sits in a satellite component
  repeat   the pair is already an edge of the cumulative union (server's
           _repeat_mask, dynamic_server.py:769-778)
  xc       at C clients under the run's own partitioner, the two endpoints are
           owned by different clients, so no client ever holds that edge

Two rows close every table. ALL pools over evaluated pairs; MEAN averages the
per-snapshot fractions, which is the weighting `mean_mrr` uses and the one that
reproduces §16's reported repeat_frac of 0.443 on uci.

usage: python analysis/probes/zero_feature_coverage.py [dataset ...]
                 [--C 9] [--rows 30] [--marks 8] [--k 50] [--solver exact|cheb]
       --marks 0 solves at every snapshot (fine for uci, slow for as733).
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
import scipy.stats as sst
import torch

DEFAULT = ["uci", "bitcoin_otc", "as733"]


def _flag(name, cast, default):
    if name in sys.argv:
        i = sys.argv.index(name)
        v = cast(sys.argv[i + 1])
        del sys.argv[i : i + 2]
        return v
    return default


C_CLIENTS = _flag("--C", int, 9)
ROWS = _flag("--rows", int, 30)
MARKS = _flag("--marks", int, 8)
K = _flag("--k", int, 50)
SOLVER = _flag("--solver", str, "exact")
TARGETS = [a for a in sys.argv[1:] if not a.startswith("-")] or DEFAULT

sys.argv = ["zero_feature_coverage", "-c", "config/uci_gru.yaml", "--set",
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
from src.utils.graph_partitioning import random_assign


def _seed(s):
    """main.py::_seed -- the partitioner's RNG state must match a real run."""
    pyrandom.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def auc(pos, neg):
    """Mann-Whitney ROC-AUC. Ties count half, which is the point: a zero-feature
    pair scores exactly 0.0, and so do many negatives."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    s = np.concatenate([pos, neg])
    r = sst.rankdata(s)
    n = len(pos)
    return (r[:n].sum() - n * (n + 1) / 2) / (n * len(neg))


def pct(a, b):
    return 100.0 * a / b if b else float("nan")


print(f"C={C_CLIENTS} clients (partitioner + seed as in main.py::run_once), "
      f"basis k={K} solver={SOLVER} at {MARKS} marked snapshots")
print("'zero' = evaluated positive with >=1 endpoint outside the giant component "
      "of the cumulative union\n(never = endpoint never seen, offGC = endpoint in "
      "a satellite component). Splits are POOLED over pairs.\n")

for name in TARGETS:
    cfg = p.load_config(p.parse_args(
        ["-c", f"config/{name}_gru.yaml", "--set", "model.data_type=feature",
         f"subgraph.num_subgraphs={C_CLIENTS}", "wandb.mode=disabled"]))
    src.config = cfg

    # main.py::run_once ordering: seed, load, partition. random_assign is the
    # first thing partition_snapshots does, so this reproduces the real owners.
    _seed(cfg["seed"])
    snaps = datasets[name](cfg)
    N, T = snaps[0].num_nodes, len(snaps)
    owner = np.full(N, -1, dtype=np.int64)
    for c, ids in random_assign(N, C_CLIENTS).items():
        owner[ids.cpu().numpy()] = c

    split_seed = cfg["dataset"]["split_seed"]
    if split_seed is None:
        split_seed = cfg["seed"]
    _partition_edges_per_snapshot(snaps, cfg["dataset"]["split"], split_seed)

    n_tasks = T - 1
    stride = max(1, -(-n_tasks // ROWS))
    # marks span the WHOLE run: the early snapshots hold most of the new pairs,
    # so sampling only the tail would starve the covered-new AUC of data.
    marks = (set(range(n_tasks)) if MARKS <= 0 else
             set(np.unique(np.linspace(1, n_tasks - 1, MARKS).astype(int)).tolist()))

    print(f"=== {name}  N={N}  T={T}  tasks={n_tasks}  "
          f"(rows every {stride} snapshots) ===")
    print(f"{'t':>5} {'act':>7} {'gc%':>6} {'npos':>7} {'rep%':>6} | "
          f"{'zero%':>6} {'zREP%':>6} {'zNEW%':>6} | {'nevN%':>6} {'ogcN%':>6} | "
          f"{'xc%':>6} {'z|xc%':>6} {'z|in%':>6}")

    cum = None
    tot = dict(pos=0, rep=0, new=0, zrep=0, znew=0, nevrep=0, ognrep=0,
               nevnew=0, ognnew=0, xc=0, zxc=0, ixc=0, zixc=0,
               xcrep=0, xcnew=0, nact=0, ngc=0, snaps=0, gcfrac=0.0,
               srcz=0, dstz=0)
    ab = {kk: [] for kk in ("rep", "new", "cvrep", "cvnew", "zrnew", "neg", "cvneg")}
    # per-snapshot fractions too: the headline MRR is a MEAN OVER SNAPSHOTS
    # (dynamic_server aggregates mrr_history), not a pooled-over-pairs number,
    # so the snapshot-mean row is the one that reconciles with results.md §16.
    per = {kk: [] for kk in ("gc", "rep", "zero", "zrep", "znew", "nevnew",
                             "ognnew", "xc", "zxc", "zixc")}

    for t in range(n_tasks):
        # cumulative undirected union up to t, exactly as _spectral_step builds it
        e = snaps[t].edge_index.cpu()
        e = torch.cat([e, e.flip(0)], dim=1)
        if cum is not None:
            e = torch.cat([cum, e], dim=1)
        cum = torch.unique(e, dim=1)

        g = Graph(x=torch.ones(N, 1), edge_index=cum, node_ids=torch.arange(N))
        Lsym, act = g._active_lsym()
        ei = cum.numpy()
        seen = np.zeros(N, dtype=bool)
        seen[np.unique(ei)] = True          # degree > 0 in the union
        in_gc = np.zeros(N, dtype=bool)
        if act is not None:
            in_gc[act] = True
        nact, ngc = int(seen.sum()), int(in_gc.sum())

        pos = _pos_for_split(snaps[t + 1], "test").cpu().numpy().astype(np.int64)
        npos = pos.shape[1]
        if npos == 0:
            continue
        u, v = pos[0], pos[1]

        key = np.unique(np.minimum(ei[0], ei[1]) * N + np.maximum(ei[0], ei[1]))
        rep = np.isin(np.minimum(u, v) * N + np.maximum(u, v), key)
        new = ~rep
        zero = ~(in_gc[u] & in_gc[v])
        never = ~(seen[u] & seen[v])        # zero because an endpoint is unseen
        ogc = zero & ~never                 # zero only via a satellite component
        xc = owner[u] != owner[v]

        tot["snaps"] += 1
        tot["nact"] += nact
        tot["ngc"] += ngc
        tot["gcfrac"] += (ngc / nact) if nact else 0.0
        tot["pos"] += npos
        tot["rep"] += int(rep.sum())
        tot["new"] += int(new.sum())
        tot["zrep"] += int((zero & rep).sum())
        tot["znew"] += int((zero & new).sum())
        tot["nevrep"] += int((never & rep).sum())
        tot["ognrep"] += int((ogc & rep).sum())
        tot["nevnew"] += int((never & new).sum())
        tot["ognnew"] += int((ogc & new).sum())
        tot["xc"] += int(xc.sum())
        tot["zxc"] += int((zero & xc).sum())
        tot["ixc"] += int((~xc).sum())
        tot["zixc"] += int((zero & ~xc).sum())
        tot["xcrep"] += int((xc & rep).sum())
        tot["xcnew"] += int((xc & new).sum())
        # MRR negatives share the POSITIVE's source (mrr.py:194-206), so a
        # zero-featured SOURCE makes the spectral term a constant across every
        # candidate and it cancels out of the rank entirely; only a zero
        # DESTINATION under a covered source can move a rank.
        tot["srcz"] += int((new & ~in_gc[u]).sum())
        tot["dstz"] += int((new & in_gc[u] & ~in_gc[v]).sum())

        per["gc"].append(pct(ngc, nact))
        per["rep"].append(pct(rep.sum(), npos))
        per["zero"].append(pct(zero.sum(), npos))
        per["zrep"].append(pct((zero & rep).sum(), rep.sum()))
        per["znew"].append(pct((zero & new).sum(), new.sum()))
        per["nevnew"].append(pct((never & new).sum(), new.sum()))
        per["ognnew"].append(pct((ogc & new).sum(), new.sum()))
        per["xc"].append(pct(xc.sum(), npos))
        per["zxc"].append(pct((zero & xc).sum(), xc.sum()))
        per["zixc"].append(pct((zero & ~xc).sum(), (~xc).sum()))

        if t % stride == 0 or t == n_tasks - 1:
            print(f"{t:>5d} {nact:>7d} {pct(ngc, nact):>6.1f} {npos:>7d} "
                  f"{pct(rep.sum(), npos):>6.1f} | {pct(zero.sum(), npos):>6.1f} "
                  f"{pct((zero & rep).sum(), rep.sum()):>6.1f} "
                  f"{pct((zero & new).sum(), new.sum()):>6.1f} | "
                  f"{pct((never & new).sum(), new.sum()):>6.1f} "
                  f"{pct((ogc & new).sum(), new.sum()):>6.1f} | "
                  f"{pct(xc.sum(), npos):>6.1f} "
                  f"{pct((zero & xc).sum(), xc.sum()):>6.1f} "
                  f"{pct((zero & ~xc).sum(), (~xc).sum()):>6.1f}")

        # -- optional: are the COVERED pairs ranked any better than chance? --
        if t in marks and act is not None and act.size > K + 2:
            if SOLVER == "exact":
                _, U = g.calc_eigs_exact_sym(K)
            else:
                _, U = g.calc_eigs_chebyshev(K)
            U = U.numpy()
            Un = U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
            aff = (Un[u] * Un[v]).sum(1)     # the 'cos' feature, verbatim
            rng = np.random.default_rng(0)
            keyset = set(key.tolist())
            nn_all, nn_cov, want = [], [], max(npos, 2000)
            gc_ids = np.where(in_gc)[0]
            while len(nn_all) < want:
                a, b = int(rng.integers(0, N)), int(rng.integers(0, N))
                if a != b and (min(a, b) * N + max(a, b)) not in keyset:
                    nn_all.append((a, b))
            while len(nn_cov) < want and gc_ids.size > 2:
                a = int(gc_ids[rng.integers(0, gc_ids.size)])
                b = int(gc_ids[rng.integers(0, gc_ids.size)])
                if a != b and (min(a, b) * N + max(a, b)) not in keyset:
                    nn_cov.append((a, b))
            na, nc = np.array(nn_all), np.array(nn_cov)
            s_all = (Un[na[:, 0]] * Un[na[:, 1]]).sum(1)
            s_cov = (Un[nc[:, 0]] * Un[nc[:, 1]]).sum(1)
            cov = ~zero
            ab["rep"].append(aff[rep])
            ab["new"].append(aff[new])
            ab["cvrep"].append(aff[rep & cov])
            ab["cvnew"].append(aff[new & cov])
            ab["zrnew"].append(aff[new & zero])
            ab["neg"].append(s_all)
            ab["cvneg"].append(s_cov)

    n = tot["snaps"]
    P, R, W = tot["pos"], tot["rep"], tot["new"]
    print(f"{'ALL':>5} {tot['nact'] / n:>7.0f} {100 * tot['gcfrac'] / n:>6.1f} "
          f"{P:>7d} {pct(R, P):>6.1f} | {pct(tot['zrep'] + tot['znew'], P):>6.1f} "
          f"{pct(tot['zrep'], R):>6.1f} {pct(tot['znew'], W):>6.1f} | "
          f"{pct(tot['nevnew'], W):>6.1f} {pct(tot['ognnew'], W):>6.1f} | "
          f"{pct(tot['xc'], P):>6.1f} {pct(tot['zxc'], tot['xc']):>6.1f} "
          f"{pct(tot['zixc'], tot['ixc']):>6.1f}")
    nm = {kk: float(np.nanmean(vv)) for kk, vv in per.items()}
    print(f"{'MEAN':>5} {'':>7} {nm['gc']:>6.1f} {'':>7} {nm['rep']:>6.1f} | "
          f"{nm['zero']:>6.1f} {nm['zrep']:>6.1f} {nm['znew']:>6.1f} | "
          f"{nm['nevnew']:>6.1f} {nm['ognnew']:>6.1f} | {nm['xc']:>6.1f} "
          f"{nm['zxc']:>6.1f} {nm['zixc']:>6.1f}")

    print(f"  1. giant component / active nodes: {100 * tot['gcfrac'] / n:.2f}% "
          f"(mean over {n} snapshots); mean active {tot['nact'] / n:.0f} of {N}")
    print(f"  2. evaluated positives with a zero row: "
          f"{tot['zrep'] + tot['znew']}/{P} = {pct(tot['zrep'] + tot['znew'], P):.2f}%")
    print(f"  3. repeat {tot['zrep']}/{R} = {pct(tot['zrep'], R):.2f}%   "
          f"new {tot['znew']}/{W} = {pct(tot['znew'], W):.2f}%   "
          f"(new/repeat ratio {(pct(tot['znew'], W) / pct(tot['zrep'], R)) if tot['zrep'] else float('inf'):.1f}x)")
    print(f"     of the NEW zeros: never-seen endpoint {pct(tot['nevnew'], W):.2f}pp, "
          f"satellite-component endpoint {pct(tot['ognnew'], W):.2f}pp")
    print(f"     of the REPEAT zeros: never-seen {pct(tot['nevrep'], R):.2f}pp, "
          f"satellite {pct(tot['ognrep'], R):.2f}pp")
    print(f"     snapshot-mean (the weighting mean_mrr uses): repeat "
          f"{nm['zrep']:.2f}%   new {nm['znew']:.2f}%")
    print(f"     of the NEW zeros, source-side (spectral term cancels in the rank) "
          f"{pct(tot['srcz'], W):.2f}pp, destination-only {pct(tot['dstz'], W):.2f}pp")
    print(f"  4. C={C_CLIENTS}: cross-client positives {tot['xc']}/{P} = "
          f"{pct(tot['xc'], P):.2f}%  (repeat {pct(tot['xcrep'], R):.1f}% of repeat, "
          f"new {pct(tot['xcnew'], W):.1f}% of new)")
    print(f"     of cross-client positives, zero row: {tot['zxc']}/{tot['xc']} = "
          f"{pct(tot['zxc'], tot['xc']):.2f}%   "
          f"same-client: {tot['zixc']}/{tot['ixc']} = {pct(tot['zixc'], tot['ixc']):.2f}%")

    if ab["rep"] and any(len(x) for x in ab["rep"]):
        cat = {kk: np.concatenate(vv) if vv else np.array([]) for kk, vv in ab.items()}
        print(f"  affinity cos(u,v) on the same basis (k={K}, {SOLVER}), pooled over "
              f"{len(ab['rep'])} marked snapshots:")
        print(f"     mean  repeat {cat['rep'].mean():+.4f}  new {cat['new'].mean():+.4f}"
              f"  negatives {cat['neg'].mean():+.4f}")
        print(f"     AUC vs random negatives   repeat {auc(cat['rep'], cat['neg']):.3f}"
              f"   new {auc(cat['new'], cat['neg']):.3f}")
        print(f"     AUC restricted to COVERED pairs and COVERED negatives   "
              f"repeat {auc(cat['cvrep'], cat['cvneg']):.3f}"
              f"   new {auc(cat['cvnew'], cat['cvneg']):.3f}"
              f"   (n_new={cat['cvnew'].size})")
        # same negative pool for all three, so the mixture is exactly decomposed:
        # how much of the new-pair deficit is the zero rows, and how much is the
        # affinity being uninformative on new pairs it CAN see?
        print(f"     AUC vs the SAME random negatives   new-all "
              f"{auc(cat['new'], cat['neg']):.3f}   new-covered "
              f"{auc(cat['cvnew'], cat['neg']):.3f}   new-zero "
              f"{auc(cat['zrnew'], cat['neg']):.3f}   repeat-covered "
              f"{auc(cat['cvrep'], cat['neg']):.3f}")
    print()

print("Read: column zNEW% vs zREP% is the crux. A large gap means the spectral term "
      "is UNDEFINED on\nnew pairs and the coverage story survives; but it can only "
      "account for the null up to zNEW% of\nthe new pairs -- the covered remainder "
      "is settled by the last AUC line, where 0.5 is chance.")
