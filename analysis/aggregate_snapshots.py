"""Read per-(seed, snapshot) records and report the snapshot axis four ways.

Every headline number in results.md means over SNAPSHOTS first and only then over
seeds, which cannot distinguish a uniform effect from one carried by a subset of
snapshots, and cannot test whether a deficit is a warm-up artifact of the sparse
early cumulative graph. This reads the JSONL records main.py writes and reports:

  per-snapshot   mean +- spread ACROSS SEEDS at each t, with the contributing-seed
                 count, so a t backed by one seed is not read as one backed by all
  distribution   mean/median/Q1/Q3/min/max OVER SNAPSHOTS -- the readable form when
                 a dataset has 733 of them
  early/late     both groups at a stated boundary, with basis coverage per group
  weighted       subset metrics weighted by each snapshot's positive count, beside
                 the unweighted snapshot mean -- they answer different questions and
                 diverge whenever the counts vary

Read-only: never re-runs a model, never writes into the training path.

usage: python analysis/aggregate_snapshots.py <record.jsonl> [more.jsonl ...]
              [--metric mrr_new] [--early 0.2] [--vs <arm-substring>]
       --early takes a FRACTION of run length (default 0.2) or an integer >= 1 for
       a fixed snapshot count.
"""
import json
import os
import statistics as st
import sys
from collections import defaultdict

# subset metric -> the per-snapshot count that weights it
WEIGHT_OF = {"mrr_repeat": "n_repeat", "mrr_new": "n_new"}
COVERAGE = ("basis_covered", "basis_zeroed", "basis_total", "basis_zeroed_pair_frac")


def _flag(name, cast, default):
    if name in sys.argv:
        i = sys.argv.index(name)
        v = cast(sys.argv[i + 1])
        del sys.argv[i : i + 2]
        return v
    return default


def load(paths):
    """arm -> t -> seed -> row."""
    out = defaultdict(lambda: defaultdict(dict))
    for p in paths:
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                out[r["arm"]][r["t"]][r["seed"]] = r
    return out


def _finite(vals):
    # nan drops out per key: a metric undefined at a snapshot for one seed must not
    # be averaged as 0, and must not poison the other seeds' contribution either.
    return [v for v in vals if isinstance(v, (int, float)) and v == v]


def per_snapshot(arm_rows, metric):
    """t -> (mean across seeds, spread, contributing-seed count)."""
    out = {}
    for t, by_seed in sorted(arm_rows.items()):
        vals = _finite([r.get(metric) for r in by_seed.values()])
        if not vals:
            continue
        out[t] = (st.mean(vals), st.pstdev(vals) if len(vals) > 1 else 0.0, len(vals))
    return out


def distribution(curve):
    """mean/median/quartiles/min/max OVER SNAPSHOTS of a per-snapshot curve."""
    v = sorted(m for m, _, _ in curve.values())
    if not v:
        return None
    q = st.quantiles(v, n=4) if len(v) > 3 else [v[0], st.median(v), v[-1]]
    return {
        "n_snapshots": len(v), "mean": st.mean(v), "median": st.median(v),
        "q1": q[0], "q3": q[2], "min": v[0], "max": v[-1],
    }


def weighted(arm_rows, metric):
    """sum(m_t * n_t) / sum(n_t) over snapshots with n_t > 0.

    A snapshot contributing no positives to the subset has an UNDEFINED metric
    there; counting it as a zero is the error that would make a dataset whose
    snapshots are mostly empty in one subset look far worse than it is.
    """
    wkey = WEIGHT_OF.get(metric)
    if not wkey:
        return None
    num = den = 0.0
    for by_seed in arm_rows.values():
        for r in by_seed.values():
            m, n = r.get(metric), r.get(wkey)
            if m is None or n is None or m != m or not n:
                continue
            num += m * n
            den += n
    return num / den if den else None


def seed_order_mean(arm_rows, metric):
    """Mean over SEEDS of each seed's mean over snapshots.

    This is the order every banked number in results.md uses. It differs from the
    snapshot-order mean whenever snapshots have unequal seed coverage -- which is
    routine, since mrr_new is nan on any snapshot with no new positives. Both are
    legitimate; reporting only one makes the printed mean irreconcilable with the
    record, so both are printed and labelled.
    """
    per_seed = {}
    for by_seed in arm_rows.values():
        for seed, r in by_seed.items():
            v = r.get(metric)
            if isinstance(v, (int, float)) and v == v:
                per_seed.setdefault(seed, []).append(v)
    means = [st.mean(v) for v in per_seed.values() if v]
    return (st.mean(means), len(means)) if means else (None, 0)


def split(curve, arm_rows, boundary):
    """Early group vs the rest, with basis coverage per group where present."""
    ts = sorted(curve)
    if not ts:
        return None
    # int(), not the raw arg: --early parses as a float, so the documented fixed
    # count path handed ts[:5.0] to the slice and raised.
    k = int(boundary) if boundary >= 1 else max(1, int(round(boundary * len(ts))))
    k = min(k, len(ts) - 1) if len(ts) > 1 else len(ts)
    groups = {"early": ts[:k], "late": ts[k:]}
    out = {"boundary_snapshots": k, "boundary_arg": boundary}
    for name, sel in groups.items():
        vals = [curve[t][0] for t in sel]
        cov = _finite([
            r.get("basis_zeroed_pair_frac")
            for t in sel for r in arm_rows[t].values()
        ])
        out[name] = {
            "n_snapshots": len(vals),
            "mean": st.mean(vals) if vals else None,
            "median": st.median(vals) if vals else None,
            "zeroed_pair_frac": st.mean(cov) if cov else None,
        }
    return out


def compare(a_curve, b_curve):
    """Mean difference AND the fraction of snapshots where a leads b.

    A positive mean difference with a win fraction at or below one half means the
    advantage is carried by a minority of snapshots, which the mean alone hides.
    Ties count as neither.
    """
    shared = sorted(set(a_curve) & set(b_curve))
    if not shared:
        return None
    diffs = [a_curve[t][0] - b_curve[t][0] for t in shared]
    wins = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)
    # win_frac is wins/n_shared, so ties sit in the denominator. Reporting only
    # wins makes a tie indistinguishable from a loss; report all three.
    return {
        "n_shared": len(shared), "mean_diff": st.mean(diffs),
        "median_diff": st.median(diffs), "wins": wins,
        "losses": losses, "ties": len(shared) - wins - losses,
        "win_frac": wins / len(shared),
    }


def main():
    metric = _flag("--metric", str, "mrr_new")
    early = _flag("--early", float, 0.2)
    vs = _flag("--vs", str, None)
    paths = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not paths:
        print(__doc__)
        return 1
    data = load(paths)
    print(f"metric={metric}  early-boundary-arg={early}\n")
    curves = {}
    for arm, rows in sorted(data.items()):
        c = per_snapshot(rows, metric)
        if not c:
            continue
        curves[arm] = c
        d = distribution(c)
        w = weighted(rows, metric)
        sp = split(c, rows, early)
        seeds = sorted({s for by in rows.values() for s in by})
        short = arm.split("_C")[-1] if "_C" in arm else arm
        print(f"### {arm}")
        print(f"  seeds={seeds}  snapshots={d['n_snapshots']}")
        # AGGREGATION ORDER is stated: this is mean-over-snapshots of the
        # cross-seed mean, which equals the run-level mean only when every
        # snapshot has the same contributing-seed count.
        print(f"  over-snapshots of cross-seed mean : mean={d['mean']:.4f} "
              f"median={d['median']:.4f} q1={d['q1']:.4f} q3={d['q3']:.4f} "
              f"min={d['min']:.4f} max={d['max']:.4f}")
        so_, n_seeds = seed_order_mean(rows, metric)
        if so_ is not None:
            agree = abs(so_ - d["mean"]) < 1e-12
            print(f"  over-seeds of per-seed snapshot mean : mean={so_:.4f} "
                  f"(n_seeds={n_seeds})  <- the order results.md uses"
                  f"{'  [agrees]' if agree else '  [DIFFERS: unequal seed coverage]'}")
        if w is not None:
            print(f"  pair-weighted (by {WEIGHT_OF[metric]})       : {w:.4f}"
                  f"   [unweighted snapshot mean above; they answer different questions]")
        if sp:
            e, l = sp["early"], sp["late"]
            # a 2-snapshot run leaves the late group empty; report it as empty
            # rather than dying in the format string.
            fmt = lambda g: (f"n={g['n_snapshots']} mean=" +
                             ("n/a" if g["mean"] is None else f"{g['mean']:.4f}") +
                             (f" zpair={g['zeroed_pair_frac']:.3f}" if g["zeroed_pair_frac"] is not None else ""))
            print(f"  early/late @ first {sp['boundary_snapshots']} snapshots:")
            print(f"     early: {fmt(e)}")
            print(f"     late : {fmt(l)}")
            if e["mean"] is not None and l["mean"] is not None:
                print(f"     late - early = {l['mean'] - e['mean']:+.4f}")
        print()
    if vs:
        base = [a for a in curves if vs in a]
        if len(base) != 1:
            print(f"--vs {vs!r} matched {len(base)} arms; need exactly 1")
            return 1
        b = base[0]
        print(f"### pairwise vs {b}")
        for arm, c in sorted(curves.items()):
            if arm == b:
                continue
            cmp = compare(c, curves[b])
            if cmp:
                print(f"  {arm.split('_C')[0][:40]:<40} mean_diff={cmp['mean_diff']:+.4f} "
                      f"median_diff={cmp['median_diff']:+.4f} "
                      f"wins={cmp['wins']}/{cmp['n_shared']} ({cmp['win_frac']:.0%}) "
                      f"losses={cmp['losses']} ties={cmp['ties']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
