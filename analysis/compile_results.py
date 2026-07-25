"""Compile run-log RESULT lines into mean+-std tables, a CSV, and a coverage report.

Usage (from the repo root, over NAS run dirs or any dir of *.log files):
  python analysis/compile_results.py <run_dir> [<run_dir> ...] \
      [--csv out.csv] [--metric mean_mrr] [--group dataset,pipeline,mode,basis,decoder,depth,clients] \
      [--coverage] [--min-seeds 3]

Each *.log may hold several runs (--repeat); every RESULT line becomes one row.
Condition metadata (pipeline/mode/basis/decoder/depth) is recovered from the log
filename tokens; the RESULT line supplies dataset/clients/seed and all metrics.
"""

import argparse
import csv
import re
import statistics
import sys
from pathlib import Path

RESULT_RE = re.compile(r"RESULT (dataset=\S.*)$")
KV_RE = re.compile(r"(\w+)=([^\s]+)")

MODES = ("keep", "update", "recompute")
BASES = ("random_fixed", "shuffled_fixed", "laplacian", "random", "shuffled")  # _fixed first
DECODERS = ("cosine_similarity", "dot")
FLOAT_KEYS = ("mean_mrr", "std", "auc", "ap", "f1", "mcc")


def parse_stem(stem):
    toks = stem.split("_")
    meta = {"pipeline": "", "mode": "", "basis": "", "decoder": "concat", "depth": ""}
    joined = "_" + stem + "_"
    for dec in DECODERS:
        if f"_{dec}_" in joined:
            meta["decoder"] = dec
            break
    for m in MODES:
        if m in toks:
            meta["mode"] = m
            break
    for b in BASES:
        if f"_{b}_" in joined or joined.endswith(f"_{b}_") or f"_{b}_C" in joined:
            meta["basis"] = b
            break
    for t in toks:
        if re.fullmatch(r"L\d+", t):
            meta["depth"] = t[1:]
            break
    if "feature" in toks:
        meta["pipeline"] = "feature"
    elif "SignNet" in toks:
        meta["pipeline"] = "f+s-signnet"
    elif "pe" in toks:
        meta["pipeline"] = "f+pe"
    elif "add" in toks or "concat" in toks:
        meta["pipeline"] = "f+s-laplace-" + ("concat" if "concat" in toks else "add")
    return meta


def harvest(dirs):
    rows = []
    for d in dirs:
        for f in sorted(Path(d).glob("*.log")):
            meta = parse_stem(f.stem)
            for line in f.read_text(errors="replace").splitlines():
                m = RESULT_RE.search(line)
                if not m:
                    continue
                row = {"dir": Path(d).name, "file": f.stem, **meta}
                for k, v in KV_RE.findall(m.group(1)):
                    row[k] = float(v) if k in FLOAT_KEYS else v
                rows.append(row)
    return rows


def group_rows(rows, keys, metric):
    groups = {}
    for r in rows:
        if metric not in r:
            continue
        groups.setdefault(tuple(str(r.get(k, "")) for k in keys), []).append(r[metric])
    return groups


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--csv")
    ap.add_argument("--metric", default="mean_mrr")
    ap.add_argument("--group", default="dataset,pipeline,mode,basis,decoder,depth,clients")
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--min-seeds", type=int, default=3)
    args = ap.parse_args(argv)

    rows = harvest(args.dirs)
    if not rows:
        sys.exit("no RESULT lines found")
    if args.csv:
        cols = sorted({k for r in rows for k in r})
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} rows -> {args.csv}", file=sys.stderr)

    keys = [k.strip() for k in args.group.split(",")]
    groups = group_rows(rows, keys, args.metric)
    width = max(len(" ".join(k)) for k in groups)
    print(f"{'condition':{width}s}  {args.metric} (mean+-std, n)")
    for key in sorted(groups):
        v = groups[key]
        mean = statistics.mean(v)
        std = statistics.pstdev(v) if len(v) > 1 else 0.0
        flag = "" if len(v) >= args.min_seeds else f"  <-- n={len(v)}"
        print(f"{' '.join(key):{width}s}  {mean:.4f}+-{std:.4f} (n={len(v)}){flag}")

    if args.coverage:
        short = [g for g, v in sorted(groups.items()) if len(v) < args.min_seeds]
        print(f"\ncoverage: {len(groups)} conditions, {len(short)} below {args.min_seeds} seeds")
        for g in short:
            print(f"  MISSING seeds: {' '.join(g)} (n={len(groups[g])})")


if __name__ == "__main__":
    main()
