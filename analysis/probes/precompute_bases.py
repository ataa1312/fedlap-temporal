"""Fill the exact-eigenbasis cache that proto_fusion.py reads (PROTO_BASIS_CACHE).

The eigensolve is the bottleneck on large graphs (reddit_body: ~40 s per snapshot
at mid-run, ~120 s at the end — ~3 h for one pass) and every C-job would
otherwise redo it. Snapshots are independent, so N hosts can fill the same NAS
directory concurrently: give each host a different `offset` with `stride` = the
number of hosts. Stride (not contiguous ranges) matters — cost grows with t, so
contiguous blocks would leave the last host with all the expensive snapshots.

Writes are atomic (tmp + rename), so concurrent fillers and readers are safe.
Only the unthinned basis (thin=1.0) is cached; the `thin` sweep is a local
uci-sized experiment.

usage: PROTO_BASIS_CACHE=<dir> python analysis/probes/precompute_bases.py \\
           <dataset> <config> <K> <offset> <stride>
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import torch

DATASET = sys.argv[1]
CONFIG = sys.argv[2]
K_PE = int(sys.argv[3])
OFFSET = int(sys.argv[4])
STRIDE = int(sys.argv[5])

sys.argv = ["precompute", "-c", CONFIG, "--set", "model.data_type=feature",
            "subgraph.num_subgraphs=1", "wandb.mode=disabled"]
from parser import Parser

p = Parser()
cfg = p.load_config(p.parse_args())
import src

src.config = cfg
from registries import datasets
import src.datasets  # noqa: F401
from src.utils.graph import Graph

CACHE = os.environ.get("PROTO_BASIS_CACHE")
assert CACHE, "set PROTO_BASIS_CACHE to the shared cache directory"
Path(CACHE).mkdir(parents=True, exist_ok=True)


def und(ei):
    e = ei.cpu().numpy()
    return {(min(a, b), max(a, b)) for a, b in zip(e[0], e[1]) if a != b}


snaps = datasets[DATASET](cfg)
N, T = snaps[0].num_nodes, len(snaps)
print(f"{DATASET}: N={N} T={T}; filling offset={OFFSET} stride={STRIDE} K={K_PE}", flush=True)

cum = set()
done = 0
t0 = time.time()
for t in range(T - 1):
    cum |= und(snaps[t].edge_index)
    if t % STRIDE != OFFSET:
        continue
    path = Path(CACHE) / f"{DATASET}_K{K_PE}_thin1.0_t{t}.npy"
    if path.exists():
        continue
    ts = time.time()
    a = np.array([x for x, _ in cum], dtype=np.int64)
    b = np.array([y for _, y in cum], dtype=np.int64)
    e = torch.tensor(np.stack([np.r_[a, b], np.r_[b, a]]), dtype=torch.long)
    g = Graph(x=torch.ones(N, 1), edge_index=e, node_ids=torch.arange(N))
    _, U = g.calc_eigs_exact_sym(K_PE)
    Q = U.numpy().astype(np.float32)
    Qn = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)
    tmp = path.with_suffix(f".{os.getpid()}.tmp.npy")  # PID-unique: hosts race on the same t
    np.save(tmp, Qn)
    os.replace(tmp, path)
    done += 1
    print(f"t={t} |E|={len(cum)} {time.time() - ts:.1f}s (done {done})", flush=True)
print(f"offset {OFFSET}: {done} bases in {time.time() - t0:.0f}s", flush=True)
