"""Encoder edge budget at each gnn.encoder_edge_drop, and the encoder-vs-basis
edge deficit it produces (comparable to the sharding deficit 1 - intra_frac)."""
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import torch, yaml
from src import *
from config.config import overlay_config
from registries import datasets
import src.datasets  # noqa

with open("config/uci_gru.yaml") as f:
    overlay_config(config, yaml.safe_load(f))
snaps = datasets["uci"](config)
n = len(snaps)
E = [s.edge_index.shape[1] for s in snaps]
print(f"uci: {n} snapshots, |E_t| min={min(E)} median={sorted(E)[n//2]} max={max(E)} total={sum(E)}")

# cumulative undirected union through t, as _spectral_step builds it
cum = None
cum_sizes = []
cur_und = []
for s in snaps:
    e = s.edge_index.cpu()
    e = torch.cat([e, e.flip(0)], dim=1)
    cur_und.append(torch.unique(e, dim=1).shape[1])
    cum = e if cum is None else torch.cat([cum, e], dim=1)
    cum = torch.unique(cum, dim=1)
    cum_sizes.append(cum.shape[1])

import statistics as st
ratio = [c / u for c, u in zip(cur_und, cum_sizes)]
print(f"mean_t |E_t|/|cum_t| (undirected) = {st.fmean(ratio):.4f}  -> deficit at p=0, C=1: "
      f"{1 - st.fmean(ratio):.4f}")

print("\n p     kept_frac(actual)  floor_binds  mean_enc_edges  deficit 1-(kept*0.138)")
for p in (0.0, 0.5, 0.75, 0.9, 0.95, 0.99):
    keeps, binds = [], 0
    for e in E:
        k = min(e, max(2, int(round(e * (1.0 - p)))))
        if k != min(e, int(round(e * (1.0 - p)))):
            binds += 1
        keeps.append(k)
    kf = sum(keeps) / sum(E)
    print(f"{p:5} {kf:17.4f} {binds:12d} {st.fmean(keeps):15.1f} "
          f"{1 - kf * st.fmean(ratio):22.4f}")
print("\nreference: C9 intra-edge fraction measured 0.106 (t=3) / 0.127 (t=12) -> "
      "C9 deficit 1 - 0.106*0.138/0.138... see results.md; established C9 deficit 0.985")
