"""How much of the mrr_filter delta is just the second draw?

results.md §12's Tier-2 number compares the `split` arm against the `snapshot`
arm of one `mrr_filter=both` run. Those arms share a model and a snapshot but
draw their negatives INDEPENDENTLY, so their difference carries sampling
variance of unmeasured size. The measured effect was -0.0029 +/- 0.0022, which
is only meaningful against that floor.

This measures the floor directly: it neutralises `_extra_forbidden` so BOTH arms
use the split filter, leaving the second draw as the only difference between
them. Any delta reported here is pure draw noise. If it is the same size as
§12's effect, §12's effect is not separable from sampling.

usage: python analysis/probes/mrr_draw_noise.py [dataset] [repeats]
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

_a = sys.argv[1:]
DATASET = _a[0] if _a else "uci"
REPEATS = _a[1] if len(_a) > 1 else "3"

import src.metrics.mrr as M

# Both arms now resolve to the split filter; the `both` code path still runs, so
# the second arm is a second independent draw under identical forbidden edges.
M._extra_forbidden = lambda snap, mode: None

sys.argv = [
    "main.py", "-c", f"config/{DATASET}_gru.yaml", "--repeat", REPEATS, "--set",
    "model.data_type=feature", "subgraph.num_subgraphs=1", "wandb.mode=disabled",
    "metric.mrr_filter=both",
]

import main

print(f"### draw-noise control: {DATASET}, both arms on the split filter, "
      f"{REPEATS} seeds\n### every reported mrr_delta below is sampling noise only\n")
main.main()
