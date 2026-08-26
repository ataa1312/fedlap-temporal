"""Do the random hop2vec features actually get trained toward reducing the loss?

FedLap's paper says the structural feature vectors (SFV, called hop2vec here) are
random at init and learned by gradient descent. This probe checks that claim on
OUR dynamic path, where W is NOT what upstream builds: with
num_spectral_features=spectral_len, add_structural_features returns a
(spectral_len, num_structural_features) matrix, so W is indexed by SPECTRAL SLOT,
not by node. The embedding is S = MLP(relu(Q_t @ W)).

Three things get measured, at every local training step:

  in_optimizer   is W actually in a param group (not just requires_grad)
  |grad W|       gradient norm reaching W
  rel movement   |W - W_0|_F / |W_0|_F, cumulative
  dL_W           THE test: take the optimizer's REAL step, then roll every
                 parameter back except W, and recompute the loss on the SAME
                 batch. dL_W < 0 means W alone moved downhill -- i.e. W is
                 being trained toward reducing the loss, not just drifting.
  dL_rest        the complement: everything except W moved. Lets us compare how
                 much of the step's progress W is responsible for.

dL_W is the decisive one. Movement alone proves nothing (a parameter can drift
while contributing nothing); a systematically negative dL_W is what "trained"
means.

It has to be the optimizer's own step, not a hand-rolled -lr*grad one: the
optimizer is Adam at lr=1e-2, whose step is ~lr*sign(g) elementwise, while
|grad W| ~ 0.08 against |W| ~ 19.6. A raw SGD probe step is ~4e-5 of |W| and
measures float noise rather than the update the run actually applies.

usage: python analysis/probes/hop2vec_training.py [dataset] [n_snapshots]
"""
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import torch

_a = sys.argv[1:]
DATASET = _a[0] if _a else "uci"
N_SNAPS = int(_a[1]) if len(_a) > 1 else 4
NUM_C = int(_a[2]) if len(_a) > 2 else 1
SEED = int(_a[3]) if len(_a) > 3 else None

sys.argv = ["hop2vec_training", "-c", f"config/{DATASET}_gru.yaml", "--set",
            "model.data_type=f+s", f"subgraph.num_subgraphs={NUM_C}",
            "spectral.update_mode=update", "wandb.mode=disabled",
            "train.auto_resume=false", "train.ckpt_period=0"] \
    + ([f"seed={SEED}"] if SEED is not None else [])
from parser import Parser

p = Parser()
cfg = p.load_config(p.parse_args())
import src

src.config = cfg
from registries import datasets
import src.datasets  # noqa: F401
from src.utils.graph_partitioning import partition_snapshots
from src.dynamic_server import DynamicServer
import src.train.federated_orchestrator as fo

torch.manual_seed(cfg["seed"])
np.random.seed(cfg["seed"])

REC = {"steps": 0, "in_opt": 0, "gnorm": [], "rel": [], "dW_rel": [],
       "dL_W": [], "dL_rest": [], "dL_all": [], "W0": None, "shape": None}


def _sfv(model):
    sm = getattr(model, "smodel", None)
    if sm is None or not hasattr(sm, "get_SFV"):
        return None
    return sm.get_SFV()


_orig = fo._step_train_pair


def _patched(model, snap_today, snap_tomorrow, hs, loss_fn, optimizer, device,
             is_recurrent, pos_split="train", prepared_snap=None):
    model.train()
    if prepared_snap is None:
        pos = fo._pos_for_split(snap_tomorrow, pos_split).to(device)
        prepared_snap = fo._attach_future_link_pred_labels(
            snap_today.to(device), snap_tomorrow.to(device), pos)

    W = _sfv(model)
    if W is None or not W.requires_grad:
        return _orig(model, snap_today, snap_tomorrow, hs, loss_fn, optimizer,
                     device, is_recurrent, pos_split, prepared_snap)

    if REC["W0"] is None:
        REC["W0"] = W.detach().clone()
        REC["shape"] = tuple(W.shape)

    # is W in a param group, and at what lr?
    lr = None
    for g in optimizer.param_groups:
        for prm in g["params"]:
            if prm is W:
                lr = g["lr"]
    REC["steps"] += 1
    if lr is not None:
        REC["in_opt"] += 1

    pred, label, new_hs = fo._model_forward(model, prepared_snap, hs, is_recurrent)
    loss = loss_fn(pred, label.float())
    reg = cfg["spectral"]["regularizer_coef"]
    if reg:
        loss = loss + reg * model.intrinsic_regularizer()
    L0 = loss.item()
    optimizer.zero_grad()
    loss.backward()

    g = None if W.grad is None else W.grad.detach().clone()
    REC["gnorm"].append(0.0 if g is None else float(g.norm()))

    params = [prm for grp in optimizer.param_groups for prm in grp["params"]]

    def _relit():
        with torch.no_grad():
            pr, lb, _ = fo._model_forward(model, prepared_snap, hs, is_recurrent)
            out = loss_fn(pr, lb.float())
            if reg:
                out = out + reg * model.intrinsic_regularizer()
        return float(out)

    def _restore(vals):
        with torch.no_grad():
            for prm, v in zip(params, vals):
                prm.copy_(v)

    # --- the test: take the REAL step, then roll back all but W ---
    pre = [prm.detach().clone() for prm in params]
    w_idx = next(i for i, prm in enumerate(params) if prm is W)  # `is`, not ==
    optimizer.step()
    post = [prm.detach().clone() for prm in params]

    _restore([p_ if prm is not W else q_ for prm, p_, q_ in zip(params, pre, post)])
    REC["dL_W"].append(_relit() - L0)
    _restore([q_ if prm is not W else p_ for prm, p_, q_ in zip(params, pre, post)])
    REC["dL_rest"].append(_relit() - L0)
    _restore(post)

    with torch.no_grad():
        REC["dL_all"].append(_relit() - L0)
        REC["rel"].append(float((W - REC["W0"]).norm() / REC["W0"].norm()))
        REC["dW_rel"].append(
            float((W - pre[w_idx]).norm() / W.norm()))

    return L0, new_hs


fo._step_train_pair = _patched
# dynamic_client imports the symbol directly, so rebind there too
import src.dynamic_client as dc

if hasattr(dc, "_step_train_pair"):
    dc._step_train_pair = _patched

snaps = datasets[DATASET](cfg)[: N_SNAPS + 1]
print(f"dataset={DATASET} snapshots={len(snaps)} (capped at {N_SNAPS}+1)")
client_snaps = partition_snapshots(snaps, 1)
server = DynamicServer(snaps)
for s in client_snaps:
    server.add_client(s)
server.joint_train_w(FL=True)


def _stat(name, v):
    if not v:
        print(f"  {name:14s} (no samples)")
        return
    a = np.asarray(v, dtype=float)
    print(f"  {name:14s} mean={a.mean():+.6f}  median={np.median(a):+.6f}  "
          f"min={a.min():+.6f}  max={a.max():+.6f}  n={len(a)}")


print()
print("=" * 72)
print(f"hop2vec SFV: shape={REC['shape']}  "
      f"(spectral_len x num_structural_features -> indexed by SPECTRAL SLOT)")
print(f"steps={REC['steps']}  in_optimizer={REC['in_opt']}/{REC['steps']}")
_stat("|grad W|", REC["gnorm"])
_stat("rel movement", REC["rel"])
_stat("|dW|/|W| step", REC["dW_rel"])
_stat("dL_W", REC["dL_W"])
_stat("dL_rest", REC["dL_rest"])
_stat("dL_all", REC["dL_all"])
for nm in ("dL_W", "dL_rest", "dL_all"):
    if REC[nm]:
        a = np.asarray(REC[nm])
        print(f"\n  {nm} < 0 in {(a < 0).sum()}/{len(a)} steps "
              f"({100.0 * (a < 0).mean():.1f}%)")
if REC["gnorm"]:
    z = (np.asarray(REC["gnorm"]) == 0).sum()
    print(f"  zero-gradient steps: {z}/{len(REC['gnorm'])}")
print("=" * 72)
