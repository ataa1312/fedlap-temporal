import os
import math
import random
import statistics
from dataclasses import dataclass

from src import *
from src.server import Server
from config.registry import Registry
from src.utils.graph import Graph
from src.dynamic_client import DynamicClient
from src.GNN.dynamic_classifier import DynamicClassifier
from src.GNN.fed_dynamic_classifier import (
    FedDynamicEdgeScoreClassifier,
    FedDynamicPEClassifier,
    make_fed_dynamic_classifier,
)
from src.metrics.mrr import compute_mrr_from_z, compute_mrr_splits_from_z, compute_hard_auc_ap_from_z
from src.metrics.classification import binary_classification_metrics
from src.train.federated_orchestrator import (
    _partition_edges_per_snapshot,
    _precompute_keep_ratio,
    _precompute_encoder_edge_drop,
    _pos_for_split,
    _attach_future_link_pred_labels,
    _step_eval_with_mrr_pair,
    _stitch_global_z,
    _clone_state,
    _mrr_filter_mode,
)
from registries import losses


@dataclass
class SpectralFeatures:
    U: torch.Tensor
    D: torch.Tensor
    Q: torch.Tensor | None = None


def _to_cpu_sf(U, D, Q) -> SpectralFeatures:
    """Detach + move spectral tensors to CPU for the first/prev caches so the big
    n x spectral_len eigenvectors never accumulate on the GPU across snapshots."""
    return SpectralFeatures(
        U=U.detach().cpu() if U is not None else None,
        D=D.detach().cpu() if D is not None else None,
        Q=Q.detach().cpu() if Q is not None else None,
    )


def _cheb_cutoff(prev_D, safety=0.9):
    """Low-pass cutoff for the filtered solver: the previous snapshot's
    lambda_k, pulled down by `safety`. It must sit AT or just below lambda_k —
    a higher cutoff admits far more than k modes on these dense spectra and the
    block can no longer span them (results.md §10.12). None on the first
    snapshot, where the solver falls back to its own default."""
    if prev_D is None:
        return None
    d = prev_D[prev_D > 0] if hasattr(prev_D, "__getitem__") else None
    if d is None or len(d) == 0:
        return None
    return safety * float(d.max())


def _repeat_new_split() -> bool:
    """metric.repeat_new_split, tolerant of configs predating the key."""
    m = config["metric"]
    return bool(m["repeat_new_split"]) if "repeat_new_split" in m else False


def _get_sfv(clf):
    """The learnable SFV W — a leaf param that is NOT in the state_dict when
    federated.sfv_share='local', so checkpoints capture it here.

    Goes through the smodel's get_SFV/set_SFV protocol rather than reaching into
    ``smodel.graph``: only DynamicSLaplace owns an SFV, while SignNet and the
    f+es Invariant smodel own none (their learnables are already in state_dict)
    and return None."""
    smodel = getattr(clf, "smodel", None)
    w = smodel.get_SFV() if smodel is not None else None
    return None if w is None else w.detach().cpu()


def _set_sfv(clf, w):
    smodel = getattr(clf, "smodel", None)
    if smodel is not None and w is not None:
        smodel.set_SFV(w)


def _procrustes_on(data_type=None) -> bool:
    """Effective spectral.use_procrustes for the current data type.

    'auto' means on wherever the readout reads U's coordinates (f+s, f+pe,
    structure) and off on f+es, whose features are rotation-invariant -- the
    rotation cannot help there and it breaks the phi block's identity, since
    procrustes_project rotates U but not D. An explicit bool wins everywhere.
    """
    v = config["spectral"]["use_procrustes"]
    if isinstance(v, bool):
        return v
    dt = config["model"]["data_type"] if data_type is None else data_type
    return dt != "f+es"


def _sfv_flag(key: str) -> bool:
    """structure_model.{sfv_reset_per_snapshot,freeze_sfv}, tolerant of configs
    predating the keys."""
    sm = config["structure_model"]
    return bool(sm[key]) if key in sm else False



# Config keys EXCLUDED from the identity fingerprint: they cannot change a number.
# Everything else is included, so a knob added later is covered without anyone
# remembering to give it a token -- which is the failure mode that let mrr_filter,
# base_lr, iterations, dataset.split and spectral_len (on f+pe/f+es) collide.
_FP_EXCLUDE_TOP = {
    "seed",            # already its own token
    "outdir", "device", "num_workers", "num_threads", "print", "remark",
    "experiment", "num_runs", "metric_best", "downstream_task",
    "wandb",           # reporting only
}
# Every exclusion is a hole in the backstop, so each one must be justified as
# "no live code can read it", not "it looks like reporting". Verified by
# grep: none of the keys above is read anywhere outside config/. Two below are
# read, and are excluded as a DELIBERATE trade rather than as no-ops:
#   dataset.path  -- the same dataset lives at different roots on the laptop and
#                    on the NAS home, and including it would stop a cluster run
#                    resuming a checkpoint written beside it. It CAN move a
#                    number (point it at other data and you get other numbers).
#   train.*       -- checkpoint plumbing. auto_resume MUST be excluded: it is the
#                    switch that turns resuming on, so including it would change
#                    the identity of the very run you are trying to resume.
# node2vec is NOT excluded, despite reading as a dormant upstream section:
# structure_model.structure_type='node2vec' reaches find_node2vect_embedings,
# which reads walk_length/context_size/walks_per_node/p/q/num_negative_samples/
# batch_size at call time. assert_cfg does not restrict structure_type, so the
# path is one --set away and every one of those knobs moves the numbers.
_FP_EXCLUDE_PATH = {
    ("train", "ckpt_dir"), ("train", "auto_resume"), ("train", "ckpt_period"),
    ("train", "ckpt_clean"),
    ("metric", "snapshot_record_dir"),
    ("dataset", "path"),
}


def _config_fingerprint() -> str:
    """Short hash over every config value that can move a number.

    The explicit tokens in _run_id stay -- they are readable and carry the arm's
    meaning. This is the BACKSTOP: it makes the identity complete, so two runs
    differing in any knob at all resolve to different checkpoints AND different
    per-snapshot record files. Before it, a run differing only in mrr_filter,
    hard_neg, base_lr, iterations, dataset.split, rank_eval_multiplier, or
    spectral_len on f+pe/f+es was INDISTINGUISHABLE from its neighbour: a
    wrong-arm checkpoint resume, and record rows that silently overwrote each
    other last-write-wins.

    Deliberately over-separates rather than under-separates. Adding a new config
    key changes the hash for every arm, which orphans old checkpoints -- that is
    the safe direction, and correctness of identity was chosen over preserving
    banked ids.
    """
    import hashlib

    def encode(v):
        # NOT bare repr(): repr of a set or a dict depends on iteration order,
        # which for strings is PYTHONHASHSEED dependent / insertion dependent, and
        # repr of anything without a __repr__ carries a memory address. Any of the
        # three makes the hash differ between processes, so a run could not find
        # its own checkpoint. List order stays significant -- dataset.split
        # [0.8,0.1,0.1] and [0.1,0.8,0.1] are different runs. Anything else raises
        # rather than being hashed unstably: a loud failure at startup beats an
        # identity that silently drifts.
        if isinstance(v, (str, int, float, bool, type(None))):
            return repr(v)
        if isinstance(v, (list, tuple)):
            return "[" + ",".join(encode(x) for x in v) + "]"
        if isinstance(v, (set, frozenset)):
            return "{" + ",".join(sorted(encode(x) for x in v)) + "}"
        if isinstance(v, dict):
            return "{" + ",".join(f"{k!r}:{encode(v[k])}" for k in sorted(v)) + "}"
        raise TypeError(
            f"config value of type {type(v).__name__!r} has no stable encoding for "
            "the run fingerprint; add one rather than hashing an unstable repr"
        )

    def walk(node, path=()):
        # Registry supports __iter__/__getitem__ but NOT .keys(), so a
        # hasattr(node, "keys") test never recurses and silently hashes repr() of
        # each whole section -- which separates everything by accident and makes
        # every exclusion above dead code.
        if isinstance(node, Registry):
            for k in sorted(iter(node)):
                if not path and k in _FP_EXCLUDE_TOP:
                    continue
                if (path + (k,)) in _FP_EXCLUDE_PATH:
                    continue
                yield from walk(node[k], path + (k,))
        else:
            yield ".".join(path) + "=" + encode(node)

    blob = "\n".join(sorted(walk(config)))
    return hashlib.sha1(blob.encode()).hexdigest()[:8]


def _weighted_mean_metrics(
    metrics_list: list[dict], weights: list[float]
) -> dict[str, float]:
    """Per-key weighted mean over a list of metric dicts, skipping nan
    contributions per key (a key with all-nan or zero weight -> nan)."""
    if not metrics_list:
        return {}
    out = {}
    # Union of keys, not metrics_list[0]'s: a resumed run can restore a
    # metrics_history whose dicts carry the repeat/new keys and then append dicts
    # that do not (or the reverse), which raised KeyError at the very END of the
    # run -- after all the compute and before the done-checkpoint. Absent keys
    # contribute nothing rather than exploding.
    # dict.fromkeys, not a set: set iteration over strings is PYTHONHASHSEED
    # dependent, so a set here makes mean_metrics' key order (and the bytes of the
    # done-checkpoint that stores it) differ between processes. First-seen order
    # also keeps the fully-populated case byte-identical to metrics_list[0]'s.
    keys = dict.fromkeys(k for md in metrics_list for k in md)
    for k in keys:
        num, den = 0.0, 0.0
        for md, w in zip(metrics_list, weights):
            v = md.get(k, float("nan"))
            if v == v:  # skip nan
                num += v * w
                den += w
        out[k] = num / den if den > 0 else float("nan")
    return out


class DynamicServer(Server):
    """Federated ROLAND live-update over client subgraphs, reusing FedLap's Server
    FedAvg primitives (share_weights / get_weights / sum_lod / node-count coef).

    Mirrors the GNNServer surface: add_client / initialize / initialize_FL /
    joint_train_w. Per global snapshot pair (t, t+1): reported eval, then `epochs`
    FedAvg rounds of `model.local_epochs` local ROLAND steps, weight-averaged via
    sum_lod with round-level early stopping on the aggregated client val loss,
    then a per-client hidden-state refresh. Inactive clients (a subgraph too small
    to form a train/val split) abstain from the aggregation. FL=False = local-only
    baseline: no broadcast, no aggregation; the reported eval becomes the
    node-count-weighted mean of per-client local MRRs.
    """

    def __init__(self, global_snaps):
        gsnap0 = global_snaps[0]
        global_graph = Graph(
            x=gsnap0.x,
            edge_index=gsnap0.edge_index,
            edge_attr=gsnap0.edge_attr,
            node_ids=torch.arange(gsnap0.num_nodes, device=device),
        )
        super().__init__(graph=global_graph)
        self.global_snaps = global_snaps
        self.clients = list[DynamicClient]()
        # ROLAND spectral tracking needs only the FIRST snapshot (keep basis +
        # procrustes anchor) and the PREVIOUS one (update tracking), held on CPU.
        # (The old growing per-snapshot dict was a DySAT-era artifact and OOMed.)
        self._first_spectral: SpectralFeatures | None = None
        self._prev_spectral: SpectralFeatures | None = None
        self._cum_edges = None  # undirected union of global edges up to the current t
        # (pair_key, snapshot) appearances backing the age kernel; rebuilt on resume
        self._cum_events_key = None
        self._cum_events_t = None
        self._basis_coverage = None  # served-basis coverage counts for the current t
        self._basis_zero_rows = None  # zero-row mask of the basis SERVED at the current t

    def add_client(self, snaps):
        client = DynamicClient(snaps, id=self.num_clients)
        self.clients.append(client)
        self.num_clients += 1

    def initialize(
        self,
        smodel_type=None,
        fmodel_type=None,
        data_type=None,
        **kwargs,
    ):
        data_type = config["model"]["data_type"] if data_type is None else data_type
        smodel_type = config["model"]["smodel_type"] if smodel_type is None else smodel_type
        share = {}
        if data_type == "feature":
            self.classifier = DynamicClassifier(self.graph)  # global model (decode in eval)
        elif data_type == "f+pe":
            # input-side exact-eigenvector PE; nothing spectral to share (no SFV)
            self.classifier = FedDynamicPEClassifier(self.graph)
        elif data_type == "f+es":
            # decision-level spectral term over rotation-invariant features
            self.classifier = FedDynamicEdgeScoreClassifier(self.graph)
        elif data_type == "f+s":
            # learnable W created ONCE on the server and shared so every owner
            # starts from the same init (GNNServer.initialize -> share["SFV"];
            # hop2vec -> random (spectral_len, num_structural_features) leaf)
            SFV = kwargs.get("SFV")
            if SFV is None:
                self.graph.add_structural_features(
                    structure_type=config["structure_model"]["structure_type"],
                    num_structural_features=config["structure_model"]["num_structural_features"],
                    num_spectral_features=config["spectral"]["spectral_len"],
                )
                SFV = self.graph.structural_features
            if _sfv_flag("freeze_sfv"):
                # One place covers every owner: make_sgraph re-wraps with
                # requires_grad=SFV.requires_grad, and SFVMixin.parameters()
                # appends graph.x only when it requires grad -- so clearing it
                # here keeps W out of every optimizer and out of get_grads.
                SFV = SFV.detach().requires_grad_(False)
            share["SFV"] = SFV
            self.classifier = make_fed_dynamic_classifier(smodel_type, self.graph, SFV)
        else:
            raise NotImplementedError(
                f"data_type={data_type!r}: structure-only needs an smodel-only subclass (deferred)"
            )
        self.classifier.eval()
        return share

    def initialize_FL(
        self,
        smodel_type=None,
        fmodel_type=None,
        data_type=None,
        **kwargs,
    ) -> None:
        share = self.initialize(
            smodel_type=smodel_type,
            fmodel_type=fmodel_type,
            data_type=data_type,
            **kwargs,
        )
        kwargs.update(share)
        for client in self.clients:
            client.initialize(
                smodel_type=smodel_type,
                fmodel_type=fmodel_type,
                data_type=data_type,
                **kwargs,
            )
        self.share_weights()  # sync all clients to the server init

    def _coef(self, clients):
        total = sum(c.num_nodes() for c in clients)
        return [c.num_nodes() / total for c in clients]

    def joint_train_g(self, *args, **kwargs):
        raise NotImplementedError(
            "gradient averaging is unused in the dynamic path; use joint_train_w"
        )

    def joint_train_w(
        self,
        epochs=None,
        smodel_type=None,
        fmodel_type=None,
        FL=True,
        data_type=None,
        log=True,
        plot=False,
        model_type="",
        log_cb=None,
        **kwargs,
    ):
        # epochs = FedAvg rounds per snapshot; None -> config at call time (the
        # YAML overlay lands after import, so no def-time default binding).
        self.initialize_FL(
            smodel_type=smodel_type,
            fmodel_type=fmodel_type,
            data_type=data_type,
            **kwargs,
        )
        dt = config["model"]["data_type"] if data_type is None else data_type
        smt = config["model"]["smodel_type"] if smodel_type is None else smodel_type
        use_spectral = dt in ("structure", "f+s", "f+pe", "f+es")
        self._first_spectral = self._prev_spectral = None  # runs are independent (fresh W, fresh Bernoulli draws)
        self._cum_edges = None
        self._cum_events_key = self._cum_events_t = None
        rounds = config["model"]["iterations"] if epochs is None else epochs
        local_epochs = config["model"]["local_epochs"]
        tol = config["train"]["internal_validation_tolerance"]
        embed = config["gnn"]["embed_update_method"]
        split = config["dataset"]["split"]
        ds = config["dataset"]
        seed = (
            ds["split_seed"]
            if ("split_seed" in ds and ds["split_seed"] is not None)
            else config["seed"]
        )
        loss_fn = losses[config["model"]["loss_fun"]]
        mrr_k = config["experimental"]["rank_eval_multiplier"]
        mrr_method = config["metric"]["mrr_method"]
        is_meta = config["meta"]["is_meta"]
        meta_alpha = config["meta"]["alpha"]
        meta_method = config["meta"]["method"]
        # Which test set the reported metrics are scored on. 'auto' keeps the historical
        # coupling (global when FL, per-client otherwise); 'local' scores both FL settings
        # on each client's own subgraph, so fl=true vs fl=false differ only in TRAINING.
        scope = config["metric"]["eval_scope"]
        global_eval = scope == "global" or (scope == "auto" and FL)
        if log:
            tag = model_type or ("FL WA" if FL else "Local WA")
            LOGGER.info(f"{tag} live-update starts! (eval_scope={scope} -> "
                        f"{'global' if global_eval else 'per-client'})")

        # Global edge split -> pos_test for the reported eval; per-client splits ->
        # pos_train/pos_val for local fine-tune (seed offset per client).
        _partition_edges_per_snapshot(self.global_snaps, split, seed)
        for c, cl in enumerate(self.clients):
            _partition_edges_per_snapshot(cl.snaps, split, seed + 1000 * (c + 1))
        if embed in {"moving_average", "masked_gru"}:
            km = config["gnn"]["keep_ratio_mode"]
            _precompute_keep_ratio(self.global_snaps, km)
            for cl in self.clients:
                _precompute_keep_ratio(cl.snaps, km)

        # Encoder-only edge starvation (gnn.encoder_edge_drop). Attached to the
        # CLIENT snapshots alone, because those are the only objects ever encoded:
        # the server classifier decodes the stitched z and never message-passes,
        # and self.global_snaps feeds the eval targets, the negatives and the
        # cumulative union of _spectral_step -- all of which must stay complete.
        # Placed after _precompute_keep_ratio so keep_ratio degrees are still
        # counted over the FULL snapshot (ROLAND's edge_train_mode='all').
        edrop = config["gnn"]["encoder_edge_drop"] if "encoder_edge_drop" in config["gnn"] else 0.0
        for c, cl in enumerate(self.clients):
            _precompute_encoder_edge_drop(cl.snaps, float(edrop), seed + 500000 * (c + 1))

        n_tasks = len(self.global_snaps) - 1
        if n_tasks < 1:
            raise ValueError(
                f"federated live_update needs >= 2 snapshots; got {len(self.global_snaps)}"
            )

        # Checkpointing (gated on train.auto_resume): dump resumable state every
        # ckpt_period snapshots; on restart skip already-done runs or resume mid-run.
        ckpt_every = config["train"]["ckpt_period"] if config["train"]["auto_resume"] else 0

        mrr_history: list[float] = []
        t_history: list[int] = []  # TRUE snapshot index per entry; see the append site
        metrics_history: list[dict] = []
        w_init = None  # running meta W_init (moving-avg across snapshots; FL path)
        t_start = 0
        if ckpt_every > 0:
            done = self._load_done_ckpt()
            if done is not None:
                if log:
                    LOGGER.info(f"{self._run_id()}: already complete, skipping (checkpoint)")
                done["_resumed_complete"] = True
                return done
            resumed = self._load_partial_ckpt()
            if resumed is not None:
                t_start, w_init, mrr_history, metrics_history, t_history = resumed
                # Checkpoints predating t_history carry no axis for the pre-crash
                # segment. -1 marks it as unknown; the writer refuses such a record
                # rather than emitting rows that all share one index -- the reader
                # keys on (arm, seed, t), so identical indices collapse to a single
                # row and silently move every aggregate.
                if t_history is None:
                    t_history = [-1] * len(mrr_history)
                if log:
                    LOGGER.info(f"{self._run_id()}: resuming from t={t_start}")
        for t in range(t_start, n_tasks):
            # 0. Spectral provider: U_t on the cumulative global graph, sliced to
            #    every owner via set_QD (before eval — encoding snap_t needs Q_t).
            if use_spectral:
                self._spectral_step(t, smt)
            elif _repeat_new_split():
                # The split needs the cumulative union, which _spectral_step would
                # otherwise be the only thing maintaining. Keep it for backbone-only
                # runs too, so per-subset gains can be measured against a baseline.
                self._accumulate_cum_edges(t)

            # 1. Reported eval. FL: clients hold the global weights (share_weights
            #    at initialize_FL / end of the previous snapshot) -> global stitch.
            #    Local-only: weighted mean of per-client local MRRs.
            if global_eval:
                mrr, metrics = self._eval_mrr(t, mrr_k, mrr_method)
            else:
                mrr, metrics = self._eval_mrr_local(t, loss_fn, mrr_k, mrr_method)
            if mrr is not None and not math.isnan(mrr):
                # TRUE snapshot index. mrr_history is appended ONLY when the eval
                # returns a finite MRR (an edgeless snapshot yields no test positives
                # and _eval_mrr returns None), so position in the list is NOT t. Any
                # consumer that relabels survivors 0..n-1 silently compresses the axis:
                # the early/late split lands on the wrong snapshots, and a cross-seed
                # or cross-arm join pairs DIFFERENT snapshots the moment two runs skip
                # different ones.
                t_history.append(t)
                mrr_history.append(mrr)
                if metrics is not None:
                    metrics_history.append(metrics)
                if log_cb is not None:  # live per-snapshot wandb (resumable run)
                    log_cb(t, mrr, metrics)
                if log:
                    m = metrics or {}
                    # The repeat/new numbers are per-snapshot MEANS over sources, and
                    # the subsets can be tiny (as733 carries a median of a few new
                    # positives per snapshot). Without the per-snapshot value AND its
                    # sample count, a run-level mrr_new cannot be re-weighted, bootstrapped,
                    # or checked for concentration in a handful of snapshots -- which is
                    # exactly the audit that results.md §20.6 records as impossible.
                    split = ""
                    if "mrr_repeat" in m:
                        split = (
                            f" mrr_repeat={m.get('mrr_repeat', float('nan')):.4f}"
                            f" mrr_new={m.get('mrr_new', float('nan')):.4f}"
                            f" n_repeat={m.get('n_repeat', -1)} n_new={m.get('n_new', -1)}"
                            f" src_repeat={m.get('src_repeat', -1)} src_new={m.get('src_new', -1)}"
                        )
                    # Basis coverage, per snapshot. The pair fraction is the one that
                    # attributes a subset effect: pairs touching an all-zero row get a
                    # constant edge-score term and fall back to backbone-only ranking.
                    cov = ""
                    if "basis_covered" in m:
                        cov = (
                            f" cov={m['basis_covered']}/{m['basis_total']}"
                            f" zeroed={m['basis_zeroed']}"
                            f" zpair={m.get('basis_zeroed_pair_frac', float('nan')):.4f}"
                        )
                    LOGGER.info(
                        f"t={t} mrr={mrr:.4f} auc={m.get('roc_auc', float('nan')):.4f} "
                        f"ap={m.get('ap', float('nan')):.4f} f1={m.get('f1', float('nan')):.4f} "
                        f"mcc={m.get('mcc', float('nan')):.4f}{split}{cov}"
                    )

            # 1b. Optional SFV reset -- AFTER the reported eval, BEFORE training on
            #     this snapshot. Resetting before the eval would measure a random W
            #     rather than a W trained without accumulation, which is a different
            #     (and useless) arm. Placed here, snapshot t is trained from a fresh
            #     draw and the eval at t+1 sees a W fitted to t alone.
            if _sfv_flag("sfv_reset_per_snapshot"):
                self._reset_sfvs()

            # 2. Training. FL: `rounds` communication rounds of `local_epochs`
            #    local steps, weight-averaged via sum_lod, with round-level early
            #    stopping on the aggregated (node-count-weighted) client val loss.
            #    Local-only: the same step budget without broadcast/aggregation.
            if FL:
                # meta warm-start: reload the running W_init before the rounds;
                # the round loop's share_weights broadcasts it to the clients.
                if is_meta and w_init is not None:
                    self.load_state_dict(_clone_state(w_init))
                best = {"val": float("inf"), "state": None}
                stale = 0
                for _ in range(rounds):
                    self.share_weights()  # broadcast global -> clients
                    parts = [cl for cl in self.clients if cl.can_train(t)]  # abstention
                    if not parts:
                        break
                    for cl in parts:
                        cl.local_finetune(t, local_epochs, loss_fn)
                    coef = self._coef(parts)
                    self.load_state_dict(sum_lod([cl.state_dict() for cl in parts], coef))
                    self.share_weights()  # push the aggregate so clients report val on it
                    vloss = sum_lod([cl.val_loss(t, loss_fn) for cl in parts], coef)
                    if vloss < best["val"]:
                        best = {"val": vloss, "state": _clone_state(self.state_dict())}
                        stale = 0
                    else:
                        stale += 1
                    if stale >= tol:
                        break
                if best["state"] is not None:
                    self.load_state_dict(best["state"])
                self.share_weights()  # sync clients to the best global weights
                # meta blend: fold the (restored, best) aggregate into W_init.
                if is_meta:
                    if w_init is None:
                        w_init = _clone_state(self.state_dict())
                    else:
                        w = meta_alpha if meta_method == "moving_average" else 1.0 / (t + 1)
                        w_init = sum_lod([w_init, self.state_dict()], [1.0 - w, w])
            else:
                for _ in range(rounds):
                    for cl in self.clients:
                        if cl.can_train(t):
                            cl.local_finetune(t, local_epochs, loss_fn)

            # 3. Refresh each client's carried hidden state (from the aggregate
            #    when FL, from its own weights when local-only).
            #    refresh() deliberately INHERITS the current mode (ROLAND parity), but
            #    nothing guarantees one: local_finetune ends in train mode when it
            #    exhausts local_epochs, and a client that abstains (can_train false, or
            #    every client abstains so the round loop breaks) is never touched by
            #    val_loss/local_finetune at all -- at t=0 it is still in nn.Module's
            #    default train mode. refresh() then hits the encoder BatchNorm with a
            #    1-node batch and raises. Normalize both paths to eval so refresh is
            #    well-defined for every client, and fl=true vs fl=false differ only in
            #    TRAINING rather than in how the carried hs is rebuilt.
            for cl in self.clients:
                cl.classifier.eval()
            for cl in self.clients:
                cl.refresh(t)

            if ckpt_every > 0 and (t + 1) % ckpt_every == 0 and t + 1 < n_tasks:
                self._save_partial_ckpt(t, w_init, mrr_history, metrics_history, t_history)

        mean = statistics.fmean(mrr_history) if mrr_history else None
        std = statistics.pstdev(mrr_history) if len(mrr_history) > 1 else 0.0
        mean_metrics = _weighted_mean_metrics(
            metrics_history, [1.0] * len(metrics_history)
        )
        if log:
            LOGGER.info(
                f"live-update done: mean_mrr={mean} std={std:.4f} "
                f"over {len(mrr_history)} snapshots"
            )
        results = {
            "mean_mrr": mean,
            "std_mrr": std,
            "mrr_history": mrr_history,
            "t_history": t_history,
            "mean_metrics": mean_metrics,
            "metrics_history": metrics_history,
        }
        if ckpt_every > 0:
            self._save_done_ckpt(results)
        return results

    # ---- checkpointing: resume long live-update runs across crashes ---- #

    def _reset_sfvs(self) -> None:
        """Re-draw the learnable SFV for every owner at a snapshot boundary.

        One draw is shared by all owners, mirroring initialize/make_sgraph, where
        every client starts from the SAME init and only then diverges. Drawing
        independently per client would change two things at once (no carry-forward
        AND a different init regime across clients), so the arm would no longer
        isolate accumulation.

        The draw comes off the global torch RNG, which main._seed() seeds, so two
        runs sharing a seed see the same reset sequence.
        """
        w = _get_sfv(self.classifier)
        if w is None:
            return  # smodel owns no SFV (SignNet, f+es Invariant) -- nothing to reset
        fresh = Graph.initialize_random_features(size=tuple(w.shape))
        _set_sfv(self.classifier, fresh)
        for cl in self.clients:
            _set_sfv(cl.classifier, fresh)

    def _run_id(self) -> str:
        """Filesystem-safe identity for this exact run (mirrors the wandb group +
        seed) so a re-launched run finds its own checkpoint and no other."""
        ds = config["dataset"]
        dt = config["model"]["data_type"]
        um = config["spectral"]["update_mode"]
        parts = [
            ds["name"], config["gnn"]["embed_update_method"], str(dt),
            f"C{config['subgraph']['num_subgraphs']}",
        ]
        sp = config["spectral"]
        fed = config["federated"]
        spectral_dt = dt in ("f+s", "structure", "f+pe", "f+es")
        if dt in ("f+s", "structure"):
            # basis_source is the placebo switch -- without it a real arm and its
            # shuffled_fixed control share an identity, and f+s checkpointing works.
            # spectral_len is this path's k; smodel_type selects a different model.
            parts += [f"um-{um}", f"sfv-{fed['sfv_share']}",
                      f"basis-{sp['basis_source']}", f"k{sp['spectral_len']}",
                      f"sm-{config['model']['smodel_type']}"]
        elif dt == "f+pe":
            parts += [f"um-{um}", f"pe{sp['pe_dim']}", f"basis-{sp['basis_source']}"]
        elif dt == "f+es":
            parts += [f"um-{um}", f"pe{sp['pe_dim']}", f"basis-{sp['basis_source']}",
                      f"esf-{sp['es_features']}", f"esp-{sp['es_spec_parts']}"]
        # Procrustes applies on every spectral path under update/recompute, but
        # 'auto' resolves it per data type (off for f+es), so ask for the effective value.
        if spectral_dt and um in ("update", "recompute"):
            # EFFECTIVE value, not the configured one: under 'auto' an f+es run and
            # an f+s run resolve differently, and two runs that differ in whether
            # the basis was rotated must not share a checkpoint.
            parts.append(f"proc-{'on' if _procrustes_on(dt) else 'off'}")
        # Solver changes the basis and so the numbers. Appended only when it is not
        # the default, so existing default-solver identities stay byte-identical.
        if spectral_dt and "solver" in sp and sp["solver"] != "arnoldi":
            parts.append(f"solver-{sp['solver']}")
        # The tracking path's operator changes the basis and so the numbers.
        # Appended only when it is not the default, keeping current identities intact.
        if spectral_dt and "L_type" in sp and sp["L_type"] != "sym":
            parts.append(f"L-{sp['L_type']}")
        # Age kernel on the cumulative adjacency: changes the operator, so the
        # arms must not share a checkpoint. Non-default only, and the parameter
        # only for the kernels that read it, so defaults stay byte-identical.
        if spectral_dt and "cum_decay" in sp and sp["cum_decay"] != "none":
            cd = sp["cum_decay"]
            parts.append(
                f"cum-{cd}" if cd in ("count", "harmonic")
                else f"cum-{cd}{sp['cum_decay_param']}"
            )
        # SFV lifetime: the three arms of the hop2vec ablation (carry / reset /
        # freeze) differ only by these, so without them all three resolve to one
        # identity and, under auto_resume, arms 2 and 3 load arm 1's checkpoint.
        # Appended only when non-default, keeping existing identities byte-identical.
        if _sfv_flag("sfv_reset_per_snapshot"):
            parts.append("sfvreset")
        if _sfv_flag("freeze_sfv"):
            parts.append("sfvfrozen")
        # Federation axes: fl=false is the local-only floor and eval_scope picks the
        # test set, both real experiment axes (they are already wandb group axes).
        # Appended only when non-default so existing identities stay byte-identical.
        # Encoder edge starvation is a real experiment axis: p=0/0.5/0.75 are three
        # different runs and must not share a checkpoint. Appended only when on, so
        # existing identities stay byte-identical.
        gcfg = config["gnn"]
        edrop = gcfg["encoder_edge_drop"] if "encoder_edge_drop" in gcfg else 0.0
        if edrop:
            parts.append(f"edrop{edrop:g}")
        if not fed["fl"]:
            parts.append("local")
        if config["metric"]["eval_scope"] != "auto":
            parts.append(f"eval-{config['metric']['eval_scope']}")
        sf = ds["snapshot_freq"]
        if isinstance(sf, str) and sf.endswith("s") and sf[:-1].isdigit():
            parts.append(f"freq-{sf}")
        # Deterministic runs converge to a different fixed point, so they must not
        # share a checkpoint with non-deterministic ones. Appended ONLY when on, so
        # existing identities stay byte-identical and old checkpoints still load.
        exp = config["experimental"]
        if "deterministic" in exp and exp["deterministic"]:
            parts.append("det")
        # The split draws its OWN negatives from the global RNG, so a split-on run
        # diverges from a split-off one at the same seed and reports a different
        # headline mean_mrr. Without this token the two share an identity: under
        # auto_resume one loads the other's checkpoint, and they collide on the same
        # deterministic wandb id. Appended only when on, so existing identities stay
        # byte-identical and banked checkpoints still load.
        if _repeat_new_split():
            parts.append("split")
        # Coverage control and the window floor change the served basis and so the
        # numbers; arms differing only in these must not share a checkpoint or a
        # deterministic wandb id. Non-default only, so existing identities stay
        # byte-identical and banked checkpoints still resolve.
        if spectral_dt and "coverage_drop" in sp and sp["coverage_drop"]:
            parts.append(f"cov{sp['coverage_drop']:g}")
        if spectral_dt and "window_floor" in sp and sp["window_floor"]:
            parts.append(f"wfl{sp['window_floor']:g}")
        # Injection point changes where S enters the model and so the numbers.
        if spectral_dt and "inject_at" in sp and sp["inject_at"] != "output":
            parts.append(f"inj-{sp['inject_at']}")
        # Completeness backstop: covers every knob no explicit token above does.
        parts.append(f"cfg-{_config_fingerprint()}")
        parts.append(f"s{config['seed']}")
        return "_".join(parts)

    def _wandb_id(self) -> str:
        """Deterministic wandb run id (from run_id) so a resumed run continues the
        SAME wandb run instead of minting a new one. main.py derives the same id."""
        import hashlib
        return hashlib.sha1(self._run_id().encode()).hexdigest()[:20]

    def _ckpt_paths(self):
        d = config["train"]["ckpt_dir"]
        os.makedirs(d, exist_ok=True)
        base = os.path.join(d, self._run_id())
        return base + ".ckpt", base + ".done"

    def _save_partial_ckpt(self, t, w_init, mrr_history, metrics_history, t_history=None):
        ckpt_path, _ = self._ckpt_paths()
        ckpt = {
            "run_id": self._run_id(),
            "wandb_id": self._wandb_id(),  # so resume continues the same wandb run
            "t": t,  # last COMPLETED snapshot; resume at t+1
            "server_state": self.state_dict(),
            "server_sfv": _get_sfv(self.classifier),
            "client_sfv": [_get_sfv(cl.classifier) for cl in self.clients],
            "client_hs": [cl.hs for cl in self.clients],
            "w_init": w_init,
            "first_spectral": self._first_spectral,
            "prev_spectral": self._prev_spectral,
            "cum_edges": self._cum_edges,
            "mrr_history": mrr_history,
            "metrics_history": metrics_history,
            # The TRUE snapshot index per entry. Without it a resumed run cannot
            # know which snapshots the pre-crash segment skipped, and the record
            # for that segment has no usable axis at all. Absent on checkpoints
            # written before this key existed -- those resume with t unknown.
            "t_history": t_history,
            # Without these a resumed run continues on a freshly seeded stream, so
            # it never matches an uninterrupted one -- and experimental.deterministic
            # cannot fix that, since it pins kernels rather than stream position.
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": (torch.cuda.get_rng_state_all()
                         if torch.cuda.is_available() else None),
            },
        }
        tmp = ckpt_path + ".tmp"
        torch.save(ckpt, tmp)
        os.replace(tmp, ckpt_path)  # atomic: a crash mid-save can't corrupt the ckpt
        LOGGER.info(f"checkpoint saved at t={t} -> {ckpt_path}")

    def _load_partial_ckpt(self):
        ckpt_path, _ = self._ckpt_paths()
        if not os.path.exists(ckpt_path):
            return None
        try:
            # map_location='cpu': set_rng_state needs a CPU ByteTensor, and the
            # caller moves hs/model state to the device itself.
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except Exception as e:
            LOGGER.warning(f"could not load checkpoint {ckpt_path}: {e}")
            return None
        if ckpt.get("run_id") != self._run_id():
            return None
        # Restore the streams before anything draws. Absent on checkpoints written
        # before RNG capture existed -- those resume as they always did.
        rng = ckpt.get("rng")
        if rng is not None:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            if rng.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])
        self.load_state_dict(ckpt["server_state"])
        self.share_weights()  # sync clients' fmodel to the restored global weights
        _set_sfv(self.classifier, ckpt.get("server_sfv"))
        for cl, w in zip(self.clients, ckpt.get("client_sfv", [])):
            _set_sfv(cl.classifier, w)  # local SFV isn't in state_dict for sfv_share=local
        for cl, hs in zip(self.clients, ckpt["client_hs"]):
            cl.hs = None if hs is None else [h.to(device) for h in hs]
        self._first_spectral = ckpt["first_spectral"]
        self._prev_spectral = ckpt["prev_spectral"]
        self._cum_edges = ckpt["cum_edges"]
        # A pure function of global_snaps, so rebuild rather than trusting a
        # possibly older checkpoint format to carry it.
        self._rebuild_cum_events(int(ckpt["t"]) if "t" in ckpt else -1)
        # None on a checkpoint predating the key: the axis is genuinely unknown
        # there, and the caller marks it rather than inventing one.
        th = ckpt.get("t_history")
        return (ckpt["t"] + 1, ckpt["w_init"], ckpt["mrr_history"],
                ckpt["metrics_history"], th)

    def _save_done_ckpt(self, results):
        ckpt_path, done_path = self._ckpt_paths()
        done = dict(results)
        done["run_id"] = self._run_id()
        torch.save(done, done_path)
        if config["train"]["ckpt_clean"] and os.path.exists(ckpt_path):
            os.remove(ckpt_path)

    def _load_done_ckpt(self):
        _, done_path = self._ckpt_paths()
        if not os.path.exists(done_path):
            return None
        try:
            done = torch.load(done_path, map_location="cpu", weights_only=False)
        except Exception:
            return None
        return done if done.get("run_id") == self._run_id() else None

    # ---- spectral provider (ported from GNNServer; graph passed explicitly) ---- #

    def _substitute_basis(self, U, Q, ss_idx):
        """spectral.basis_source control: swap the Laplacian eigenbasis for a
        null basis of the SAME shape, to test whether the spectral branch uses
        the graph spectrum or merely a well-conditioned per-node signature.
        'random' = Haar-random orthonormal (QR of a Gaussian); 'shuffled' = the
        real eigenvectors with the node->row assignment permuted (matched value
        distribution and orthonormality, structure destroyed). Applied only where
        the basis is actually (re)computed, so each update_mode keeps its own
        semantics for free: keep freezes the t=0 substitute, recompute re-draws
        one per snapshot, update tracks the substitute via Rayleigh-Ritz.
        D is left untouched (it feeds only the off-by-default regularizer)."""
        src = config["spectral"]["basis_source"]
        if src == "laplacian":
            return U, Q
        # '*_fixed' variants seed by the run seed only, NOT the snapshot index:
        # under recompute the plain nulls are redrawn every solve (temporally
        # unstable), which confounds structure with stability (results.md 10.2b).
        # random_fixed = one constant random basis; shuffled_fixed = one constant
        # permutation of the DRIFTING real basis — matched numerics AND matched
        # temporal drift, structure severed. The strict structure control.
        base = src.removesuffix("_fixed")
        idx = 0 if src.endswith("_fixed") else ss_idx
        g = torch.Generator().manual_seed(config["seed"] * 1000003 + idx)
        if base == "random":
            M = torch.randn(U.shape, generator=g, dtype=torch.float32)
            sub, _ = torch.linalg.qr(M, mode="reduced")
        elif base == "shuffled":
            sub = U.detach().float().cpu()[torch.randperm(U.shape[0], generator=g)]
        else:
            raise ValueError(f"unknown spectral.basis_source={src!r}")
        sub = sub.to(dtype=U.dtype, device=U.device)
        # Q is the tracking basis for `update`; substitute it too so the control
        # stays coherent there (otherwise update would evolve the REAL subspace).
        return sub, (sub if Q is not None else None)

    def _apply_coverage_drop(self, U, Q, ss_idx):
        """spectral.coverage_drop control: zero the served basis rows of a random
        subset of the SOLVER-COVERED nodes, without touching the graph, the age
        kernel, the split, `persist` or `cn`.

        Why it exists. A hard `window` kernel drops nodes whose every edge has aged
        out; they leave the active set, `calc_eigs_*` zero-pads their rows, and a
        pair touching a zero row gets all-zero spectral features -- so the edge-score
        head returns the same constant for every such pair and cannot reorder them.
        Those pairs are ranked by the backbone alone, and the backbone alone scores
        BETTER on new pairs than any spectral arm. A window can therefore raise
        new-pair MRR purely by switching the spectral term off. This control
        reproduces that coverage loss with NO recency change, so the two
        explanations separate.

        Drawn from the covered rows only: sampling over all rows would spend part of
        the budget re-zeroing rows the active-set restriction already zeroed, and the
        arm would not match the window it is meant to match. Returns the served basis
        plus (covered, zeroed) counts for the coverage instrumentation.
        """
        n = U.shape[0]
        covered_mask = U.abs().sum(dim=1) > 0
        covered = int(covered_mask.sum())
        p = config["spectral"]["coverage_drop"] if "coverage_drop" in config["spectral"] else 0.0
        if not p or covered == 0:
            return U, Q, covered, n - covered
        k = int(round(float(p) * covered))
        if k <= 0:
            return U, Q, covered, n - covered
        # seeded by (seed, snapshot): a window's dropped set also changes as edges
        # age out, so the control matches the treatment's temporal character.
        g = torch.Generator().manual_seed(config["seed"] * 7919 + ss_idx)
        cov_idx = torch.nonzero(covered_mask, as_tuple=False).flatten()
        pick = cov_idx[torch.randperm(covered, generator=g)[:k]]
        U = U.clone()
        U[pick] = 0
        if Q is not None:
            Q = Q.clone()
            Q[pick] = 0
        return U, Q, covered, n - covered + k

    def get_previous_UD(
        self, spectral_update_mode: str, ss_idx: int
    ) -> SpectralFeatures:
        empty = SpectralFeatures(U=None, D=None, Q=None)
        if spectral_update_mode == "keep":
            return self._first_spectral or empty
        if ss_idx > 0 and self._prev_spectral is not None:
            return self._prev_spectral
        return empty

    def get_spectral_features(
        self,
        graph: Graph,
        smodel_type,
        ss_idx,
        spectral_len,
        spectral_update_mode,
        prev_spectrals: SpectralFeatures,
        first_spectral: SpectralFeatures,
    ):
        prev_U, prev_D, prev_Q = prev_spectrals.U, prev_spectrals.D, prev_spectrals.Q
        share = {}
        num_spectral_features = None
        solver = config["spectral"]["solver"]

        if smodel_type in ["SpectralLaplace", "LanczosLaplace", "SignNet", "ExactPE", "Invariant"]:
            if (
                spectral_update_mode == "keep"
                and prev_U is not None
                and prev_D is not None
            ):
                LOGGER.info("Keeping previous eigenvectors U and eigenvalues D...")
                D, U, Q = prev_D, prev_U, prev_Q
            elif (
                spectral_update_mode == "update"
                and prev_U is not None
                and prev_D is not None
            ):
                LOGGER.info("Updating spectral features...")
                if solver == "chebyshev":
                    # tracking, done as filtered subspace iteration: the previous
                    # basis warm-starts the block and the previous lambda_k sets
                    # the cutoff. Lands on the exact subspace (overlap 1.000,
                    # results.md §10.12b) where update_eigpairs inherits whatever
                    # the Krylov estimate produced.
                    D, U = graph.calc_eigs_chebyshev(
                        spectral_len, cutoff=_cheb_cutoff(prev_D), X0=prev_U
                    )
                    Q = U
                    U, Q = self._substitute_basis(U, Q, ss_idx)
                else:
                    D, U, Q = graph.update_eigpairs(prev_Q)
            else:
                LOGGER.info("Computing spectral features...")
                if smodel_type == "ExactPE" or solver == "exact":
                    # input-PE path: exact low-k sym-Laplacian eigenpairs — the
                    # Krylov estimate is near-uninformative at the clustered low
                    # end (results.md §10.6), which is the whole signal here.
                    D, U = graph.calc_eigs_exact_sym(spectral_len)
                    Q = U
                elif solver == "chebyshev":
                    D, U = graph.calc_eigs_chebyshev(
                        spectral_len, cutoff=_cheb_cutoff(prev_D)
                    )
                    Q = U
                else:
                    D, U, Q = graph.calc_eignvalues(
                        estimate=not (smodel_type.startswith("Spectral")),
                        spectral_len=spectral_len,
                        log=False,
                        canonicalize_sign=(smodel_type != "SignNet"),
                    )
                assert Q is not None
                U, Q = self._substitute_basis(U, Q, ss_idx)

            if (
                spectral_update_mode in ["recompute", "update"]
                and ss_idx > 0
                and _procrustes_on()
            ):
                U = graph.procrustes_project(U, first_spectral.U)

            # Coverage control + coverage measurement, applied to the FINAL served
            # basis so every branch above (exact, chebyshev, update_eigpairs) is
            # covered, not just the two that route through _substitute_basis.
            U_solved, Q_solved = U, Q
            U, Q, _cov, _zero = self._apply_coverage_drop(U, Q, ss_idx)
            self._basis_coverage = {
                "basis_covered": _cov,
                "basis_zeroed": _zero,
                "basis_total": int(U.shape[0]),
            }
            # Zero-row mask of the SERVED basis, for the pair-level attribution in
            # _eval_mrr. Held separately from the caches below, which must keep the
            # SOLVED basis (see there).
            self._basis_zero_rows = (U.abs().sum(dim=1) == 0).detach().cpu()

            share["D"] = D
            share["U"] = U
            num_spectral_features = D.shape[0]

            # Cache only the ROLAND-minimal set, on CPU: first (keep basis +
            # procrustes anchor) and previous (update tracking). No GPU growth.
            # The SOLVED basis, NOT the coverage-dropped one. coverage_drop is a
            # serve-time mask and nothing else: cached, it would warm-start the
            # next chebyshev solve (X0) or the next Rayleigh-Ritz (prev_Q) and
            # anchor Procrustes, so it would change the SPECTRUM -- the exact
            # confound the control exists to remove -- and it would compound under
            # `keep`, decaying coverage like (1-p)^t instead of holding at p.
            sf = _to_cpu_sf(U_solved, D, Q_solved)
            if ss_idx == 0:
                self._first_spectral = sf
            self._prev_spectral = sf

        return share, num_spectral_features

    def _accumulate_cum_edges(self, t):
        """Advance the cumulative undirected union to include snapshot t.

        _spectral_step does this as a side effect; backbone-only runs need it too
        when metric.repeat_new_split is on, and it must accumulate the same way so
        the two paths agree."""
        e = self.global_snaps[t].edge_index.cpu()
        e = torch.cat([e, e.flip(0)], dim=1)
        if self._cum_edges is not None:
            e = torch.cat([self._cum_edges, e], dim=1)
        self._cum_edges = torch.unique(e, dim=1)
        self._append_cum_events(t)

    def _cum_edge_weight(self, t):
        """Age-kernel weights for the cumulative adjacency at snapshot `t`,
        aligned to `_cum_edges`' COLUMNS. None under the default kernel, which
        leaves the binarizing path in _active_lsym untouched.

            w_t(e) = sum over snapshots s <= t containing e of f(t - s)

        Computed by one scatter-add over the appearance-event record rather than
        a per-kernel recursion: `exp` admits one (w <- gamma*w; w[E_t] += 1) but
        `harmonic` does not, since every existing term's denominator moves each
        step. One exact primitive serves all five kernels.
        """
        sp_cfg = config["spectral"]
        kind = sp_cfg["cum_decay"] if "cum_decay" in sp_cfg else "none"
        if kind == "none" or self._cum_edges is None or self._cum_events_key is None:
            return None
        param = sp_cfg["cum_decay_param"]
        age = (t - self._cum_events_t).to(torch.float64)
        if kind == "count":
            f = torch.ones_like(age)
        elif kind == "harmonic":
            f = 1.0 / (age + 1.0)
        elif kind == "exp":
            f = torch.pow(torch.tensor(float(param), dtype=torch.float64), age)
        elif kind == "window":
            # spectral.window_floor is the weight of an out-of-horizon edge. At the
            # default 0.0 this is the original hard window, bit-identical. Positive
            # keeps every ever-seen edge present, so no node loses its last edge and
            # leaves the active set -- which is what separates a RECENCY change from
            # the COVERAGE change a hard window also makes.
            floor = sp_cfg["window_floor"] if "window_floor" in sp_cfg else 0.0
            inside = (age < float(param)).to(torch.float64)
            f = inside if not floor else inside + (1.0 - inside) * float(floor)
        else:
            raise ValueError(f"unknown spectral.cum_decay={kind!r}")

        uniq, inv = torch.unique(self._cum_events_key, return_inverse=True)
        wu = torch.zeros(uniq.numel(), dtype=torch.float64).scatter_add_(0, inv, f)
        # map per-undirected-pair weights onto both directed columns
        n = self.global_snaps[0].num_nodes
        ce = self._cum_edges.cpu().to(torch.long)
        col = torch.minimum(ce[0], ce[1]) * n + torch.maximum(ce[0], ce[1])
        return wu[torch.searchsorted(uniq, col)]

    def _rebuild_cum_events(self, upto_t):
        """Appearance record for snapshots 0..upto_t. A pure function of
        global_snaps, so resume rebuilds it instead of checkpointing it."""
        self._cum_events_key = None
        self._cum_events_t = None
        for s in range(upto_t + 1):
            self._append_cum_events(s)

    def _append_cum_events(self, t):
        n = self.global_snaps[0].num_nodes
        e = self.global_snaps[t].edge_index.cpu().to(torch.long)
        if e.numel() == 0:
            return
        key = torch.unique(torch.minimum(e[0], e[1]) * n + torch.maximum(e[0], e[1]))
        ts = torch.full_like(key, int(t))
        if self._cum_events_key is None:
            self._cum_events_key, self._cum_events_t = key, ts
        else:
            self._cum_events_key = torch.cat([self._cum_events_key, key])
            self._cum_events_t = torch.cat([self._cum_events_t, ts])

    def _repeat_mask(self, pos_edges):
        """True where a positive pair is already in the cumulative union."""
        if self._cum_edges is None or self._cum_edges.numel() == 0:
            return torch.zeros(pos_edges.size(1), dtype=torch.bool)
        n = self.global_snaps[0].num_nodes
        ce = self._cum_edges.cpu().to(torch.long)
        keys = torch.unique(torch.minimum(ce[0], ce[1]) * n + torch.maximum(ce[0], ce[1]))
        p = pos_edges.cpu().to(torch.long)
        pk = torch.minimum(p[0], p[1]) * n + torch.maximum(p[0], p[1])
        return torch.isin(pk, keys)

    def _spectral_step(self, t, smodel_type):
        # Laplacian over the CUMULATIVE undirected edge union up to t: per-window
        # slices are spectrally degenerate (0-eigenvalue multiplicity from isolated
        # nodes exceeds spectral_len), and history <= t is leakage-free for t+1.
        # The union lives on CPU (the decomposition paths are CPU-bound); the
        # per-owner slices land on `device` via set_QD.
        # ONE accumulator for both paths (this one and the split-only path), so
        # the union and the appearance record can never drift apart.
        self._accumulate_cum_edges(t)
        num_nodes = self.global_snaps[0].num_nodes
        graph_t = Graph(
            x=torch.ones(num_nodes, 1),
            edge_index=self._cum_edges,
            node_ids=torch.arange(num_nodes),
        )
        # Age kernel weights the BASIS only. _repeat_mask (the split) and set_adj
        # ('persist') keep reading the unweighted union below -- if they moved
        # with the treatment, 'repeat' would mean something different in every
        # arm and no arm could be compared with another.
        graph_t.cum_weight = self._cum_edge_weight(t)

        mode = config["spectral"]["update_mode"]
        if mode == "update" and random.random() < config["spectral"]["recompute_prob"]:
            mode = "recompute"  # Bernoulli basis refresh: fresh Lanczos, new Q
        dt = config["model"]["data_type"]
        is_pe = dt == "f+pe"
        if is_pe:
            smodel_type = "ExactPE"
        elif dt == "f+es":
            # invariant edge-score readout: same low-k exact/filtered basis as the
            # probes that measured the effect, and no input scaling (the features
            # are products of rows, so a global scale is absorbed by the MLP)
            smodel_type = "Invariant"
        k = config["spectral"]["pe_dim"] if dt in ("f+pe", "f+es") else config["spectral"]["spectral_len"]
        prev = self.get_previous_UD(mode, t)
        first = self._first_spectral or SpectralFeatures(U=None, D=None, Q=None)
        share, _ = self.get_spectral_features(
            graph_t, smodel_type, t, k, mode, prev, first
        )
        if not share:
            return
        U, D = share["U"], share["D"]
        if is_pe:
            # eigenvector entries are O(1/sqrt(N)); scale to O(1) input features.
            # Applied at serve time (cache stays unscaled) and AFTER any
            # _substitute_basis swap, so the null controls get identical treatment.
            U = U * (U.shape[0] ** 0.5)
        self.classifier.set_QD(U.to(device), D.to(device))
        for cl in self.clients:
            nid = cl.snaps[t].node_ids.cpu()
            cl.classifier.set_QD(U[nid].to(device), D.to(device))

        # persistence control (data_type=f+es): serve the cumulative graph too, so
        # the smodel can read "is this pair already an edge". Clients get their own
        # induced subgraph, remapped to local ids in the same order as their Q rows.
        if hasattr(self.classifier, "set_scale"):
            self.classifier.set_scale(self.global_snaps[0].num_nodes)
            for cl in self.clients:
                cl.classifier.set_scale(self.global_snaps[0].num_nodes)
        if hasattr(self.classifier, "set_adj"):
            n_glob = self.global_snaps[0].num_nodes
            self.classifier.set_adj(self._cum_edges.to(device), n_glob)
            for cl in self.clients:
                nid = cl.snaps[t].node_ids.cpu()
                keep_mask = torch.zeros(n_glob, dtype=torch.bool)
                keep_mask[nid] = True
                e = self._cum_edges
                sel = keep_mask[e[0]] & keep_mask[e[1]]
                remap = torch.full((n_glob,), -1, dtype=torch.long)
                remap[nid] = torch.arange(nid.numel())
                cl.classifier.set_adj(remap[e[:, sel]].to(device), int(nid.numel()))

    def _eval_mrr(self, t, mrr_k, mrr_method):
        zs, ids = [], []
        for cl in self.clients:
            z, nid = cl.encode(t)
            zs.append(z)
            ids.append(nid)
        dim = zs[0].shape[1]
        gz = _stitch_global_z(zs, ids, self.global_snaps[0].num_nodes, dim, device)
        pos_test = _pos_for_split(self.global_snaps[t + 1], "test").to(device)
        if pos_test.size(1) == 0:
            return None, None
        # global_snaps are torch_geometric Data (in-place .to); clone before moving
        # so the persistent snapshot isn't stranded on the GPU (per-snapshot leak).
        eval_snap = _attach_future_link_pred_labels(
            self.global_snaps[t].clone().to(device),
            self.global_snaps[t + 1].clone().to(device),
            pos_test,
        )
        # MRR first, in the server classifier's current mode (unchanged from
        # before metrics were added — keeps the headline reproducible).
        mrr_filter = _mrr_filter_mode()
        mrr = compute_mrr_from_z(
            gz, eval_snap, mrr_k, mrr_method, device, self.classifier,
            "snapshot" if mrr_filter == "snapshot" else "split",
        )
        # Metrics in eval mode (clean BN/dropout, like the local path); restore
        # after so training mode/RNG is untouched. Eval-mode decode is
        # deterministic, so the run stays bit-identical to the MRR-only baseline.
        was_training = self.classifier.model.training
        self.classifier.eval()
        with torch.no_grad():
            pred, label = self.classifier.decode(gz, eval_snap)
            metrics = binary_classification_metrics(pred, label)
            if config["metric"]["hard_neg"] != "random":
                # de-saturate auc/ap with hard (degree-weighted) negatives; overrides the
                # ~1:1 random-negative roc_auc/ap. mrr + f1/mcc stay on the original set.
                # 1 hard negative per positive (1:1) so auc AND ap stay comparable to the
                # easy 1:1 baseline (K:1 would crush ap via the positive base-rate).
                h_auc, h_ap = compute_hard_auc_ap_from_z(gz, eval_snap, 1, device, self.classifier)
                metrics["roc_auc"], metrics["ap"] = h_auc, h_ap
        self.classifier.train(was_training)
        if mrr_filter == "both":
            # paired: same model state, same snapshot, same seed — only the
            # forbidden set differs, so the delta is the filter and nothing else.
            metrics["mrr_snapshot"] = compute_mrr_from_z(
                gz, eval_snap, mrr_k, mrr_method, device, self.classifier, "snapshot"
            )
        if _repeat_new_split():
            pm = eval_snap.edge_label == 1.0
            pos_edges = eval_snap.edge_label_index[:, pm]
            rmask = self._repeat_mask(pos_edges)
            _, m_rep, m_new = compute_mrr_splits_from_z(
                gz, eval_snap, mrr_k, mrr_method, device, self.classifier, rmask,
                "snapshot" if mrr_filter == "snapshot" else "split",
            )
            metrics["mrr_repeat"], metrics["mrr_new"] = m_rep, m_new
            metrics["repeat_frac"] = float(rmask.float().mean()) if rmask.numel() else float("nan")
            # Sample counts for the two subsets. mrr_repeat/mrr_new are means over
            # SOURCES (a source with no positive in a subset is skipped, not zeroed),
            # so src_* is the actual denominator and n_* is the positive count behind
            # it. Both are needed to re-weight or bootstrap a run-level mrr_new, and
            # to see whether an effect rides on a handful of snapshots -- neither was
            # recoverable before (results.md §20.6). Derived from rmask here rather
            # than returned from mrr.py, so _rank_and_aggregate's signature stays
            # stable for analysis/probes/ that import it.
            # _repeat_mask returns CPU while pos_edges lives on `device`; index a
            # CUDA tensor with a CPU mask and you are relying on an implicit
            # transfer. compute_mrr_splits_from_z already moves it explicitly for
            # the same reason. This box is CPU-only, so the mismatch cannot be
            # exercised here -- move it rather than trust the untested path.
            src = pos_edges[0]
            rb = rmask.to(src.device).bool()
            metrics["n_repeat"] = int(rb.sum())
            metrics["n_new"] = int((~rb).sum())
            metrics["src_repeat"] = int(torch.unique(src[rb]).numel()) if rb.any() else 0
            metrics["src_new"] = int(torch.unique(src[~rb]).numel()) if (~rb).any() else 0
        # Basis coverage of the SERVED basis. The pair-level fraction is the one that
        # matters for attribution: a pair touching an all-zero row gets a constant
        # edge-score contribution and is ordered by the backbone alone, so a change in
        # a subset metric cannot be split between "ranked differently" and "the term
        # stopped applying" without it. Absent for the backbone, so runs predating
        # these fields are unaffected.
        # ||S|| / ||h_last||: the gate on reading any injection result (see
        # FedDynamicClassifier._record_inject_scale). Absent when no smodel is served.
        # Recorded on the CLIENT classifiers: the global eval stitches per-client
        # embeddings, so the server's own encode is not what produced them.
        _isc = [v for v in (getattr(c.classifier, "inject_scale", None) for c in self.clients)
                if isinstance(v, float) and v == v]
        if not _isc:
            _s = getattr(self.classifier, "inject_scale", None)
            if isinstance(_s, float) and _s == _s: _isc = [_s]
        if _isc:
            metrics["inject_scale"] = sum(_isc) / len(_isc)
        if self._basis_coverage is not None:
            metrics.update(self._basis_coverage)
            eli = eval_snap.edge_label_index
            # the mask of the basis SERVED at this snapshot (set_QD ran for t in
            # _spectral_step, just above this eval). NOT _prev_spectral, which
            # holds the pre-drop solved basis for the tracker.
            zero_row = self._basis_zero_rows
            if zero_row is not None and eli.numel():
                zero_row = zero_row.to(eli.device)
                touched = zero_row[eli[0]] | zero_row[eli[1]]
                metrics["basis_zeroed_pair_frac"] = float(touched.float().mean())
            else:
                metrics["basis_zeroed_pair_frac"] = float("nan")
        return mrr, metrics

    def _eval_mrr_local(self, t, loss_fn, mrr_k, mrr_method):
        vals, metrics_list, weights = [], [], []
        for cl in self.clients:
            if _pos_for_split(cl.snaps[t + 1], "test").size(1) == 0:
                continue
            _, mrr, metrics = _step_eval_with_mrr_pair(
                cl.classifier, cl.snaps[t], cl.snaps[t + 1], cl._hs_in(),
                loss_fn, device, True, mrr_k, mrr_method,
            )
            if not math.isnan(mrr):
                vals.append(mrr)
                metrics_list.append(metrics)
                weights.append(cl.num_nodes())
        if not vals:
            return None, None
        mrr = sum(v * w for v, w in zip(vals, weights)) / sum(weights)
        return mrr, _weighted_mean_metrics(metrics_list, weights)
