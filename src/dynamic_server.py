import os
import math
import random
import statistics
from dataclasses import dataclass

from src import *
from src.server import Server
from src.utils.graph import Graph
from src.dynamic_client import DynamicClient
from src.GNN.dynamic_classifier import DynamicClassifier
from src.GNN.fed_dynamic_classifier import (
    FedDynamicEdgeScoreClassifier,
    FedDynamicPEClassifier,
    make_fed_dynamic_classifier,
)
from src.metrics.mrr import compute_mrr_from_z, compute_hard_auc_ap_from_z
from src.metrics.classification import binary_classification_metrics
from src.train.federated_orchestrator import (
    _partition_edges_per_snapshot,
    _precompute_keep_ratio,
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


def _get_sfv(clf):
    """The learnable SFV W (smodel.graph.x) — a leaf param that is NOT in the
    state_dict when federated.sfv_share='local', so checkpoints capture it here."""
    smodel = getattr(clf, "smodel", None)
    if smodel is None or smodel.graph.x is None:
        return None
    return smodel.graph.x.detach().cpu()


def _set_sfv(clf, w):
    smodel = getattr(clf, "smodel", None)
    if smodel is None or w is None:
        return
    with torch.no_grad():
        smodel.graph.x.copy_(w.to(smodel.graph.x.device))


def _weighted_mean_metrics(
    metrics_list: list[dict], weights: list[float]
) -> dict[str, float]:
    """Per-key weighted mean over a list of metric dicts, skipping nan
    contributions per key (a key with all-nan or zero weight -> nan)."""
    if not metrics_list:
        return {}
    out = {}
    for k in metrics_list[0]:
        num, den = 0.0, 0.0
        for md, w in zip(metrics_list, weights):
            v = md[k]
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

        n_tasks = len(self.global_snaps) - 1
        if n_tasks < 1:
            raise ValueError(
                f"federated live_update needs >= 2 snapshots; got {len(self.global_snaps)}"
            )

        # Checkpointing (gated on train.auto_resume): dump resumable state every
        # ckpt_period snapshots; on restart skip already-done runs or resume mid-run.
        ckpt_every = config["train"]["ckpt_period"] if config["train"]["auto_resume"] else 0

        mrr_history: list[float] = []
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
                t_start, w_init, mrr_history, metrics_history = resumed
                if log:
                    LOGGER.info(f"{self._run_id()}: resuming from t={t_start}")
        for t in range(t_start, n_tasks):
            # 0. Spectral provider: U_t on the cumulative global graph, sliced to
            #    every owner via set_QD (before eval — encoding snap_t needs Q_t).
            if use_spectral:
                self._spectral_step(t, smt)

            # 1. Reported eval. FL: clients hold the global weights (share_weights
            #    at initialize_FL / end of the previous snapshot) -> global stitch.
            #    Local-only: weighted mean of per-client local MRRs.
            if global_eval:
                mrr, metrics = self._eval_mrr(t, mrr_k, mrr_method)
            else:
                mrr, metrics = self._eval_mrr_local(t, loss_fn, mrr_k, mrr_method)
            if mrr is not None and not math.isnan(mrr):
                mrr_history.append(mrr)
                if metrics is not None:
                    metrics_history.append(metrics)
                if log_cb is not None:  # live per-snapshot wandb (resumable run)
                    log_cb(t, mrr, metrics)
                if log:
                    m = metrics or {}
                    LOGGER.info(
                        f"t={t} mrr={mrr:.4f} auc={m.get('roc_auc', float('nan')):.4f} "
                        f"ap={m.get('ap', float('nan')):.4f} f1={m.get('f1', float('nan')):.4f} "
                        f"mcc={m.get('mcc', float('nan')):.4f}"
                    )

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
                self._save_partial_ckpt(t, w_init, mrr_history, metrics_history)

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
            "mean_metrics": mean_metrics,
            "metrics_history": metrics_history,
        }
        if ckpt_every > 0:
            self._save_done_ckpt(results)
        return results

    # ---- checkpointing: resume long live-update runs across crashes ---- #

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
        if dt in ("f+s", "structure"):
            parts += [f"um-{um}", f"sfv-{config['federated']['sfv_share']}"]
            if um in ("update", "recompute"):
                parts.append(f"proc-{'on' if config['spectral']['use_procrustes'] else 'off'}")
        elif dt == "f+pe":
            parts += [f"um-{um}", f"pe{config['spectral']['pe_dim']}",
                      f"basis-{config['spectral']['basis_source']}"]
        sf = ds["snapshot_freq"]
        if isinstance(sf, str) and sf.endswith("s") and sf[:-1].isdigit():
            parts.append(f"freq-{sf}")
        # Deterministic runs converge to a different fixed point, so they must not
        # share a checkpoint with non-deterministic ones. Appended ONLY when on, so
        # existing identities stay byte-identical and old checkpoints still load.
        exp = config["experimental"]
        if "deterministic" in exp and exp["deterministic"]:
            parts.append("det")
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

    def _save_partial_ckpt(self, t, w_init, mrr_history, metrics_history):
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
            ckpt = torch.load(ckpt_path, weights_only=False)  # our own trusted ckpt
        except Exception as e:
            LOGGER.warning(f"could not load checkpoint {ckpt_path}: {e}")
            return None
        if ckpt.get("run_id") != self._run_id():
            return None
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
        return ckpt["t"] + 1, ckpt["w_init"], ckpt["mrr_history"], ckpt["metrics_history"]

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
                and config["spectral"]["use_procrustes"]
            ):
                U = graph.procrustes_project(U, first_spectral.U)

            share["D"] = D
            share["U"] = U
            num_spectral_features = D.shape[0]

            # Cache only the ROLAND-minimal set, on CPU: first (keep basis +
            # procrustes anchor) and previous (update tracking). No GPU growth.
            sf = _to_cpu_sf(U, D, Q)
            if ss_idx == 0:
                self._first_spectral = sf
            self._prev_spectral = sf

        return share, num_spectral_features

    def _spectral_step(self, t, smodel_type):
        # Laplacian over the CUMULATIVE undirected edge union up to t: per-window
        # slices are spectrally degenerate (0-eigenvalue multiplicity from isolated
        # nodes exceeds spectral_len), and history <= t is leakage-free for t+1.
        # The union lives on CPU (the decomposition paths are CPU-bound); the
        # per-owner slices land on `device` via set_QD.
        e = self.global_snaps[t].edge_index.cpu()
        e = torch.cat([e, e.flip(0)], dim=1)
        if self._cum_edges is not None:
            e = torch.cat([self._cum_edges, e], dim=1)
        self._cum_edges = torch.unique(e, dim=1)
        num_nodes = self.global_snaps[0].num_nodes
        graph_t = Graph(
            x=torch.ones(num_nodes, 1),
            edge_index=self._cum_edges,
            node_ids=torch.arange(num_nodes),
        )

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
