import math
import random
import statistics
from dataclasses import dataclass

from src import *
from src.server import Server
from src.utils.graph import Graph
from src.dynamic_client import DynamicClient
from src.GNN.dynamic_classifier import DynamicClassifier
from src.GNN.fed_dynamic_classifier import make_fed_dynamic_classifier
from src.metrics.mrr import compute_mrr_from_z
from src.metrics.classification import binary_classification_metrics
from src.train.federated_orchestrator import (
    _partition_edges_per_snapshot,
    _precompute_keep_ratio,
    _pos_for_split,
    _attach_future_link_pred_labels,
    _step_eval_with_mrr_pair,
    _stitch_global_z,
    _clone_state,
)
from registries import losses


@dataclass
class SpectralFeatures:
    U: torch.Tensor
    D: torch.Tensor
    Q: torch.Tensor | None = None


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
        self.stored_spectrals = dict[int, SpectralFeatures]()
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
        use_spectral = dt in ("structure", "f+s")
        self.stored_spectrals.clear()  # runs are independent (fresh W, fresh Bernoulli draws)
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
        if log:
            tag = model_type or ("FL WA" if FL else "Local WA")
            LOGGER.info(f"{tag} live-update starts!")

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

        mrr_history: list[float] = []
        metrics_history: list[dict] = []
        w_init = None  # running meta W_init (moving-avg across snapshots; FL path)
        for t in range(n_tasks):
            # 0. Spectral provider: U_t on the cumulative global graph, sliced to
            #    every owner via set_QD (before eval — encoding snap_t needs Q_t).
            if use_spectral:
                self._spectral_step(t, smt)

            # 1. Reported eval. FL: clients hold the global weights (share_weights
            #    at initialize_FL / end of the previous snapshot) -> global stitch.
            #    Local-only: weighted mean of per-client local MRRs.
            if FL:
                mrr, metrics = self._eval_mrr(t, mrr_k, mrr_method)
            else:
                mrr, metrics = self._eval_mrr_local(t, loss_fn, mrr_k, mrr_method)
            if mrr is not None and not math.isnan(mrr):
                mrr_history.append(mrr)
                if metrics is not None:
                    metrics_history.append(metrics)
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
            for cl in self.clients:
                cl.refresh(t)

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
        return {
            "mean_mrr": mean,
            "std_mrr": std,
            "mrr_history": mrr_history,
            "mean_metrics": mean_metrics,
            "metrics_history": metrics_history,
        }

    # ---- spectral provider (ported from GNNServer; graph passed explicitly) ---- #

    def get_previous_UD(
        self, spectral_update_mode: str, ss_idx: int
    ) -> SpectralFeatures:
        prev_spectrals = SpectralFeatures(U=None, D=None, Q=None)

        if spectral_update_mode in ["keep"]:
            if len(self.stored_spectrals) > 0:
                first_index = min(self.stored_spectrals.keys())
                prev_spectrals = self.stored_spectrals[first_index]
        else:
            if ss_idx - 1 in self.stored_spectrals and ss_idx > 0:
                prev_spectrals = self.stored_spectrals[ss_idx - 1]

        return prev_spectrals

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

        if smodel_type in ["SpectralLaplace", "LanczosLaplace"]:
            if (
                spectral_update_mode == "keep"
                and prev_U is not None
                and prev_D is not None
            ):
                LOGGER.info("Keeping previous eigenvectors U and eigenvalues D...")
                D, U = prev_D, prev_U
            elif (
                spectral_update_mode == "update"
                and prev_U is not None
                and prev_D is not None
            ):
                if ss_idx not in self.stored_spectrals:
                    LOGGER.info("Updating spectral features...")
                    D, U, Q = graph.update_eigpairs(prev_Q)
                    self.stored_spectrals[ss_idx] = SpectralFeatures(U=U, D=D, Q=Q)
                else:
                    stored = self.stored_spectrals[ss_idx]
                    D, U, Q = stored.D, stored.U, stored.Q
            else:
                if ss_idx not in self.stored_spectrals:
                    LOGGER.info("Computing spectral features...")
                    D, U, Q = graph.calc_eignvalues(
                        estimate=not (smodel_type.startswith("Spectral")),
                        spectral_len=spectral_len,
                        log=False,
                    )
                    assert Q is not None
                    self.stored_spectrals[ss_idx] = SpectralFeatures(U=U, D=D, Q=Q)
                else:
                    stored = self.stored_spectrals[ss_idx]
                    D, U, Q = stored.D, stored.U, stored.Q

            if (
                spectral_update_mode in ["recompute", "update"]
                and ss_idx > 0
                and config["spectral"]["use_procrustes"]
            ):
                U = graph.procrustes_project(U, first_spectral.U)

            share["D"] = D
            share["U"] = U
            num_spectral_features = D.shape[0]

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
        prev = self.get_previous_UD(mode, t)
        first = self.stored_spectrals.get(0, SpectralFeatures(U=None, D=None, Q=None))
        share, _ = self.get_spectral_features(
            graph_t, smodel_type, t, config["spectral"]["spectral_len"], mode, prev, first
        )
        if not share:
            return
        U, D = share["U"], share["D"]
        self.classifier.set_QD(U.to(device), D.to(device))
        for cl in self.clients:
            nid = cl.snaps[t].node_ids.cpu()
            cl.classifier.set_QD(U[nid].to(device), D.to(device))
        # Bound GPU memory: only snapshot 0 (procrustes/keep reference) and the last
        # two (prev for tracking, current) are ever read again. Without this,
        # stored_spectrals retains every snapshot's U/D/Q and OOMs on long series.
        for k in [k for k in self.stored_spectrals if k not in (0, t - 1, t)]:
            del self.stored_spectrals[k]

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
        eval_snap = _attach_future_link_pred_labels(
            self.global_snaps[t].to(device), self.global_snaps[t + 1].to(device), pos_test
        )
        # MRR first, in the server classifier's current mode (unchanged from
        # before metrics were added — keeps the headline reproducible).
        mrr = compute_mrr_from_z(gz, eval_snap, mrr_k, mrr_method, device, self.classifier)
        # Metrics in eval mode (clean BN/dropout, like the local path); restore
        # after so training mode/RNG is untouched. Eval-mode decode is
        # deterministic, so the run stays bit-identical to the MRR-only baseline.
        was_training = self.classifier.model.training
        self.classifier.eval()
        with torch.no_grad():
            pred, label = self.classifier.decode(gz, eval_snap)
        metrics = binary_classification_metrics(pred, label)
        self.classifier.train(was_training)
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
